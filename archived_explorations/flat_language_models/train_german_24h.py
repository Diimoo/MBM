#!/usr/bin/env python3
"""
24-Hour German Training with quality datasets.
Uses: TinyStories German, BabyLM German, DBMDZ corpus, Nanochat wordlist
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import time
import json
import random
import os

from digital_brain.modules.brain_language import (
    BrainLanguageSystem,
    create_brain_language_config
)

# ==================== GPU CHECK ====================
def check_gpu():
    if not torch.cuda.is_available():
        print("⚠️  No GPU, using CPU")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    print(f"✅ GPU: {props.name}")
    print(f"✅ VRAM: {props.total_memory / 1e9:.1f} GB")
    return device


# ==================== DATA LOADING ====================
def load_german_datasets():
    """Load German datasets from Hugging Face."""
    from datasets import load_dataset
    
    all_sentences = []
    
    # 1. TinyStories German - Simple stories, great for language learning
    print("\n[1/4] Loading TinyStories German...")
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train")
        for item in tqdm(ds, desc="TinyStories"):
            text = item.get('text', '') or item.get('story', '') or str(item)
            # Split into sentences
            for sent in text.replace('\n', ' ').split('.'):
                sent = sent.strip()
                if 20 <= len(sent) <= 200:
                    all_sentences.append(sent)
        print(f"  ✓ TinyStories: {len(all_sentences):,} sentences")
    except Exception as e:
        print(f"  ⚠️ TinyStories failed: {e}")
    
    count_after_tiny = len(all_sentences)
    
    # 2. BabyLM German - Child-directed speech
    print("\n[2/4] Loading BabyLM German...")
    try:
        ds = load_dataset("bbunzeck/babylm-german", split="train")
        for item in tqdm(ds, desc="BabyLM"):
            text = item.get('text', '') or str(item)
            for sent in text.replace('\n', ' ').split('.'):
                sent = sent.strip()
                if 20 <= len(sent) <= 200:
                    all_sentences.append(sent)
        print(f"  ✓ BabyLM: {len(all_sentences) - count_after_tiny:,} sentences")
    except Exception as e:
        print(f"  ⚠️ BabyLM failed: {e}")
    
    count_after_baby = len(all_sentences)
    
    # 3. DBMDZ German BERT corpus - Wikipedia/news
    print("\n[3/4] Loading DBMDZ German corpus...")
    try:
        ds = load_dataset("stefan-it/german-dbmdz-bert-corpus", split="train", streaming=True)
        count = 0
        for item in tqdm(ds, desc="DBMDZ", total=500000):
            text = item.get('text', '') or str(item)
            for sent in text.replace('\n', ' ').split('.'):
                sent = sent.strip()
                if 20 <= len(sent) <= 200:
                    all_sentences.append(sent)
                    count += 1
            if count >= 500000:  # Limit to 500k sentences
                break
        print(f"  ✓ DBMDZ: {len(all_sentences) - count_after_baby:,} sentences")
    except Exception as e:
        print(f"  ⚠️ DBMDZ failed: {e}")
    
    count_after_dbmdz = len(all_sentences)
    
    # 4. Nanochat German wordlist - vocabulary
    print("\n[4/4] Loading Nanochat German wordlist...")
    try:
        ds = load_dataset("stefan-it/nanochat-german-wordlist", split="train")
        for item in tqdm(ds, desc="Nanochat"):
            text = item.get('text', '') or item.get('word', '') or str(item)
            text = text.strip()
            if 3 <= len(text) <= 50:  # Words/short phrases
                all_sentences.append(text)
        print(f"  ✓ Nanochat: {len(all_sentences) - count_after_dbmdz:,} items")
    except Exception as e:
        print(f"  ⚠️ Nanochat failed: {e}")
    
    # Shuffle and deduplicate
    all_sentences = list(set(all_sentences))
    random.shuffle(all_sentences)
    
    print(f"\n{'='*60}")
    print(f"✅ Total unique sentences: {len(all_sentences):,}")
    print(f"{'='*60}")
    
    return all_sentences


def text_to_tensor(text, max_len, device):
    """Convert text to byte tensor."""
    safe_bytes = text.encode('utf-8', errors='replace')[:max_len]
    byte_vals = [min(b, 255) for b in safe_bytes]
    if len(byte_vals) < max_len:
        byte_vals = byte_vals + [0] * (max_len - len(byte_vals))
    return torch.tensor(byte_vals, dtype=torch.long, device=device)


# ==================== TRAINING ====================
def train_epoch(brain, sentences, optimizer, scaler, device, batch_size=256, max_len=128):
    """Train one epoch with large batches."""
    brain.train()
    
    random.shuffle(sentences)
    total_loss = 0
    total_recon = 0
    total_next = 0
    num_batches = 0
    
    pbar = tqdm(range(0, len(sentences), batch_size), desc="Training", leave=False)
    for i in pbar:
        batch_sentences = sentences[i:i+batch_size]
        if len(batch_sentences) < 8:
            continue
        
        # Convert to tensors
        batch_tensors = []
        for sent in batch_sentences:
            try:
                tensor = text_to_tensor(sent, max_len, device)
                if tensor.sum() > 0:
                    batch_tensors.append(tensor)
            except:
                continue
        
        if len(batch_tensors) < 8:
            continue
        
        batch_tensor = torch.stack(batch_tensors)
        input_bytes = batch_tensor[:, :-1]
        target_bytes = batch_tensor[:, 1:]
        
        optimizer.zero_grad()
        
        with autocast('cuda' if device.type == 'cuda' else 'cpu'):
            # Comprehend
            semantic, _ = brain.comprehend(input_bytes, store_in_memory=False)
            
            # Production loss
            logits = brain.produce(semantic, target_bytes=input_bytes)
            next_loss = F.cross_entropy(
                logits.reshape(-1, 256),
                target_bytes.reshape(-1),
                ignore_index=0,
                label_smoothing=0.1
            )
            
            # Reconstruction loss
            recon_logits = brain.reconstruct_input(semantic, seq_len=input_bytes.shape[1])
            recon_loss = F.cross_entropy(
                recon_logits.reshape(-1, 256),
                input_bytes.reshape(-1),
                ignore_index=0
            )
            
            # Combined
            loss = next_loss + 2.0 * recon_loss
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_next += next_loss.item()
        num_batches += 1
        
        if num_batches % 20 == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.3f}',
                'gpu': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
    
    if num_batches == 0:
        return 0, 0, 0
    
    return total_loss / num_batches, total_recon / num_batches, total_next / num_batches


def test_generation(brain, prompts, device, max_len=100):
    """Test generation with longer outputs."""
    brain.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                tensor = text_to_tensor(prompt, 128, device).unsqueeze(0)
                sem, _ = brain.comprehend(tensor, store_in_memory=False)
                output = brain.produce(sem, max_length=max_len)
                out_bytes = output[0].cpu().numpy()
                out_text = bytes(b for b in out_bytes if 32 <= b < 127 or b > 160).decode('utf-8', errors='replace')
                results.append((prompt, out_text[:120]))
            except Exception as e:
                results.append((prompt, f"[error: {e}]"))
    
    return results


# ==================== MAIN ====================
def main():
    print("\n" + "=" * 70)
    print("🇩🇪 24-HOUR GERMAN TRAINING")
    print("=" * 70)
    
    device = check_gpu()
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load German datasets
    print("\n📚 Loading German datasets...")
    sentences = load_german_datasets()
    
    if len(sentences) < 1000:
        print("❌ Not enough data!")
        return
    
    # Save data for reference
    with open('german_sentences.txt', 'w', encoding='utf-8') as f:
        for sent in sentences[:10000]:
            f.write(sent + '\n')
    print(f"💾 Saved sample to german_sentences.txt")
    
    # Model config - LARGE
    print("\n" + "=" * 70)
    print("🧠 Initializing model...")
    print("=" * 70)
    
    config = {
        'd_pattern': 512,
        'd_semantic': 1024,
        'd_hidden': 1024,
        'num_concepts': 5000,
        'memory_capacity': 50000,
    }
    
    brain = BrainLanguageSystem(config).to(device)
    
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"✅ Parameters: {n_params:,} (~{n_params/1e6:.0f}M)")
    
    # Optimizer
    optimizer = optim.AdamW(brain.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # Config
    BATCH_SIZE = 256  # Large batches
    MAX_LEN = 128     # Longer sequences
    
    print(f"\n{'=' * 70}")
    print(f"🚀 TRAINING CONFIG")
    print(f"   Model: {n_params/1e6:.0f}M params")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Max length: {MAX_LEN}")
    print(f"   Sentences: {len(sentences):,}")
    print(f"   Batches/epoch: {len(sentences) // BATCH_SIZE}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    end_time = start_time + 24 * 3600
    
    best_loss = float('inf')
    metrics = []
    
    epoch = 0
    while time.time() < end_time:
        epoch += 1
        epoch_start = time.time()
        
        loss, recon, next_l = train_epoch(
            brain, sentences, optimizer, scaler, device,
            batch_size=BATCH_SIZE, max_len=MAX_LEN
        )
        scheduler.step()
        
        elapsed = time.time() - start_time
        remaining = max(0, end_time - time.time())
        epoch_time = time.time() - epoch_start
        gpu_mem = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
        
        m = {
            'epoch': epoch,
            'loss': loss,
            'recon': recon,
            'next': next_l,
            'gpu_gb': gpu_mem,
            'elapsed_h': elapsed / 3600,
        }
        metrics.append(m)
        
        # Log every epoch
        print(f"\n{'=' * 60}")
        print(f"📊 EPOCH {epoch}")
        print(f"{'=' * 60}")
        print(f"Loss:      {loss:.4f} (recon: {recon:.4f}, next: {next_l:.4f})")
        print(f"GPU:       {gpu_mem:.2f} GB")
        print(f"LR:        {scheduler.get_last_lr()[0]:.6f}")
        print(f"Time:      {elapsed/3600:.2f}h elapsed, {remaining/3600:.1f}h left")
        print(f"Speed:     {epoch_time:.1f}s/epoch")
        
        if loss < best_loss:
            best_loss = loss
            print(f"✅ NEW BEST: {best_loss:.4f}")
        
        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model': brain.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'best': best_loss,
            }, f'checkpoints/german_e{epoch}.pth')
            torch.save(brain.state_dict(), 'checkpoints/german_latest.pth')
            
            with open('training_german.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Generation test every 20 epochs
        if epoch % 20 == 0:
            print("\n🎯 GENERATION (longer outputs):")
            prompts = [
                "Die Katze",
                "Es war einmal",
                "Der kleine",
                "Heute ist",
            ]
            for prompt, output in test_generation(brain, prompts, device, max_len=100):
                print(f"  '{prompt}' → '{output}'")
    
    # Final
    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"   Epochs: {epoch}")
    print(f"   Final: {loss:.4f}")
    print(f"   Best: {best_loss:.4f}")
    print(f"{'=' * 70}")
    
    torch.save(brain.state_dict(), 'checkpoints/german_final.pth')


if __name__ == "__main__":
    main()
