#!/usr/bin/env python3
"""
LARGE Model Training - Uses full GPU capacity.
Target: Loss < 2.5, real word generation.
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
        raise RuntimeError("❌ NO GPU!")
    
    device = torch.device('cuda')
    props = torch.cuda.get_device_properties(0)
    print(f"✅ GPU: {props.name}")
    print(f"✅ VRAM: {props.total_memory / 1e9:.1f} GB")
    return device


# ==================== DATA LOADING ====================
def load_wikitext(path='wikitext_en.txt', max_sentences=500000):
    """Load WikiText data."""
    if not os.path.exists(path):
        print(f"⚠️  {path} not found")
        return []
    
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 30 <= len(line) <= 120:  # Slightly longer sentences
                sentences.append(line)
                if len(sentences) >= max_sentences:
                    break
    
    print(f"✓ Loaded {len(sentences):,} sentences")
    return sentences


def text_to_tensor(text, max_len, device):
    """Convert text to byte tensor."""
    safe_bytes = text.encode('utf-8', errors='replace')[:max_len]
    byte_vals = [min(b, 255) for b in safe_bytes]
    if len(byte_vals) < max_len:
        byte_vals = byte_vals + [0] * (max_len - len(byte_vals))
    return torch.tensor(byte_vals, dtype=torch.long, device=device)


# ==================== LARGE BATCH TRAINING ====================
def train_epoch(brain, sentences, optimizer, scaler, device, batch_size=128, max_len=80):
    """Train one epoch with large batches."""
    brain.train()
    
    random.shuffle(sentences)
    total_loss = 0
    total_recon = 0
    total_next = 0
    num_batches = 0
    
    # Process all data
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
        
        with autocast('cuda'):
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
        
        # Update progress bar
        if num_batches % 10 == 0:
            pbar.set_postfix({
                'loss': f'{total_loss/num_batches:.3f}',
                'gpu': f'{torch.cuda.memory_allocated()/1e9:.1f}GB'
            })
    
    if num_batches == 0:
        return 0, 0, 0
    
    return total_loss / num_batches, total_recon / num_batches, total_next / num_batches


def test_generation(brain, prompts, device):
    """Test generation."""
    brain.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                tensor = text_to_tensor(prompt, 80, device).unsqueeze(0)
                sem, _ = brain.comprehend(tensor, store_in_memory=False)
                output = brain.produce(sem, max_length=50)
                out_bytes = output[0].cpu().numpy()
                out_text = bytes(b for b in out_bytes if 32 <= b < 127).decode('utf-8', errors='replace')
                results.append((prompt, out_text[:60]))
            except Exception as e:
                results.append((prompt, f"[error: {e}]"))
    
    return results


# ==================== MAIN ====================
def main():
    print("\n" + "=" * 70)
    print("🚀 LARGE MODEL TRAINING")
    print("=" * 70)
    
    device = check_gpu()
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    sentences = load_wikitext()
    if not sentences:
        print("❌ No data!")
        return
    
    # LARGE CONFIG
    print("\n" + "=" * 70)
    print("🧠 Initializing LARGE model...")
    print("=" * 70)
    
    config = {
        'd_pattern': 512,      # 2x larger
        'd_semantic': 1024,    # 2x larger  
        'd_hidden': 1024,      # 2x larger
        'num_concepts': 5000,  # 5x larger
        'memory_capacity': 50000,
    }
    
    brain = BrainLanguageSystem(config).to(device)
    
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"✅ Parameters: {n_params:,} (~{n_params/1e6:.0f}M)")
    
    # Optimizer
    optimizer = optim.AdamW(brain.parameters(), lr=2e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
    scaler = GradScaler('cuda')
    
    # Config
    BATCH_SIZE = 128  # Larger batches for GPU utilization
    
    print(f"\n{'=' * 70}")
    print(f"🚀 TRAINING CONFIG")
    print(f"   Model: {n_params/1e6:.0f}M params")
    print(f"   Batch size: {BATCH_SIZE}")
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
            batch_size=BATCH_SIZE
        )
        scheduler.step()
        
        elapsed = time.time() - start_time
        remaining = max(0, end_time - time.time())
        epoch_time = time.time() - epoch_start
        gpu_mem = torch.cuda.memory_allocated() / 1e9
        
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
            }, f'checkpoints/LARGE_e{epoch}.pth')
            torch.save(brain.state_dict(), 'checkpoints/LARGE_latest.pth')
            
            with open('training_LARGE.json', 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"💾 Saved checkpoint")
        
        # Generation test every 25 epochs
        if epoch % 25 == 0:
            print("\n🎯 GENERATION:")
            for prompt, output in test_generation(brain, ["the cat", "hello world", "today"], device):
                print(f"  '{prompt}' → '{output}'")
    
    # Final
    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"   Epochs: {epoch}")
    print(f"   Final: {loss:.4f}")
    print(f"   Best: {best_loss:.4f}")
    print(f"{'=' * 70}")
    
    torch.save(brain.state_dict(), 'checkpoints/LARGE_final.pth')


if __name__ == "__main__":
    main()
