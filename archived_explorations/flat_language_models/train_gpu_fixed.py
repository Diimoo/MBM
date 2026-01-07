#!/usr/bin/env python3
"""
FIXED 24-Hour GPU Training for Brain Language System.
Key fixes:
- Larger batch size (256)
- More batches per epoch (ALL data)
- Proper GPU utilization
- Verified reconstruction loss
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
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# ==================== DATA LOADING ====================
def load_data(en_path='wikitext_en.txt', de_path='wiki_de.txt'):
    """Load training data."""
    en_sentences = []
    de_sentences = []
    
    if os.path.exists(en_path):
        with open(en_path, 'r', encoding='utf-8') as f:
            en_sentences = [line.strip() for line in f if 20 <= len(line.strip()) <= 100]
        print(f"✓ English: {len(en_sentences)} sentences")
    
    if os.path.exists(de_path):
        with open(de_path, 'r', encoding='utf-8') as f:
            de_sentences = [line.strip() for line in f if 20 <= len(line.strip()) <= 100]
        print(f"✓ German: {len(de_sentences)} sentences")
    
    return en_sentences, de_sentences


def text_to_tensor(text, max_len, device):
    """Convert text to byte tensor safely."""
    safe_bytes = text.encode('utf-8', errors='replace')[:max_len]
    byte_vals = [min(b, 255) for b in safe_bytes]
    if len(byte_vals) < max_len:
        byte_vals = byte_vals + [0] * (max_len - len(byte_vals))
    return torch.tensor(byte_vals, dtype=torch.long, device=device)


# ==================== BATCH TRAINING (FIXED) ====================
def train_epoch(brain, sentences, optimizer, scaler, device, batch_size=256, max_len=64):
    """Train one full epoch over all data."""
    brain.train()
    
    random.shuffle(sentences)
    total_loss = 0
    total_recon = 0
    total_next = 0
    num_batches = 0
    
    # Process ALL data, not just 100 batches
    for i in tqdm(range(0, len(sentences), batch_size), desc="Training", leave=False):
        batch_sentences = sentences[i:i+batch_size]
        if len(batch_sentences) < 4:
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
        
        if len(batch_tensors) < 4:
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
            
            # Reconstruction loss - CRITICAL!
            recon_logits = brain.reconstruct_input(semantic, seq_len=input_bytes.shape[1])
            recon_loss = F.cross_entropy(
                recon_logits.reshape(-1, 256),
                input_bytes.reshape(-1),
                ignore_index=0
            )
            
            # Combined loss - BOTH must be used!
            loss = next_loss + 2.0 * recon_loss
        
        # Backward
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_next += next_loss.item()
        num_batches += 1
    
    if num_batches == 0:
        return 0, 0, 0
    
    return total_loss / num_batches, total_recon / num_batches, total_next / num_batches


def train_n400(brain, optimizer, device):
    """Train N400 with ranking loss."""
    n400_pairs = [
        ("the cat sat on the", "mat", "fish"),
        ("the dog ran in the", "park", "ocean"),
        ("the bird flew in the", "sky", "table"),
        ("the fish swam in the", "water", "forest"),
        ("the sun shone", "brightly", "quietly"),
        ("red apples grow on", "trees", "rocks"),
        ("the child plays with", "toys", "clouds"),
        ("hot coffee warms the", "body", "ice"),
    ]
    
    total_loss = 0
    correct = 0
    
    for context, expected, unexpected in n400_pairs:
        optimizer.zero_grad()
        
        ctx = text_to_tensor(context, 64, device).unsqueeze(0)
        exp = text_to_tensor(expected, 64, device).unsqueeze(0)
        unexp = text_to_tensor(unexpected, 64, device).unsqueeze(0)
        
        with torch.no_grad():
            ctx_sem, _ = brain.comprehend(ctx, store_in_memory=False)
            exp_sem, _ = brain.comprehend(exp, store_in_memory=False)
            unexp_sem, _ = brain.comprehend(unexp, store_in_memory=False)
        
        ctx_expanded = ctx_sem.unsqueeze(1)
        lstm_out, _ = brain.wernicke.predictor(ctx_expanded)
        predicted = brain.wernicke.prediction_head(lstm_out.squeeze(1))
        
        exp_surprise = F.mse_loss(predicted, exp_sem)
        unexp_surprise = F.mse_loss(predicted, unexp_sem)
        
        loss = F.relu(exp_surprise - unexp_surprise + 1.0)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        if exp_surprise.item() < unexp_surprise.item():
            correct += 1
    
    return total_loss / len(n400_pairs), correct / len(n400_pairs)


def compute_diversity(brain, sentences, device, n=50):
    """Compute embedding diversity."""
    brain.eval()
    embeddings = []
    
    samples = random.sample(sentences, min(n, len(sentences)))
    with torch.no_grad():
        for sent in samples:
            try:
                tensor = text_to_tensor(sent, 64, device).unsqueeze(0)
                sem, _ = brain.comprehend(tensor, store_in_memory=False)
                embeddings.append(sem[0].cpu())
            except:
                continue
    
    if len(embeddings) < 2:
        return 0.0
    
    embeddings = torch.stack(embeddings)
    embeddings = F.normalize(embeddings, dim=-1)
    sim = torch.matmul(embeddings, embeddings.T)
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    return sim[mask].mean().item()


def test_generation(brain, prompts, device):
    """Test generation quality."""
    brain.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                tensor = text_to_tensor(prompt, 64, device).unsqueeze(0)
                sem, _ = brain.comprehend(tensor, store_in_memory=False)
                output = brain.produce(sem, max_length=40)
                out_bytes = output[0].cpu().numpy()
                out_text = bytes(b for b in out_bytes if 32 <= b < 127).decode('utf-8', errors='replace')
                results.append((prompt, out_text[:50]))
            except Exception as e:
                results.append((prompt, f"[error]"))
    
    return results


# ==================== MAIN ====================
def main():
    print("\n" + "=" * 70)
    print("🧠 FIXED 24-HOUR GPU TRAINING")
    print("=" * 70)
    
    device = check_gpu()
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    en_sentences, de_sentences = load_data()
    if not en_sentences:
        print("❌ No training data!")
        return
    
    # Model
    print("\nInitializing model...")
    config = create_brain_language_config()
    config['num_concepts'] = 2000
    brain = BrainLanguageSystem(config).to(device)
    
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"✓ Parameters: {n_params:,}")
    
    # Optimizers
    main_opt = optim.AdamW(brain.parameters(), lr=3e-4, weight_decay=1e-5)
    n400_params = list(brain.wernicke.predictor.parameters()) + \
                  list(brain.wernicke.prediction_head.parameters())
    n400_opt = optim.Adam(n400_params, lr=1e-3)
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(main_opt, T_max=500, eta_min=1e-6)
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # Training config
    BATCH_SIZE = 256  # BIGGER!
    
    print(f"\n{'=' * 70}")
    print(f"🚀 TRAINING CONFIG")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Sentences/epoch: {len(en_sentences)}")
    print(f"   Batches/epoch: {len(en_sentences) // BATCH_SIZE}")
    print(f"{'=' * 70}\n")
    
    start_time = time.time()
    end_time = start_time + 24 * 3600
    
    best_loss = float('inf')
    metrics = []
    
    epoch = 0
    while time.time() < end_time:
        epoch += 1
        epoch_start = time.time()
        
        # Train epoch
        loss, recon, next_l = train_epoch(
            brain, en_sentences, main_opt, scaler, device, 
            batch_size=BATCH_SIZE
        )
        scheduler.step()
        
        # N400 training every 5 epochs
        n400_loss, n400_acc = 0, 0
        if epoch % 5 == 0:
            for _ in range(5):
                l, a = train_n400(brain, n400_opt, device)
                n400_loss += l
                n400_acc += a
            n400_loss /= 5
            n400_acc /= 5
        
        # Diversity check every 10 epochs
        diversity = 0
        if epoch % 10 == 0:
            diversity = compute_diversity(brain, en_sentences, device)
        
        elapsed = time.time() - start_time
        remaining = max(0, end_time - time.time())
        epoch_time = time.time() - epoch_start
        
        m = {
            'epoch': epoch,
            'loss': loss,
            'recon_loss': recon,
            'next_loss': next_l,
            'n400_acc': n400_acc,
            'diversity': diversity,
            'lr': scheduler.get_last_lr()[0],
            'elapsed_h': elapsed / 3600,
            'remaining_h': remaining / 3600,
        }
        metrics.append(m)
        
        # Logging every 5 epochs
        if epoch % 5 == 0:
            print(f"\n{'=' * 60}")
            print(f"📊 EPOCH {epoch}")
            print(f"{'=' * 60}")
            print(f"Loss:      {loss:.4f} (recon: {recon:.4f}, next: {next_l:.4f})")
            print(f"N400:      acc={n400_acc:.1%}")
            print(f"Diversity: {diversity:.4f}")
            print(f"LR:        {scheduler.get_last_lr()[0]:.6f}")
            print(f"Time:      {elapsed/3600:.2f}h elapsed, {remaining/3600:.1f}h left")
            print(f"Speed:     {epoch_time:.1f}s/epoch")
            if device.type == 'cuda':
                print(f"GPU Mem:   {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
            # Sanity check
            if loss > 6.0 and epoch > 50:
                print(f"\n⚠️  WARNING: Loss {loss:.2f} > 6.0 after {epoch} epochs!")
                print(f"⚠️  Recon loss: {recon:.4f}, Next loss: {next_l:.4f}")
        
        # Checkpoint
        if epoch % 25 == 0 or loss < best_loss * 0.95:
            if loss < best_loss:
                best_loss = loss
                print(f"✅ NEW BEST: {best_loss:.4f}")
            
            torch.save({
                'epoch': epoch,
                'model': brain.state_dict(),
                'optimizer': main_opt.state_dict(),
                'loss': loss,
                'best_loss': best_loss,
            }, f'checkpoints/brain_fixed_e{epoch}.pth')
            torch.save(brain.state_dict(), 'checkpoints/brain_fixed_latest.pth')
            
            with open('training_fixed.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        
        # Generation test every 50 epochs
        if epoch % 50 == 0:
            print("\n🎯 GENERATION:")
            for prompt, output in test_generation(brain, ["the cat", "hello"], device):
                print(f"  '{prompt}' → '{output}'")
    
    print(f"\n{'=' * 70}")
    print(f"✅ TRAINING COMPLETE")
    print(f"   Epochs: {epoch}")
    print(f"   Final loss: {loss:.4f}")
    print(f"   Best loss: {best_loss:.4f}")
    print(f"{'=' * 70}")
    
    torch.save(brain.state_dict(), 'checkpoints/brain_fixed_final.pth')


if __name__ == "__main__":
    main()
