#!/usr/bin/env python3
"""
24-Hour GPU Training for Brain Language System.
Uses WikiText-103 (English) and German Wikipedia for cross-lingual learning.
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
from datetime import datetime
from pathlib import Path

from digital_brain.modules.brain_language import (
    BrainLanguageSystem,
    create_brain_language_config
)

# ==================== GPU CHECK ====================
def check_gpu():
    if not torch.cuda.is_available():
        print("⚠️  No GPU available, using CPU (will be slower)")
        return torch.device('cpu')
    
    device = torch.device('cuda')
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    return device


# ==================== DATA LOADING ====================
def load_english_data(path: str = 'wikitext_en.txt', max_sentences: int = 500000):
    """Load English sentences from WikiText."""
    print(f"Loading English data from {path}...")
    
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, using template corpus")
        return generate_template_corpus(max_sentences // 10)
    
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 20 <= len(line) <= 120:  # Good length range
                sentences.append(line)
                if len(sentences) >= max_sentences:
                    break
    
    print(f"✓ Loaded {len(sentences)} English sentences")
    return sentences


def load_german_data(path: str = 'wiki_de.txt', max_sentences: int = 100000):
    """Load German sentences."""
    print(f"Loading German data from {path}...")
    
    if not os.path.exists(path):
        print(f"⚠️  {path} not found, generating German templates")
        return generate_german_templates(max_sentences // 10)
    
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if 20 <= len(line) <= 120:
                sentences.append(line)
                if len(sentences) >= max_sentences:
                    break
    
    print(f"✓ Loaded {len(sentences)} German sentences")
    return sentences


def generate_template_corpus(n: int = 10000):
    """Generate template-based corpus as fallback."""
    templates = [
        "the {noun} {verb} {adverb}",
        "a {adj} {noun} {verb} in the {place}",
        "{noun} always {verb} {adverb}",
        "the {adj} {noun} is {state}",
        "{noun} and {noun2} {verb} together",
    ]
    nouns = ["cat", "dog", "bird", "tree", "house", "car", "child", "person", "flower", "sun"]
    verbs = ["runs", "jumps", "sleeps", "works", "plays", "learns", "thinks", "walks", "flies"]
    adverbs = ["quickly", "slowly", "quietly", "loudly", "happily", "sadly", "carefully"]
    adjs = ["big", "small", "old", "new", "beautiful", "bright", "dark", "fast", "slow"]
    places = ["garden", "house", "forest", "park", "room", "office", "field", "street"]
    states = ["tired", "happy", "sad", "hungry", "calm", "excited", "ready", "waiting"]
    
    sentences = []
    for _ in range(n):
        template = random.choice(templates)
        sent = template.format(
            noun=random.choice(nouns),
            noun2=random.choice(nouns),
            verb=random.choice(verbs),
            adverb=random.choice(adverbs),
            adj=random.choice(adjs),
            place=random.choice(places),
            state=random.choice(states)
        )
        sentences.append(sent)
    
    return sentences


def generate_german_templates(n: int = 10000):
    """Generate German template corpus."""
    templates = [
        "die {noun} {verb} {adverb}",
        "ein {adj} {noun} {verb} im {place}",
        "{noun} {verb} immer {adverb}",
        "der {adj} {noun} ist {state}",
    ]
    nouns = ["katze", "hund", "vogel", "baum", "haus", "auto", "kind", "mann", "frau"]
    verbs = ["läuft", "springt", "schläft", "arbeitet", "spielt", "lernt", "denkt"]
    adverbs = ["schnell", "langsam", "leise", "laut", "gut", "glücklich"]
    adjs = ["groß", "klein", "alt", "neu", "schön", "hell", "dunkel"]
    places = ["garten", "haus", "wald", "park", "zimmer", "büro"]
    states = ["müde", "glücklich", "traurig", "hungrig", "ruhig"]
    
    sentences = []
    for _ in range(n):
        template = random.choice(templates)
        sent = template.format(
            noun=random.choice(nouns),
            verb=random.choice(verbs),
            adverb=random.choice(adverbs),
            adj=random.choice(adjs),
            place=random.choice(places),
            state=random.choice(states)
        )
        sentences.append(sent)
    
    return sentences


# ==================== TRAINING FUNCTIONS ====================
def train_batch(brain, batch, optimizer, scaler, device, max_len=64):
    """Train on one batch with mixed precision."""
    brain.train()
    
    total_loss = 0
    valid_samples = 0
    
    # Convert sentences to byte tensors (with safe encoding)
    batch_tensors = []
    for sentence in batch:
        try:
            # Encode to bytes safely, replacing invalid chars
            safe_bytes = sentence.encode('utf-8', errors='replace')[:max_len]
            byte_vals = [min(b, 255) for b in safe_bytes]  # Clamp to valid range
            
            # Pad to max_len
            if len(byte_vals) < max_len:
                byte_vals = byte_vals + [0] * (max_len - len(byte_vals))
            
            tensor = torch.tensor(byte_vals, dtype=torch.long, device=device)
            if tensor.sum() > 0:  # Not empty
                batch_tensors.append(tensor)
        except:
            continue
    
    if len(batch_tensors) < 2:
        return 0.0
    
    # Stack into batch
    batch_tensor = torch.stack(batch_tensors)  # [batch, seq_len]
    
    input_bytes = batch_tensor[:, :-1]
    target_bytes = batch_tensor[:, 1:]
    
    optimizer.zero_grad()
    
    # Forward with mixed precision
    with autocast('cuda' if device.type == 'cuda' else 'cpu'):
        # Comprehend
        semantic, concepts = brain.comprehend(input_bytes, store_in_memory=True)
        
        # Production loss (next byte prediction)
        logits = brain.produce(semantic, target_bytes=input_bytes)
        next_loss = F.cross_entropy(
            logits.reshape(-1, 256),
            target_bytes.reshape(-1),
            ignore_index=0,
            label_smoothing=0.1
        )
        
        # Reconstruction loss (forces embedding diversity)
        recon_logits = brain.reconstruct_input(semantic, seq_len=input_bytes.shape[1])
        recon_loss = F.cross_entropy(
            recon_logits.reshape(-1, 256),
            input_bytes.reshape(-1),
            ignore_index=0
        )
        
        # Combined loss
        loss = next_loss + 2.0 * recon_loss
    
    # Backward with gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()


def train_n400_ranking(brain, sentences, optimizer, device, margin=1.0):
    """Train N400 with ranking loss."""
    # N400 pairs
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
    total = 0
    
    random.shuffle(n400_pairs)
    
    for context, expected, unexpected in n400_pairs:
        optimizer.zero_grad()
        
        # Encode using safe function
        context_bytes = safe_text_to_tensor(context, 64, device).unsqueeze(0)
        expected_bytes = safe_text_to_tensor(expected, 64, device).unsqueeze(0)
        unexpected_bytes = safe_text_to_tensor(unexpected, 64, device).unsqueeze(0)
        
        with torch.no_grad():
            context_sem, _ = brain.comprehend(context_bytes, store_in_memory=False)
            expected_sem, _ = brain.comprehend(expected_bytes, store_in_memory=False)
            unexpected_sem, _ = brain.comprehend(unexpected_bytes, store_in_memory=False)
        
        # Predict from context
        context_expanded = context_sem.unsqueeze(1)
        lstm_out, _ = brain.wernicke.predictor(context_expanded)
        predicted_sem = brain.wernicke.prediction_head(lstm_out.squeeze(1))
        
        # Compute surprise
        expected_surprise = F.mse_loss(predicted_sem, expected_sem)
        unexpected_surprise = F.mse_loss(predicted_sem, unexpected_sem)
        
        # Ranking loss
        ranking_loss = F.relu(expected_surprise - unexpected_surprise + margin)
        ranking_loss.backward()
        optimizer.step()
        
        total_loss += ranking_loss.item()
        total += 1
        if expected_surprise.item() < unexpected_surprise.item():
            correct += 1
    
    return total_loss / max(total, 1), correct / max(total, 1)


def safe_text_to_tensor(text, max_len, device):
    """Safely convert text to byte tensor with proper clamping."""
    safe_bytes = text.encode('utf-8', errors='replace')[:max_len]
    byte_vals = [min(b, 255) for b in safe_bytes]
    if len(byte_vals) < max_len:
        byte_vals = byte_vals + [0] * (max_len - len(byte_vals))
    return torch.tensor(byte_vals, dtype=torch.long, device=device)


def compute_embedding_diversity(brain, sentences, device, n_samples=100):
    """Compute embedding diversity metric."""
    brain.eval()
    embeddings = []
    
    samples = random.sample(sentences, min(n_samples, len(sentences)))
    
    with torch.no_grad():
        for sent in samples:
            try:
                tensor = safe_text_to_tensor(sent, 64, device).unsqueeze(0)
                sem, _ = brain.comprehend(tensor, store_in_memory=False)
                embeddings.append(sem[0].cpu())  # Move to CPU to avoid GPU memory issues
            except:
                continue
    
    if len(embeddings) < 2:
        return 0.0
    
    embeddings = torch.stack(embeddings)
    embeddings = F.normalize(embeddings, dim=-1)
    
    # Average pairwise similarity
    sim_matrix = torch.matmul(embeddings, embeddings.T)
    mask = ~torch.eye(len(embeddings), dtype=torch.bool)
    avg_sim = sim_matrix[mask].mean().item()
    
    return avg_sim


def test_generation(brain, prompts, device):
    """Test text generation."""
    brain.eval()
    results = []
    
    with torch.no_grad():
        for prompt in prompts:
            try:
                prompt_bytes = safe_text_to_tensor(prompt, 64, device).unsqueeze(0)
                semantic, _ = brain.comprehend(prompt_bytes, store_in_memory=False)
                output = brain.produce(semantic, max_length=40)
                # Convert output bytes to text
                output_bytes = output[0].cpu().numpy()
                output_text = bytes(b for b in output_bytes if b > 0).decode('utf-8', errors='replace')
                results.append((prompt, output_text))
            except Exception as e:
                results.append((prompt, f"[error: {e}]"))
    
    return results


# ==================== MAIN TRAINING LOOP ====================
def main():
    print("\n" + "=" * 70)
    print("🧠 24-HOUR GPU TRAINING - BRAIN LANGUAGE SYSTEM")
    print("=" * 70 + "\n")
    
    # Setup
    device = check_gpu()
    os.makedirs('checkpoints', exist_ok=True)
    
    # Load data
    en_sentences = load_english_data()
    de_sentences = load_german_data()
    
    # Initialize model
    print("\nInitializing Brain Language System...")
    config = create_brain_language_config()
    config['num_concepts'] = 2000  # More concepts for larger vocab
    brain = BrainLanguageSystem(config).to(device)
    
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"✓ Model parameters: {n_params:,}")
    
    # Optimizers
    main_optimizer = optim.AdamW(brain.parameters(), lr=1e-4, weight_decay=1e-5)
    n400_params = list(brain.wernicke.predictor.parameters()) + \
                  list(brain.wernicke.prediction_head.parameters())
    n400_optimizer = optim.Adam(n400_params, lr=1e-3)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        main_optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )
    
    # Mixed precision scaler
    scaler = GradScaler('cuda' if device.type == 'cuda' else 'cpu')
    
    # Training config
    BATCH_SIZE = 64 if device.type == 'cuda' else 16
    MAX_EPOCHS = 10000  # Will stop after 24h anyway
    BATCHES_PER_EPOCH = 100
    
    # Training loop
    print("\n" + "=" * 70)
    print("🚀 STARTING 24-HOUR TRAINING")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Batches/epoch: {BATCHES_PER_EPOCH}")
    print(f"   English sentences: {len(en_sentences)}")
    print(f"   German sentences: {len(de_sentences)}")
    print("=" * 70 + "\n")
    
    start_time = time.time()
    end_time = start_time + 24 * 3600  # 24 hours
    
    best_loss = float('inf')
    metrics_history = []
    
    for epoch in range(1, MAX_EPOCHS + 1):
        if time.time() >= end_time:
            print("\n⏰ 24 hours reached, stopping...")
            break
        
        epoch_start = time.time()
        
        # ===== MAIN TRAINING =====
        random.shuffle(en_sentences)
        epoch_loss = 0
        
        for batch_idx in range(BATCHES_PER_EPOCH):
            start_idx = batch_idx * BATCH_SIZE
            batch = en_sentences[start_idx:start_idx + BATCH_SIZE]
            
            if len(batch) < 2:
                continue
            
            loss = train_batch(brain, batch, main_optimizer, scaler, device)
            epoch_loss += loss
        
        avg_loss = epoch_loss / BATCHES_PER_EPOCH
        scheduler.step()
        
        # ===== N400 TRAINING (every 5 epochs) =====
        n400_loss, n400_acc = 0, 0
        if epoch % 5 == 0:
            for _ in range(5):
                loss, acc = train_n400_ranking(brain, en_sentences, n400_optimizer, device)
                n400_loss += loss
                n400_acc += acc
            n400_loss /= 5
            n400_acc /= 5
        
        # ===== METRICS =====
        elapsed = time.time() - start_time
        remaining = max(0, end_time - time.time())
        epoch_time = time.time() - epoch_start
        
        # Compute diversity every 10 epochs
        diversity = 0.0
        if epoch % 10 == 0:
            diversity = compute_embedding_diversity(brain, en_sentences, device)
        
        metrics = {
            'epoch': epoch,
            'loss': avg_loss,
            'n400_loss': n400_loss,
            'n400_accuracy': n400_acc,
            'embedding_diversity': diversity,
            'lr': scheduler.get_last_lr()[0],
            'elapsed_hours': elapsed / 3600,
            'remaining_hours': remaining / 3600,
            'epoch_time': epoch_time
        }
        metrics_history.append(metrics)
        
        # ===== LOGGING =====
        if epoch % 10 == 0:
            print(f"\n{'=' * 60}")
            print(f"📊 EPOCH {epoch}")
            print(f"{'=' * 60}")
            print(f"Loss:       {avg_loss:.4f}")
            print(f"N400:       loss={n400_loss:.4f}, acc={n400_acc:.1%}")
            print(f"Diversity:  {diversity:.4f} (want < 0.5)")
            print(f"LR:         {metrics['lr']:.6f}")
            print(f"Time:       {elapsed/3600:.1f}h elapsed, {remaining/3600:.1f}h left")
            
            if device.type == 'cuda':
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # ===== CHECKPOINTING =====
        if epoch % 50 == 0 or avg_loss < best_loss * 0.95:
            if avg_loss < best_loss:
                best_loss = avg_loss
                print(f"\n✅ NEW BEST LOSS: {best_loss:.4f}")
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': brain.state_dict(),
                'optimizer_state_dict': main_optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'metrics': metrics
            }
            
            torch.save(checkpoint, f'checkpoints/brain_gpu_epoch_{epoch}.pth')
            torch.save(brain.state_dict(), 'checkpoints/brain_gpu_latest.pth')
            
            # Save metrics
            with open('training_metrics_gpu.json', 'w') as f:
                json.dump(metrics_history, f, indent=2)
            
            print(f"💾 Checkpoint saved (epoch {epoch})")
        
        # ===== GENERATION TEST (every 50 epochs) =====
        if epoch % 50 == 0:
            print(f"\n🎯 GENERATION TEST:")
            results = test_generation(brain, ["the cat", "hello world", "deep learning"], device)
            for prompt, output in results:
                print(f"  '{prompt}' → '{output}'")
    
    # ===== FINAL SAVE =====
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Total epochs:  {epoch}")
    print(f"Final loss:    {avg_loss:.4f}")
    print(f"Best loss:     {best_loss:.4f}")
    
    torch.save(brain.state_dict(), 'checkpoints/brain_gpu_final.pth')
    
    with open('training_metrics_gpu.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)
    
    print("\n💾 Final checkpoint: checkpoints/brain_gpu_final.pth")
    print("📊 Metrics saved:    training_metrics_gpu.json")


if __name__ == "__main__":
    main()
