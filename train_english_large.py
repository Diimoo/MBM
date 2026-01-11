#!/usr/bin/env python3
"""
Train Hierarchical English Model on Large HuggingFace Corpus
Uses: alasdairforsythe/text-english-code-fiction-nonfiction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
import json
import random
import re
from tqdm import tqdm
from datasets import load_dataset

from english_hierarchical_v1 import (
    HierarchicalEnglishModel, 
    text_to_indices, indices_to_text, VOCAB_SIZE, PAD_TOKEN,
    char_to_idx, idx_to_char, VOWELS
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Syllable rules for English (heuristic-based for unsupervised learning)
CONSONANTS = set("bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ")
VOWELS = set("aeiouAEIOU")


def is_likely_syllable_boundary(text, pos):
    """
    Heuristic: syllable boundary between consonant and vowel (CV pattern)
    or between two consonants where second starts a new syllable.
    """
    if pos <= 0 or pos >= len(text):
        return False
    
    prev_char = text[pos - 1]
    curr_char = text[pos]
    
    # Space is always a boundary
    if curr_char == ' ' or prev_char == ' ':
        return curr_char != ' '
    
    # Vowel-Consonant-Vowel: split before consonant
    if pos >= 2 and pos < len(text) - 1:
        if (text[pos-1] in CONSONANTS and 
            text[pos] in VOWELS and 
            text[pos-2] in VOWELS):
            return True
    
    # Consonant cluster: split between them
    if prev_char in CONSONANTS and curr_char in CONSONANTS:
        # Common clusters that shouldn't split: bl, br, cl, cr, dr, fl, fr, gl, gr, pl, pr, sc, sk, sl, sm, sn, sp, st, sw, tr, tw
        cluster = prev_char.lower() + curr_char.lower()
        no_split = {'bl', 'br', 'cl', 'cr', 'dr', 'fl', 'fr', 'gl', 'gr', 'pl', 'pr', 
                    'sc', 'sk', 'sl', 'sm', 'sn', 'sp', 'st', 'sw', 'tr', 'tw', 'th', 'ch', 'sh', 'wh'}
        if cluster not in no_split:
            return True
    
    return False


def generate_syllable_boundaries(text):
    """Generate syllable boundary labels for text."""
    boundaries = [1]  # First char is always a boundary
    for i in range(1, len(text)):
        if text[i] == ' ':
            boundaries.append(0)
        elif text[i-1] == ' ':
            boundaries.append(1)  # Start of word
        elif is_likely_syllable_boundary(text, i):
            boundaries.append(1)
        else:
            boundaries.append(0)
    return boundaries


class LargeEnglishDataset(Dataset):
    """Dataset from HuggingFace corpus."""
    
    def __init__(self, texts, max_len=128):
        self.texts = texts
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Clean text
        text = text.strip()
        if len(text) > self.max_len:
            # Take a random window
            start = random.randint(0, len(text) - self.max_len)
            text = text[start:start + self.max_len]
        
        indices = text_to_indices(text, self.max_len)
        boundaries = generate_syllable_boundaries(text)
        
        # Pad boundaries
        while len(boundaries) < self.max_len:
            boundaries.append(0)
        boundaries = boundaries[:self.max_len]
        
        return {
            "indices": torch.tensor(indices, dtype=torch.long),
            "boundaries": torch.tensor(boundaries, dtype=torch.float),
            "text": text,
        }


def load_and_process_dataset(max_samples=500000):
    """Load HuggingFace dataset and extract clean English sentences."""
    print("\n📥 Loading HuggingFace dataset...")
    print("   Dataset: alasdairforsythe/text-english-code-fiction-nonfiction")
    
    try:
        ds = load_dataset("alasdairforsythe/text-english-code-fiction-nonfiction", split="train", streaming=True)
    except Exception as e:
        print(f"   ⚠️ Streaming failed, trying regular load: {e}")
        ds = load_dataset("alasdairforsythe/text-english-code-fiction-nonfiction", split="train")
    
    print("   Processing texts...")
    
    texts = []
    seen = set()
    
    for i, item in enumerate(tqdm(ds, desc="Loading", total=max_samples)):
        if len(texts) >= max_samples:
            break
        
        text = item.get("text", "")
        if not text:
            continue
        
        # Split into sentences/chunks
        sentences = re.split(r'[.!?]\s+', text)
        
        for sent in sentences:
            sent = sent.strip()
            
            # Filter criteria
            if len(sent) < 20 or len(sent) > 200:
                continue
            
            # Skip code-like content
            if any(c in sent for c in ['```', '{', '}', '\\', '//', '/*', '#include']):
                continue
            
            # Skip if too many numbers or special chars
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in sent) / len(sent)
            if alpha_ratio < 0.8:
                continue
            
            # Deduplicate
            key = sent[:50].lower()
            if key in seen:
                continue
            seen.add(key)
            
            texts.append(sent)
            
            if len(texts) >= max_samples:
                break
    
    print(f"   ✅ Loaded {len(texts):,} clean sentences")
    return texts


def test_model(model, test_texts=None):
    """Test syllabification on sample texts."""
    model.eval()
    
    if test_texts is None:
        test_texts = [
            "beautiful", "wonderful", "international", "understanding",
            "computer", "information", "butterfly", "entertainment",
            "responsibility", "communication", "extraordinary", "nevertheless"
        ]
    
    print("\n📝 Syllabification Test:")
    for text in test_texts[:10]:
        syllables = model.get_syllables(text)
        print(f"   {text} → {'-'.join(syllables)}")


def train_on_large_corpus(model, texts, num_epochs=10, batch_size=64, lr=3e-4):
    """Train model on large corpus."""
    print("\n" + "=" * 60)
    print("Training on Large English Corpus")
    print("=" * 60)
    print(f"Samples: {len(texts):,}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Split train/val
    random.shuffle(texts)
    split_idx = int(len(texts) * 0.95)
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    train_dataset = LargeEnglishDataset(train_texts, max_len=128)
    val_dataset = LargeEnglishDataset(val_texts, max_len=128)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=2, T_mult=2)
    
    char_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    boundary_criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    best_val_loss = float("inf")
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_char_loss = 0
        train_bound_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            indices = batch["indices"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)
            B, L = indices.shape
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(indices, return_all_levels=True)
            
            # Character reconstruction loss (masked)
            mask_rate = 0.15
            mask = (torch.rand(B, L, device=DEVICE) < mask_rate) & (indices != PAD_TOKEN)
            char_loss = char_criterion(outputs["char_recon"][mask], indices[mask])
            
            # Syllable boundary loss
            pad_mask = (indices != PAD_TOKEN).float()
            bound_loss = (boundary_criterion(outputs["syllable_boundaries"], boundaries) * pad_mask).sum() / pad_mask.sum()
            
            # Combined loss
            loss = char_loss + 0.5 * bound_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_char_loss += char_loss.item()
            train_bound_loss += bound_loss.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "char": f"{char_loss.item():.4f}",
                "bound": f"{bound_loss.item():.4f}"
            })
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_char_acc = 0
        val_bound_acc = 0
        total_chars = 0
        total_bounds = 0
        
        with torch.no_grad():
            for batch in val_loader:
                indices = batch["indices"].to(DEVICE)
                boundaries = batch["boundaries"].to(DEVICE)
                
                outputs = model(indices, return_all_levels=True)
                
                # Losses
                pad_mask = (indices != PAD_TOKEN).float()
                char_loss = char_criterion(outputs["char_recon"].reshape(-1, VOCAB_SIZE), indices.reshape(-1))
                bound_loss = (boundary_criterion(outputs["syllable_boundaries"], boundaries) * pad_mask).sum() / pad_mask.sum()
                
                val_loss += (char_loss + 0.5 * bound_loss).item()
                
                # Accuracies
                char_preds = outputs["char_recon"].argmax(dim=-1)
                val_char_acc += ((char_preds == indices) * pad_mask).sum().item()
                total_chars += pad_mask.sum().item()
                
                bound_preds = (torch.sigmoid(outputs["syllable_boundaries"]) > 0.5).float()
                val_bound_acc += ((bound_preds == boundaries) * pad_mask).sum().item()
                total_bounds += pad_mask.sum().item()
        
        val_loss /= len(val_loader)
        char_acc = val_char_acc / total_chars
        bound_acc = val_bound_acc / total_bounds
        
        print(f"\nEpoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, CharAcc={char_acc:.2%}, BoundAcc={bound_acc:.2%}")
        
        # Test
        test_model(model)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "english_large.pth")
            print(f"💾 Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    return best_val_loss


def main():
    print("=" * 60)
    print("🇬🇧 Hierarchical English - Large Corpus Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load model
    model = HierarchicalEnglishModel(
        d_char=128,
        d_syllable=256,
        d_morpheme=256,
        d_word=512,
        max_len=128
    ).to(DEVICE)
    
    # Load previous checkpoint
    checkpoint_path = CHECKPOINT_DIR / "english_hierarchical.pth"
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
        print(f"✅ Loaded checkpoint: {checkpoint_path}")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test before training
    print("\n📝 Before training:")
    test_model(model)
    
    # Load large corpus
    texts = load_and_process_dataset(max_samples=200000)
    
    # Train
    train_on_large_corpus(model, texts, num_epochs=15, batch_size=64, lr=2e-4)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    # Final test
    print("\n📝 Final Test:")
    test_model(model, [
        "beautiful", "responsibility", "communication", "extraordinary",
        "nevertheless", "international", "understanding", "entertainment",
        "The quick brown fox jumps over the lazy dog",
        "She sells seashells by the seashore",
        "How much wood would a woodchuck chuck"
    ])


if __name__ == "__main__":
    main()
