#!/usr/bin/env python3
"""
Train Hierarchical English Model - Level by Level
Phase 1: Train CharEncoder (character reconstruction)
Phase 2: Freeze CharEncoder, train SyllableDetector (boundary detection)
Phase 3: Add MorphemeParser, WordComposer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
import random
from tqdm import tqdm

from english_hierarchical_v1 import (
    HierarchicalEnglishModel, CharacterEncoder, SyllableDetector,
    text_to_indices, indices_to_text, VOCAB_SIZE, PAD_TOKEN,
    char_to_idx, idx_to_char, VOWELS
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = Path("english_basics_data")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


# =============================================================================
# DATASETS
# =============================================================================

class CharacterDataset(Dataset):
    """Dataset for character-level training (reconstruction + prediction)."""
    
    def __init__(self, texts, max_len=128):
        self.texts = texts
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = text_to_indices(text, self.max_len)
        return {
            "indices": torch.tensor(indices, dtype=torch.long),
            "text": text,
        }


class SyllableDataset(Dataset):
    """Dataset for syllable boundary detection."""
    
    def __init__(self, data_path, max_len=64):
        self.data = []
        self.max_len = max_len
        
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        word = item["word"]
        boundaries = item["boundaries"]
        
        # Convert to indices
        indices = text_to_indices(word, self.max_len)
        
        # Pad boundaries
        while len(boundaries) < self.max_len:
            boundaries.append(0)
        boundaries = boundaries[:self.max_len]
        
        return {
            "indices": torch.tensor(indices, dtype=torch.long),
            "boundaries": torch.tensor(boundaries, dtype=torch.float),
            "word": word,
            "syllables": item["syllable_str"],
        }


class SentenceDataset(Dataset):
    """Dataset for sentence-level training."""
    
    def __init__(self, data_path, max_len=128):
        self.data = []
        self.max_len = max_len
        
        with open(data_path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item["sentence"]
        indices = text_to_indices(sentence, self.max_len)
        
        return {
            "indices": torch.tensor(indices, dtype=torch.long),
            "sentence": sentence,
        }


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_level0(model, num_epochs=20, batch_size=64, lr=1e-3):
    """
    Phase 1: Train Character Encoder
    Task: Masked character prediction (like BERT but for characters)
    """
    print("\n" + "=" * 60)
    print("PHASE 1: Training Character Encoder")
    print("=" * 60)
    
    # Load sentence data for character training
    dataset = SentenceDataset(DATA_DIR / "sentences.jsonl", max_len=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.char_encoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            indices = batch["indices"].to(DEVICE)
            B, L = indices.shape
            
            # Mask 15% of characters (like BERT)
            mask = torch.rand(B, L, device=DEVICE) < 0.15
            mask = mask & (indices != PAD_TOKEN)  # Don't mask padding
            
            masked_indices = indices.clone()
            masked_indices[mask] = char_to_idx.get("?", 1)  # Replace with ?
            
            # Forward
            optimizer.zero_grad()
            char_embeds = model.char_encoder(masked_indices)
            logits = model.char_decoder(char_embeds)  # [B, L, vocab]
            
            # Loss only on masked positions
            loss = criterion(logits[mask], indices[mask])
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = logits[mask].argmax(dim=-1)
            correct += (preds == indices[mask]).sum().item()
            total += mask.sum().item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/max(1,total):.2%}"})
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        accuracy = correct / max(1, total)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={accuracy:.2%}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "english_level0.pth")
            print(f"💾 Saved checkpoint (loss={avg_loss:.4f})")
    
    return best_loss


def train_level1(model, num_epochs=30, batch_size=64, lr=5e-4):
    """
    Phase 2: Train Syllable Detector (freeze CharEncoder)
    Task: Predict syllable boundaries
    """
    print("\n" + "=" * 60)
    print("PHASE 2: Training Syllable Detector")
    print("=" * 60)
    
    # Freeze character encoder
    for param in model.char_encoder.parameters():
        param.requires_grad = False
    print("   ❄️ CharEncoder frozen")
    
    # Load syllable data
    dataset = SyllableDataset(DATA_DIR / "syllables.jsonl", max_len=64)
    
    # Split train/val
    train_size = int(len(dataset) * 0.9)
    train_data, val_data = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size]
    )
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.syllable_detector.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            indices = batch["indices"].to(DEVICE)
            targets = batch["boundaries"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward through char encoder (frozen)
            with torch.no_grad():
                char_embeds = model.char_encoder(indices)
            
            # Syllable detection
            _, boundary_logits = model.syllable_detector(char_embeds, indices)
            
            # Mask padding
            mask = (indices != PAD_TOKEN).float()
            loss = (criterion(boundary_logits, targets) * mask).sum() / mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                indices = batch["indices"].to(DEVICE)
                targets = batch["boundaries"].to(DEVICE)
                
                char_embeds = model.char_encoder(indices)
                _, boundary_logits = model.syllable_detector(char_embeds, indices)
                
                mask = (indices != PAD_TOKEN).float()
                loss = (criterion(boundary_logits, targets) * mask).sum() / mask.sum()
                val_loss += loss.item()
                
                # Accuracy
                preds = (torch.sigmoid(boundary_logits) > 0.5).float()
                correct += ((preds == targets) * mask).sum().item()
                total += mask.sum().item()
        
        val_loss /= len(val_loader)
        accuracy = correct / max(1, total)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, Acc={accuracy:.2%}")
        
        # Test syllabification
        if (epoch + 1) % 5 == 0:
            test_syllabification(model)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), CHECKPOINT_DIR / "english_level1.pth")
            print(f"💾 Saved checkpoint (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Unfreeze
    for param in model.char_encoder.parameters():
        param.requires_grad = True
    
    return best_val_loss


def test_syllabification(model):
    """Test syllable detection on sample words."""
    model.eval()
    
    test_words = [
        "beautiful", "computer", "happy", "international",
        "understanding", "butterfly", "information", "wonderful"
    ]
    
    print("\n📝 Syllable Test:")
    for word in test_words:
        syllables = model.get_syllables(word)
        print(f"   {word} → {'-'.join(syllables)}")
    print()


def train_full_model(model, num_epochs=20, batch_size=32, lr=1e-4):
    """
    Phase 3: Joint fine-tuning of all levels
    """
    print("\n" + "=" * 60)
    print("PHASE 3: Joint Fine-tuning")
    print("=" * 60)
    
    dataset = SentenceDataset(DATA_DIR / "sentences.jsonl", max_len=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    
    char_criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    
    best_loss = float("inf")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            indices = batch["indices"].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Full forward pass
            outputs = model(indices, return_all_levels=True)
            
            # Character reconstruction loss
            char_loss = char_criterion(
                outputs["char_recon"].reshape(-1, VOCAB_SIZE),
                indices.reshape(-1)
            )
            
            # Total loss
            loss = char_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        scheduler.step()
        avg_loss = total_loss / len(loader)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}")
        
        # Test
        if (epoch + 1) % 5 == 0:
            test_syllabification(model)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), CHECKPOINT_DIR / "english_hierarchical.pth")
            print(f"💾 Saved checkpoint (loss={avg_loss:.4f})")
    
    return best_loss


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("🇬🇧 Hierarchical English Model Training")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Initialize model
    model = HierarchicalEnglishModel(
        d_char=128,
        d_syllable=256,
        d_morpheme=256,
        d_word=512,
        max_len=128
    ).to(DEVICE)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 1: Character Encoder
    train_level0(model, num_epochs=15, batch_size=64, lr=1e-3)
    
    # Phase 2: Syllable Detector
    train_level1(model, num_epochs=30, batch_size=64, lr=5e-4)
    
    # Phase 3: Joint fine-tuning
    train_full_model(model, num_epochs=15, batch_size=32, lr=1e-4)
    
    print("\n" + "=" * 60)
    print("✅ Training Complete!")
    print("=" * 60)
    
    # Final test
    print("\n📝 Final Syllabification Test:")
    test_words = [
        "beautiful", "wonderful", "international", "understanding",
        "computer", "information", "butterfly", "entertainment"
    ]
    
    model.eval()
    for word in test_words:
        syllables = model.get_syllables(word)
        print(f"   {word} → {'-'.join(syllables)}")


if __name__ == "__main__":
    main()
