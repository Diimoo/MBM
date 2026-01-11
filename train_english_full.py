#!/usr/bin/env python3
"""
Train the full Hierarchical English Model with all levels + decoder.

Training objectives:
1. Character reconstruction (Level 0)
2. Syllable boundary detection (Level 1)
3. Text generation via decoder (autoregressive)

Uses HuggingFace dataset for large-scale training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from tqdm import tqdm
import random

from english_hierarchical_v1 import (
    HierarchicalEnglishModel, text_to_indices, indices_to_text,
    VOCAB_SIZE, PAD_TOKEN, char_to_idx
)

# =============================================================================
# CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Training hyperparameters
BATCH_SIZE = 32  # Smaller due to larger model
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
MAX_LEN = 128  # Shorter for faster training
NUM_SENTENCES = 100000  # Number of sentences to use

# Loss weights
CHAR_LOSS_WEIGHT = 1.0
BOUNDARY_LOSS_WEIGHT = 1.0
DECODER_LOSS_WEIGHT = 2.0  # Higher weight for generation


# =============================================================================
# DATASET
# =============================================================================

class EnglishTextDataset(Dataset):
    """Dataset for training the full hierarchical model."""
    
    def __init__(self, sentences, max_len=MAX_LEN):
        self.sentences = sentences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        
        # Convert to indices
        indices = text_to_indices(text, self.max_len)
        indices = torch.tensor(indices, dtype=torch.long)
        
        # Generate heuristic syllable boundaries
        boundaries = self._get_syllable_boundaries(text)
        boundaries = torch.tensor(boundaries, dtype=torch.float)
        
        # Target for decoder is the same as input (autoencoder-style)
        target = indices.clone()
        
        return {
            "indices": indices,
            "boundaries": boundaries,
            "target": target,
        }
    
    def _get_syllable_boundaries(self, text):
        """Heuristic syllable boundary detection."""
        vowels = set("aeiouAEIOU")
        boundaries = [0.0] * self.max_len
        boundaries[0] = 1.0  # First char is always a boundary
        
        text = text[:self.max_len]
        in_word = False
        prev_was_vowel = False
        
        for i, c in enumerate(text):
            if c.isalpha():
                if not in_word:
                    boundaries[i] = 1.0  # Start of word
                    in_word = True
                    prev_was_vowel = c in vowels
                else:
                    is_vowel = c in vowels
                    # Boundary before consonant following vowel (simplified)
                    if prev_was_vowel and not is_vowel and i + 1 < len(text):
                        next_c = text[i + 1] if i + 1 < len(text) else ''
                        if next_c in vowels:
                            boundaries[i] = 1.0
                    prev_was_vowel = is_vowel
            else:
                in_word = False
                if c == ' ' and i > 0:
                    boundaries[i] = 1.0
        
        return boundaries


def load_huggingface_dataset(num_sentences=NUM_SENTENCES):
    """Load and process HuggingFace dataset."""
    print("\n📥 Loading HuggingFace dataset...")
    print("   Dataset: alasdairforsythe/text-english-code-fiction-nonfiction")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset(
            "alasdairforsythe/text-english-code-fiction-nonfiction",
            split="train",
            streaming=True
        )
        
        sentences = []
        print("   Processing texts...")
        
        for item in tqdm(ds, desc="Loading", total=num_sentences * 2):
            text = item.get("text", "")
            if not text:
                continue
            
            # Split into sentences and filter
            for sent in text.replace('\n', ' ').split('.'):
                sent = sent.strip()
                if 20 <= len(sent) <= MAX_LEN - 10:
                    # Basic cleaning
                    sent = ' '.join(sent.split())  # Normalize whitespace
                    if sent and sent[0].isupper():  # Start with capital
                        sentences.append(sent + '.')
                        if len(sentences) >= num_sentences:
                            break
            
            if len(sentences) >= num_sentences:
                break
        
        print(f"   ✅ Loaded {len(sentences):,} clean sentences")
        return sentences
        
    except Exception as e:
        print(f"   ⚠️ Error loading dataset: {e}")
        print("   Using synthetic data instead...")
        return generate_synthetic_data(num_sentences)


def generate_synthetic_data(num_sentences):
    """Generate synthetic English sentences for training."""
    templates = [
        "The {adj} {noun} {verb} the {adj2} {noun2}.",
        "A {adj} {noun} is {verb_ing} in the {place}.",
        "The {noun} {verb} {adv} across the {place}.",
        "{name} {verb} the {adj} {noun} {adv}.",
        "Many {noun_pl} are {verb_ing} in the {adj} {place}.",
    ]
    
    adjectives = ["beautiful", "quick", "lazy", "bright", "dark", "happy", "sad", 
                  "large", "small", "old", "young", "warm", "cold", "soft", "hard"]
    nouns = ["cat", "dog", "bird", "tree", "house", "river", "mountain", "book",
             "child", "person", "flower", "sun", "moon", "star", "cloud"]
    verbs = ["sees", "finds", "loves", "watches", "follows", "catches", "holds"]
    verb_ings = ["running", "walking", "sleeping", "playing", "swimming", "flying"]
    places = ["garden", "forest", "city", "ocean", "desert", "meadow", "valley"]
    adverbs = ["quickly", "slowly", "carefully", "happily", "quietly", "loudly"]
    names = ["Alice", "Bob", "Charlie", "Diana", "Edward", "Fiona", "George"]
    
    sentences = []
    for _ in range(num_sentences):
        template = random.choice(templates)
        sentence = template.format(
            adj=random.choice(adjectives),
            adj2=random.choice(adjectives),
            noun=random.choice(nouns),
            noun2=random.choice(nouns),
            noun_pl=random.choice(nouns) + "s",
            verb=random.choice(verbs),
            verb_ing=random.choice(verb_ings),
            place=random.choice(places),
            adv=random.choice(adverbs),
            name=random.choice(names),
        )
        sentences.append(sentence)
    
    return sentences


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, epoch):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_char_loss = 0
    total_boundary_loss = 0
    total_decoder_loss = 0
    
    char_correct = 0
    char_total = 0
    boundary_correct = 0
    boundary_total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in pbar:
        indices = batch["indices"].to(DEVICE)
        boundaries = batch["boundaries"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        
        optimizer.zero_grad()
        
        # Forward pass with decoder
        outputs = model(indices, return_all_levels=True, target_indices=target)
        
        char_recon = outputs["char_recon"]
        syllable_boundaries = outputs["syllable_boundaries"]
        decoder_logits = outputs["decoder_logits"]
        
        # Mask for non-padding positions
        mask = (indices != PAD_TOKEN).float()
        
        # Loss 1: Character reconstruction
        char_loss = F.cross_entropy(
            char_recon.view(-1, VOCAB_SIZE),
            indices.view(-1),
            ignore_index=PAD_TOKEN,
            reduction='mean'
        )
        
        # Loss 2: Syllable boundary detection
        boundary_loss = F.binary_cross_entropy_with_logits(
            syllable_boundaries, boundaries, weight=mask, reduction='sum'
        ) / mask.sum().clamp(min=1)
        
        # Loss 3: Decoder (next token prediction)
        # Shift target by 1 for autoregressive prediction
        decoder_target = target[:, 1:]  # [B, L-1]
        decoder_pred = decoder_logits[:, :-1, :]  # [B, L-1, V]
        
        decoder_loss = F.cross_entropy(
            decoder_pred.reshape(-1, VOCAB_SIZE),
            decoder_target.reshape(-1),
            ignore_index=PAD_TOKEN,
            reduction='mean'
        )
        
        # Combined loss
        loss = (CHAR_LOSS_WEIGHT * char_loss + 
                BOUNDARY_LOSS_WEIGHT * boundary_loss + 
                DECODER_LOSS_WEIGHT * decoder_loss)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Metrics
        total_loss += loss.item()
        total_char_loss += char_loss.item()
        total_boundary_loss += boundary_loss.item()
        total_decoder_loss += decoder_loss.item()
        
        # Character accuracy
        char_pred = char_recon.argmax(dim=-1)
        char_mask = mask.bool()
        char_correct += ((char_pred == indices) & char_mask).sum().item()
        char_total += char_mask.sum().item()
        
        # Boundary accuracy
        boundary_pred = (torch.sigmoid(syllable_boundaries) > 0.5).float()
        boundary_correct += ((boundary_pred == boundaries) * mask).sum().item()
        boundary_total += mask.sum().item()
        
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "char": f"{char_loss.item():.4f}",
            "bound": f"{boundary_loss.item():.4f}",
            "dec": f"{decoder_loss.item():.4f}",
        })
    
    n_batches = len(dataloader)
    return {
        "loss": total_loss / n_batches,
        "char_loss": total_char_loss / n_batches,
        "boundary_loss": total_boundary_loss / n_batches,
        "decoder_loss": total_decoder_loss / n_batches,
        "char_acc": char_correct / max(char_total, 1) * 100,
        "boundary_acc": boundary_correct / max(boundary_total, 1) * 100,
    }


def validate(model, dataloader):
    """Validate the model."""
    model.eval()
    
    total_loss = 0
    total_decoder_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            indices = batch["indices"].to(DEVICE)
            boundaries = batch["boundaries"].to(DEVICE)
            target = batch["target"].to(DEVICE)
            
            outputs = model(indices, return_all_levels=True, target_indices=target)
            
            char_recon = outputs["char_recon"]
            syllable_boundaries = outputs["syllable_boundaries"]
            decoder_logits = outputs["decoder_logits"]
            
            mask = (indices != PAD_TOKEN).float()
            
            char_loss = F.cross_entropy(
                char_recon.view(-1, VOCAB_SIZE),
                indices.view(-1),
                ignore_index=PAD_TOKEN,
                reduction='mean'
            )
            
            boundary_loss = F.binary_cross_entropy_with_logits(
                syllable_boundaries, boundaries, weight=mask, reduction='sum'
            ) / mask.sum().clamp(min=1)
            
            decoder_target = target[:, 1:]
            decoder_pred = decoder_logits[:, :-1, :]
            decoder_loss = F.cross_entropy(
                decoder_pred.reshape(-1, VOCAB_SIZE),
                decoder_target.reshape(-1),
                ignore_index=PAD_TOKEN,
                reduction='mean'
            )
            
            loss = (CHAR_LOSS_WEIGHT * char_loss + 
                    BOUNDARY_LOSS_WEIGHT * boundary_loss + 
                    DECODER_LOSS_WEIGHT * decoder_loss)
            
            total_loss += loss.item()
            total_decoder_loss += decoder_loss.item()
    
    return {
        "val_loss": total_loss / len(dataloader),
        "val_decoder_loss": total_decoder_loss / len(dataloader),
    }


def test_generation(model, test_sentences):
    """Test text generation on sample sentences."""
    model.eval()
    print("\n📝 Generation Test:")
    
    with torch.no_grad():
        for sent in test_sentences[:5]:
            # Encode and generate
            generated = model.generate(sent, max_len=len(sent) + 20, temperature=0.7)
            print(f"   Input:  '{sent[:50]}...'")
            print(f"   Output: '{generated[:50]}...'")
            print()


def main():
    print("=" * 60)
    print("Training Full Hierarchical English Model")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    
    # Load or create model
    model = HierarchicalEnglishModel(max_len=MAX_LEN).to(DEVICE)
    
    # Try to load pretrained lower levels
    pretrained_path = CHECKPOINT_DIR / "english_large.pth"
    if pretrained_path.exists():
        print(f"\n📥 Loading pretrained weights from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location=DEVICE, weights_only=False)
        
        # Load matching keys only (lower levels)
        pretrained_dict = checkpoint.get("model_state_dict", checkpoint)
        model_dict = model.state_dict()
        
        # Filter to only load matching keys
        matched = {k: v for k, v in pretrained_dict.items() 
                   if k in model_dict and v.shape == model_dict[k].shape}
        
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        print(f"   Loaded {len(matched)}/{len(model_dict)} parameters")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load dataset
    sentences = load_huggingface_dataset(NUM_SENTENCES)
    
    # Split into train/val
    random.shuffle(sentences)
    val_size = int(len(sentences) * 0.1)
    train_sentences = sentences[val_size:]
    val_sentences = sentences[:val_size]
    
    train_dataset = EnglishTextDataset(train_sentences, MAX_LEN)
    val_dataset = EnglishTextDataset(val_sentences, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    print(f"\nDataset: {len(train_sentences):,} train, {len(val_sentences):,} val")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)
    
    # Training loop
    print("\n" + "=" * 60)
    print("Training")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(NUM_EPOCHS):
        train_metrics = train_epoch(model, train_loader, optimizer, scheduler, epoch)
        val_metrics = validate(model, val_loader)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}:")
        print(f"  Train: loss={train_metrics['loss']:.4f}, "
              f"char_acc={train_metrics['char_acc']:.1f}%, "
              f"bound_acc={train_metrics['boundary_acc']:.1f}%")
        print(f"  Val:   loss={val_metrics['val_loss']:.4f}, "
              f"decoder_loss={val_metrics['val_decoder_loss']:.4f}")
        
        # Test generation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_generation(model, val_sentences[:3])
        
        # Save best model
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            patience_counter = 0
            
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "train_metrics": train_metrics,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "english_full.pth")
            print(f"  💾 Saved checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping after {patience} epochs without improvement")
                break
    
    # Final generation test
    print("\n" + "=" * 60)
    print("Final Generation Test")
    print("=" * 60)
    
    test_sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "She sells seashells by the seashore.",
        "How much wood would a woodchuck chuck?",
        "The beautiful butterfly landed on the flower.",
        "Learning language is a hierarchical process.",
    ]
    
    model.eval()
    with torch.no_grad():
        for sent in test_sentences:
            print(f"\n📝 Input: '{sent}'")
            
            # Syllabification
            syllables = model.get_syllables(sent)
            print(f"   Syllables: {'-'.join(syllables)}")
            
            # Encode to sentence embedding
            embedding = model.encode(sent)
            print(f"   Embedding: {embedding.shape} (norm={embedding.norm():.2f})")
            
            # Generate
            generated = model.generate(sent, max_len=80, temperature=0.7)
            print(f"   Generated: '{generated}'")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
