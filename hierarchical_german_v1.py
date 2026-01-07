#!/usr/bin/env python3
"""
Hierarchical German Language Model - Version 1
Level 0: Character Encoder
Level 1: Syllable Detector

Following human learning: chars → syllables → (future: morphemes → words → ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from datasets import load_dataset
from tqdm import tqdm
import time
import os
import random
import re

# =============================================================================
# VOCABULARY
# =============================================================================

CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'äöüßÄÖÜ'
    '0123456789'
    ' .,!?;:\'"()-–—/\\@#$%&*+=<>[]{}|~`^_'
    '\n\t'
)

VOWELS = set('aeiouäöüAEIOUÄÖÜ')
CONSONANTS = set('bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZß')

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
SYL_START = '<SYL>'  # Syllable boundary marker

char_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1, SYL_START: 2}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)

idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

print(f"Vocabulary size: {VOCAB_SIZE}")

def text_to_indices(text, max_len=128):
    """Convert text to character indices."""
    indices = [char_to_idx.get(c, char_to_idx[UNK_TOKEN]) for c in text[:max_len]]
    while len(indices) < max_len:
        indices.append(char_to_idx[PAD_TOKEN])
    return indices

def indices_to_text(indices):
    """Convert indices back to text."""
    chars = []
    for idx in indices:
        if idx == char_to_idx[PAD_TOKEN]:
            break
        c = idx_to_char.get(idx, '?')
        if c not in [PAD_TOKEN, UNK_TOKEN, SYL_START]:
            chars.append(c)
    return ''.join(chars)

# =============================================================================
# GERMAN SYLLABLE RULES
# =============================================================================

def syllabify_german(word):
    """
    Rule-based German syllabification.
    Returns list of syllables and boundary positions.
    
    Rules:
    1. Every syllable has exactly one vowel nucleus
    2. Single consonant between vowels → belongs to next syllable
    3. Multiple consonants → split keeping onset minimal
    4. Never split digraphs (ch, sch, st, sp at syllable start)
    """
    word = word.lower()
    
    # Handle empty/short words
    if len(word) <= 1:
        return [word], []
    
    # Find vowel positions
    vowel_positions = [i for i, c in enumerate(word) if c in VOWELS]
    
    if len(vowel_positions) <= 1:
        return [word], []
    
    syllables = []
    boundaries = []
    last_boundary = 0
    
    for i in range(len(vowel_positions) - 1):
        v1_pos = vowel_positions[i]
        v2_pos = vowel_positions[i + 1]
        
        # Consonants between vowels
        consonants_between = word[v1_pos + 1:v2_pos]
        
        if len(consonants_between) == 0:
            # Adjacent vowels (diphthong or hiatus)
            # For simplicity, treat as same syllable unless specific patterns
            continue
        elif len(consonants_between) == 1:
            # Single consonant → belongs to next syllable
            boundary = v1_pos + 1
        else:
            # Multiple consonants → split
            # Keep common German onsets together (bl, br, dr, fl, fr, gl, gr, kl, kr, pl, pr, etc.)
            boundary = v1_pos + 1
            
            # Check for German-specific patterns
            remaining = consonants_between
            
            # Digraphs that shouldn't be split: ch, sch, sp, st (at start)
            if remaining.startswith('sch'):
                boundary = v2_pos - len(remaining) + 3
            elif remaining.startswith('ch'):
                boundary = v2_pos - len(remaining) + 2
            elif len(remaining) >= 2:
                # Standard: give one consonant to previous syllable
                boundary = v1_pos + 2
        
        if boundary > last_boundary and boundary < len(word):
            syllables.append(word[last_boundary:boundary])
            boundaries.append(boundary)
            last_boundary = boundary
    
    # Add final syllable
    if last_boundary < len(word):
        syllables.append(word[last_boundary:])
    
    return syllables, boundaries


def create_syllable_labels(text, max_len=128):
    """
    Create syllable boundary labels for a text.
    Returns: (char_indices, boundary_labels)
    
    boundary_labels[i] = 1 if there's a syllable boundary AFTER position i
    """
    # Tokenize into words
    words = re.findall(r'\w+|[^\w\s]|\s', text)
    
    char_indices = []
    boundary_labels = []
    
    char_pos = 0
    for word in words:
        if len(word) == 0:
            continue
            
        if word.isalpha():
            # Syllabify the word
            syllables, boundaries = syllabify_german(word)
            
            # Convert word characters to indices with boundary labels
            for i, c in enumerate(word):
                if char_pos >= max_len:
                    break
                char_indices.append(char_to_idx.get(c, char_to_idx[UNK_TOKEN]))
                # Mark boundary if this position is a syllable boundary
                is_boundary = (i + 1) in [b - (char_pos - len(char_indices) + 1) for b in boundaries]
                boundary_labels.append(1 if is_boundary else 0)
                char_pos += 1
        else:
            # Non-alphabetic (punctuation, space)
            for c in word:
                if char_pos >= max_len:
                    break
                char_indices.append(char_to_idx.get(c, char_to_idx[UNK_TOKEN]))
                boundary_labels.append(1)  # Always boundary after punctuation/space
                char_pos += 1
    
    # Pad
    while len(char_indices) < max_len:
        char_indices.append(char_to_idx[PAD_TOKEN])
        boundary_labels.append(0)
    
    return char_indices[:max_len], boundary_labels[:max_len]


# =============================================================================
# LEVEL 0: CHARACTER ENCODER
# =============================================================================

class CharacterEncoder(nn.Module):
    """
    Level 0: Learn character representations.
    
    Input: Character indices [batch, seq_len]
    Output: Character embeddings with local context [batch, seq_len, d_char]
    """
    def __init__(self, vocab_size, d_char=128, max_len=256):
        super().__init__()
        self.d_char = d_char
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=0)
        
        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(max_len, d_char)
        
        # Local context via small convolutions (like human visual processing of letters)
        self.local_context = nn.Sequential(
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1, groups=1),
        )
        
        self.norm = nn.LayerNorm(d_char)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        x: [batch, seq_len] character indices
        returns: [batch, seq_len, d_char] character embeddings
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # Character embeddings
        char_emb = self.char_embed(x)  # [batch, seq_len, d_char]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)  # [batch, seq_len, d_char]
        
        # Combine
        x = char_emb + pos_emb  # [batch, seq_len, d_char]
        
        # Local context (CNN)
        x_conv = x.transpose(1, 2)  # [batch, d_char, seq_len]
        x_conv = self.local_context(x_conv)  # [batch, d_char, seq_len]
        x_conv = x_conv.transpose(1, 2)  # [batch, seq_len, d_char]
        
        # Residual + norm
        x = self.norm(x + x_conv)
        x = self.dropout(x)
        
        return x


# =============================================================================
# LEVEL 1: SYLLABLE DETECTOR
# =============================================================================

class SyllableDetector(nn.Module):
    """
    Level 1: Detect syllable boundaries and create syllable representations.
    
    Input: Character embeddings [batch, seq_len, d_char]
    Output: 
        - Syllable boundary logits [batch, seq_len] (1 = boundary after this char)
        - Syllable embeddings [batch, max_syllables, d_syllable]
    """
    def __init__(self, d_char=128, d_syllable=256, max_syllables=64):
        super().__init__()
        self.d_syllable = d_syllable
        self.max_syllables = max_syllables
        
        # Boundary detection network
        # Uses bidirectional LSTM to see context on both sides
        self.boundary_lstm = nn.LSTM(
            d_char, d_char // 2, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        
        # Boundary classifier
        self.boundary_head = nn.Sequential(
            nn.Linear(d_char, d_char),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_char, 1)  # Binary: boundary or not
        )
        
        # Syllable representation builder
        # Takes characters within a syllable and creates a single representation
        self.syllable_encoder = nn.LSTM(
            d_char, d_syllable,
            num_layers=1,
            batch_first=True
        )
        
        # Projection for syllable embedding
        self.syllable_project = nn.Sequential(
            nn.Linear(d_syllable, d_syllable),
            nn.LayerNorm(d_syllable),
            nn.GELU()
        )
        
    def forward(self, char_embeddings, boundary_labels=None):
        """
        char_embeddings: [batch, seq_len, d_char]
        boundary_labels: [batch, seq_len] optional ground truth for training
        
        Returns:
            boundary_logits: [batch, seq_len]
            syllable_embeddings: [batch, max_syllables, d_syllable]
            syllable_mask: [batch, max_syllables] (1 = valid syllable)
        """
        batch_size, seq_len, d_char = char_embeddings.shape
        device = char_embeddings.device
        
        # Detect boundaries
        lstm_out, _ = self.boundary_lstm(char_embeddings)  # [batch, seq_len, d_char]
        boundary_logits = self.boundary_head(lstm_out).squeeze(-1)  # [batch, seq_len]
        
        # During training, use ground truth boundaries
        # During inference, use predicted boundaries
        if boundary_labels is not None:
            boundaries = boundary_labels
        else:
            boundaries = (torch.sigmoid(boundary_logits) > 0.5).float()
        
        # Build syllable representations
        syllable_embeddings = torch.zeros(
            batch_size, self.max_syllables, self.d_syllable, device=device
        )
        syllable_mask = torch.zeros(batch_size, self.max_syllables, device=device)
        
        for b in range(batch_size):
            # Find syllable boundaries for this sample
            boundary_positions = [0]  # Start of first syllable
            for i in range(seq_len):
                if boundaries[b, i] > 0.5:
                    boundary_positions.append(i + 1)
            
            # Ensure we don't exceed max_syllables
            num_syllables = min(len(boundary_positions), self.max_syllables)
            
            for s in range(num_syllables - 1):
                start = boundary_positions[s]
                end = boundary_positions[s + 1] if s + 1 < len(boundary_positions) else seq_len
                
                if start >= end or start >= seq_len:
                    continue
                
                # Get characters in this syllable
                syllable_chars = char_embeddings[b:b+1, start:end, :]  # [1, syl_len, d_char]
                
                # Encode syllable
                _, (h, _) = self.syllable_encoder(syllable_chars)
                syllable_emb = h[-1]  # [1, d_syllable]
                
                syllable_embeddings[b, s] = self.syllable_project(syllable_emb)
                syllable_mask[b, s] = 1.0
        
        return boundary_logits, syllable_embeddings, syllable_mask


# =============================================================================
# SYLLABLE DECODER (for reconstruction/generation)
# =============================================================================

class SyllableDecoder(nn.Module):
    """
    Decode syllable embeddings back to characters.
    This forces the syllable representation to be meaningful.
    """
    def __init__(self, vocab_size, d_syllable=256, d_hidden=256, max_chars_per_syllable=10):
        super().__init__()
        self.max_chars = max_chars_per_syllable
        self.vocab_size = vocab_size
        
        # Decode syllable to character sequence
        self.decoder = nn.LSTM(
            d_syllable, d_hidden,
            num_layers=2,
            batch_first=True
        )
        
        self.char_head = nn.Linear(d_hidden, vocab_size)
        
    def forward(self, syllable_embeddings, target_lengths=None):
        """
        syllable_embeddings: [batch, num_syllables, d_syllable]
        Returns: char_logits [batch, num_syllables, max_chars, vocab_size]
        """
        batch_size, num_syllables, d_syllable = syllable_embeddings.shape
        device = syllable_embeddings.device
        
        # Reshape for decoding
        syllables_flat = syllable_embeddings.view(-1, 1, d_syllable)  # [B*S, 1, d_syl]
        
        # Repeat for each output position
        syllables_expanded = syllables_flat.expand(-1, self.max_chars, -1)  # [B*S, max_chars, d_syl]
        
        # Decode
        decoded, _ = self.decoder(syllables_expanded)  # [B*S, max_chars, d_hidden]
        char_logits = self.char_head(decoded)  # [B*S, max_chars, vocab_size]
        
        # Reshape back
        char_logits = char_logits.view(batch_size, num_syllables, self.max_chars, self.vocab_size)
        
        return char_logits


# =============================================================================
# FULL HIERARCHICAL MODEL (Level 0-1)
# =============================================================================

class HierarchicalGermanV1(nn.Module):
    """
    Hierarchical German Model - Version 1
    Levels: Characters → Syllables
    
    Training objectives:
    1. Syllable boundary detection
    2. Syllable reconstruction (syllable embedding → original characters)
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syllable=256, 
                 max_len=128, max_syllables=64):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_char = d_char
        self.d_syllable = d_syllable
        
        # Level 0: Character encoder
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        
        # Level 1: Syllable detector
        self.syllable_detector = SyllableDetector(d_char, d_syllable, max_syllables)
        
        # Syllable decoder (for reconstruction loss)
        self.syllable_decoder = SyllableDecoder(vocab_size, d_syllable)
        
        # Character prediction head (for masked char prediction)
        self.char_predictor = nn.Sequential(
            nn.Linear(d_char, d_char * 2),
            nn.GELU(),
            nn.Linear(d_char * 2, vocab_size)
        )
        
    def forward(self, char_indices, boundary_labels=None):
        """
        char_indices: [batch, seq_len]
        boundary_labels: [batch, seq_len] optional syllable boundary labels
        
        Returns dict with:
            - char_embeddings
            - boundary_logits
            - syllable_embeddings
            - char_predictions (for masked LM)
        """
        # Level 0: Encode characters
        char_embeddings = self.char_encoder(char_indices)
        
        # Level 1: Detect syllables
        boundary_logits, syllable_embeddings, syllable_mask = self.syllable_detector(
            char_embeddings, boundary_labels
        )
        
        # Character prediction (for auxiliary loss)
        char_predictions = self.char_predictor(char_embeddings)
        
        return {
            'char_embeddings': char_embeddings,
            'boundary_logits': boundary_logits,
            'syllable_embeddings': syllable_embeddings,
            'syllable_mask': syllable_mask,
            'char_predictions': char_predictions
        }
    
    def get_syllables(self, text):
        """Helper to get syllable representations for a text."""
        self.eval()
        with torch.no_grad():
            indices = torch.tensor([text_to_indices(text)], device=next(self.parameters()).device)
            outputs = self.forward(indices)
            return outputs['syllable_embeddings'], outputs['syllable_mask']


# =============================================================================
# TRAINING
# =============================================================================

def load_german_sentences(max_sentences=100000):
    """Load German sentences for training."""
    print("Loading German data...")
    sentences = []
    
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_sentences):
            text = item.get('text', item.get('story', ''))
            if isinstance(text, str) and 20 < len(text) < 300:
                sentences.append(text.strip())
                if len(sentences) >= max_sentences:
                    break
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def train_hierarchical_v1(model, sentences, device, epochs=20, batch_size=64, lr=3e-4):
    """Train the hierarchical model."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    
    # Learning rate scheduler
    steps_per_epoch = len(sentences) // batch_size
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, 
                          steps_per_epoch=steps_per_epoch, pct_start=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(sentences)
        
        total_loss = 0
        total_boundary_loss = 0
        total_char_loss = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences) - batch_size, batch_size), 
                   desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch_texts = sentences[i:i+batch_size]
            
            # Create training data
            char_indices_batch = []
            boundary_labels_batch = []
            
            for text in batch_texts:
                char_idx, boundary_lbl = create_syllable_labels(text)
                char_indices_batch.append(char_idx)
                boundary_labels_batch.append(boundary_lbl)
            
            char_indices = torch.tensor(char_indices_batch, device=device)
            boundary_labels = torch.tensor(boundary_labels_batch, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(char_indices, boundary_labels)
                
                # Loss 1: Syllable boundary detection
                boundary_loss = F.binary_cross_entropy_with_logits(
                    outputs['boundary_logits'],
                    boundary_labels,
                    reduction='mean'
                )
                
                # Loss 2: Next character prediction (shifted)
                char_targets = char_indices[:, 1:]
                char_preds = outputs['char_predictions'][:, :-1, :]
                
                char_loss = F.cross_entropy(
                    char_preds.reshape(-1, model.vocab_size),
                    char_targets.reshape(-1),
                    ignore_index=char_to_idx[PAD_TOKEN]
                )
                
                # Combined loss
                loss = boundary_loss + char_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_boundary_loss += boundary_loss.item()
            total_char_loss += char_loss.item()
            num_batches += 1
            
            if num_batches % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'bnd': f'{total_boundary_loss/num_batches:.4f}',
                    'chr': f'{total_char_loss/num_batches:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f} "
              f"(boundary={total_boundary_loss/num_batches:.4f}, "
              f"char={total_char_loss/num_batches:.4f})")
        
        # Test syllabification
        test_syllabification(model, device)
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hierarchical_v1_best.pth")
            print(f"💾 Saved best model")


def test_syllabification(model, device):
    """Test the model's syllabification ability."""
    model.eval()
    
    test_words = [
        "Katze",
        "Möglichkeit", 
        "Freundschaft",
        "Donaudampfschiff",
        "Kindergarten",
        "Schmetterling",
        "Wissenschaft",
        "Geburtstag"
    ]
    
    print("\n📊 Syllabification Test:")
    
    with torch.no_grad():
        for word in test_words:
            # Get model prediction
            indices = torch.tensor([text_to_indices(word, max_len=len(word)+5)], device=device)
            outputs = model(indices)
            
            # Get predicted boundaries
            probs = torch.sigmoid(outputs['boundary_logits'][0]).cpu().numpy()
            
            # Build syllabified word
            syllabified = ""
            for i, c in enumerate(word):
                syllabified += c
                if i < len(probs) and probs[i] > 0.5:
                    syllabified += "-"
            
            # Ground truth
            gt_syllables, _ = syllabify_german(word)
            gt_str = "-".join(gt_syllables)
            
            match = "✓" if syllabified.rstrip("-") == gt_str else "✗"
            print(f"  {word:20} → {syllabified:25} (GT: {gt_str}) {match}")
    
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = HierarchicalGermanV1(
        vocab_size=VOCAB_SIZE,
        d_char=128,
        d_syllable=256,
        max_len=128,
        max_syllables=64
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load data
    sentences = load_german_sentences(max_sentences=100000)
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Train
    print("\n" + "="*60)
    print("TRAINING HIERARCHICAL MODEL (Level 0-1: Chars → Syllables)")
    print("="*60 + "\n")
    
    train_hierarchical_v1(model, sentences, device, epochs=20, batch_size=64, lr=3e-4)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
