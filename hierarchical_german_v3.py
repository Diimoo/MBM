#!/usr/bin/env python3
"""
Hierarchical German Language Model - Version 3
Level 0: Character Encoder
Level 1: Syllable Detector  
Level 2: Morpheme Parser
Level 3: Word Composer (NEW)

Following human learning: chars → syllables → morphemes → words → (future: phrases → ...)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from datasets import load_dataset
from tqdm import tqdm
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

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

char_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)

idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# Morpheme types
MORPH_TYPES = {'PAD': 0, 'PREFIX': 1, 'ROOT': 2, 'SUFFIX': 3, 'INFLECTION': 4}

# German prefixes and suffixes
GERMAN_PREFIXES = [
    'un', 'ver', 'be', 'ent', 'er', 'ge', 'miss', 'zer',
    'ab', 'an', 'auf', 'aus', 'bei', 'durch', 'ein', 'mit',
    'nach', 'vor', 'zu', 'über', 'unter', 'wieder', 'hin', 'her'
]

GERMAN_SUFFIXES = [
    'ung', 'heit', 'keit', 'schaft', 'nis', 'tum', 'ling', 'chen', 'lein',
    'lich', 'ig', 'isch', 'bar', 'sam', 'haft', 'los', 'voll',
    'en', 'ieren', 'eln', 'ern', 'er', 'es', 'em', 'e', 'st', 't', 's'
]


def text_to_indices(text, max_len=128):
    indices = [char_to_idx.get(c, char_to_idx[UNK_TOKEN]) for c in text[:max_len]]
    while len(indices) < max_len:
        indices.append(char_to_idx[PAD_TOKEN])
    return indices


# =============================================================================
# LEVEL 0: CHARACTER ENCODER
# =============================================================================

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, d_char=128, max_len=256):
        super().__init__()
        self.d_char = d_char
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_char)
        self.local_context = nn.Sequential(
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_char)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device
        char_emb = self.char_embed(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_embed(positions)
        x = char_emb + pos_emb
        x_conv = x.transpose(1, 2)
        x_conv = self.local_context(x_conv)
        x_conv = x_conv.transpose(1, 2)
        x = self.norm(x + x_conv)
        return self.dropout(x)


# =============================================================================
# LEVEL 1: SYLLABLE DETECTOR
# =============================================================================

class SyllableDetector(nn.Module):
    def __init__(self, d_char=128, d_syllable=256):
        super().__init__()
        self.boundary_lstm = nn.LSTM(
            d_char, d_char // 2, num_layers=2, 
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(d_char, d_char), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_char, 1)
        )
        
    def forward(self, char_embeddings):
        lstm_out, _ = self.boundary_lstm(char_embeddings)
        return self.boundary_head(lstm_out).squeeze(-1)


# =============================================================================
# LEVEL 2: MORPHEME PARSER
# =============================================================================

class MorphemeParser(nn.Module):
    def __init__(self, d_input=128, d_morpheme=256, num_types=5):
        super().__init__()
        self.d_morpheme = d_morpheme
        
        self.morph_lstm = nn.LSTM(
            d_input, d_input, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(d_input * 2, d_input), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_input, 1)
        )
        self.type_head = nn.Sequential(
            nn.Linear(d_input * 2, d_input), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_input, num_types)
        )
        self.morph_project = nn.Sequential(
            nn.Linear(d_input * 2, d_morpheme), nn.LayerNorm(d_morpheme), nn.GELU()
        )
        
    def forward(self, char_embeddings):
        lstm_out, _ = self.morph_lstm(char_embeddings)
        boundary_logits = self.boundary_head(lstm_out).squeeze(-1)
        type_logits = self.type_head(lstm_out)
        morpheme_embeddings = self.morph_project(lstm_out)
        return boundary_logits, type_logits, morpheme_embeddings


# =============================================================================
# LEVEL 3: WORD COMPOSER (NEW)
# =============================================================================

class WordComposer(nn.Module):
    """
    Level 3: Compose morphemes into word-level meaning.
    
    Key insight for German:
    - Compound words: meaning = composition of parts
    - Prefixes modify meaning systematically (un- = negation)
    - Suffixes change word class (verb→noun, adj→adv)
    
    Uses attention to weight morpheme contributions.
    """
    def __init__(self, d_morpheme=256, d_word=512, num_heads=8, max_morphemes=16):
        super().__init__()
        self.d_word = d_word
        self.max_morphemes = max_morphemes
        
        # Morpheme type embeddings (to encode prefix/root/suffix role)
        self.type_embed = nn.Embedding(5, d_morpheme)  # 5 morpheme types
        
        # Self-attention over morphemes within a word
        self.morpheme_attention = nn.MultiheadAttention(
            d_morpheme, num_heads, dropout=0.1, batch_first=True
        )
        
        # Composition network - combines morphemes into word meaning
        self.composer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_morpheme, nhead=num_heads, 
                dim_feedforward=d_morpheme * 4, dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Project to word embedding space
        self.word_project = nn.Sequential(
            nn.Linear(d_morpheme, d_word),
            nn.LayerNorm(d_word),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Word boundary detection (spaces, punctuation)
        self.word_boundary_head = nn.Sequential(
            nn.Linear(d_morpheme, d_morpheme // 2),
            nn.GELU(),
            nn.Linear(d_morpheme // 2, 1)
        )
        
    def forward(self, morpheme_embeddings, morpheme_types=None, morpheme_mask=None):
        """
        morpheme_embeddings: [batch, seq_len, d_morpheme]
        morpheme_types: [batch, seq_len] - type IDs (PREFIX, ROOT, SUFFIX, etc.)
        morpheme_mask: [batch, seq_len] - 1 for valid positions
        
        Returns:
            word_embeddings: [batch, seq_len, d_word]
            word_boundary_logits: [batch, seq_len]
        """
        batch_size, seq_len, d_morph = morpheme_embeddings.shape
        device = morpheme_embeddings.device
        
        # Add morpheme type information if available
        if morpheme_types is not None:
            type_emb = self.type_embed(morpheme_types)
            morpheme_embeddings = morpheme_embeddings + type_emb
        
        # Create attention mask for padding
        if morpheme_mask is not None:
            key_padding_mask = (morpheme_mask == 0)
        else:
            key_padding_mask = None
        
        # Self-attention over morphemes
        attended, _ = self.morpheme_attention(
            morpheme_embeddings, morpheme_embeddings, morpheme_embeddings,
            key_padding_mask=key_padding_mask
        )
        
        # Compose morphemes (transformer processes the sequence)
        composed = self.composer(attended)
        
        # Project to word space
        word_embeddings = self.word_project(composed)
        
        # Detect word boundaries
        word_boundary_logits = self.word_boundary_head(composed).squeeze(-1)
        
        return word_embeddings, word_boundary_logits


# =============================================================================
# FULL HIERARCHICAL MODEL (Level 0-3)
# =============================================================================

class HierarchicalGermanV3(nn.Module):
    """
    Hierarchical German Model - Version 3
    Levels: Characters → Syllables → Morphemes → Words
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syllable=256, 
                 d_morpheme=256, d_word=512, max_len=128):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_word = d_word
        
        # Level 0: Character encoder
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        
        # Level 1: Syllable detector
        self.syllable_detector = SyllableDetector(d_char, d_syllable)
        
        # Level 2: Morpheme parser
        self.morpheme_parser = MorphemeParser(d_char, d_morpheme, num_types=5)
        
        # Level 3: Word composer
        self.word_composer = WordComposer(d_morpheme, d_word, num_heads=8)
        
        # Character prediction (auxiliary)
        self.char_predictor = nn.Sequential(
            nn.Linear(d_char, d_char * 2),
            nn.GELU(),
            nn.Linear(d_char * 2, vocab_size)
        )
        
        # Word-level next word prediction (using word embeddings)
        self.word_predictor = nn.Sequential(
            nn.Linear(d_word, d_word),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_word, d_word)  # Predict next word embedding
        )
        
    def forward(self, char_indices, morpheme_types=None):
        # Level 0: Encode characters
        char_embeddings = self.char_encoder(char_indices)
        
        # Level 1: Syllable boundaries
        syllable_logits = self.syllable_detector(char_embeddings)
        
        # Level 2: Morpheme parsing
        morph_boundary_logits, morph_type_logits, morph_embeddings = \
            self.morpheme_parser(char_embeddings)
        
        # Get predicted morpheme types if not provided
        if morpheme_types is None:
            morpheme_types = morph_type_logits.argmax(dim=-1)
        
        # Level 3: Word composition
        word_embeddings, word_boundary_logits = self.word_composer(
            morph_embeddings, morpheme_types
        )
        
        # Character prediction
        char_predictions = self.char_predictor(char_embeddings)
        
        # Word prediction (predict next word embedding)
        word_predictions = self.word_predictor(word_embeddings)
        
        return {
            'char_embeddings': char_embeddings,
            'syllable_logits': syllable_logits,
            'morph_boundary_logits': morph_boundary_logits,
            'morph_type_logits': morph_type_logits,
            'morph_embeddings': morph_embeddings,
            'word_embeddings': word_embeddings,
            'word_boundary_logits': word_boundary_logits,
            'char_predictions': char_predictions,
            'word_predictions': word_predictions
        }


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_sentence_data(sentence, max_len=64):
    """
    Prepare sentence for training.
    Returns char indices and word boundary labels.
    """
    # Character indices
    chars = text_to_indices(sentence.lower(), max_len)
    
    # Word boundaries (1 after space or punctuation)
    boundaries = []
    for i, c in enumerate(sentence[:max_len].lower()):
        if c in ' .,!?;:\n\t':
            boundaries.append(1)
        else:
            boundaries.append(0)
    
    while len(boundaries) < max_len:
        boundaries.append(0)
    
    return chars, boundaries[:max_len]


def load_german_sentences(max_sentences=50000):
    """Load German sentences for training."""
    print("Loading German sentences...")
    sentences = []
    
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_sentences * 5):
            text = item.get('text', item.get('story', ''))
            if isinstance(text, str):
                # Split into sentences
                for sent in re.split(r'[.!?]+', text):
                    sent = sent.strip()
                    if 10 <= len(sent) <= 100:
                        sentences.append(sent)
                        if len(sentences) >= max_sentences:
                            break
            if len(sentences) >= max_sentences:
                break
    except Exception as e:
        print(f"Error: {e}")
    
    print(f"Loaded {len(sentences)} sentences")
    return sentences


# =============================================================================
# TRAINING
# =============================================================================

def train_hierarchical_v3(model, sentences, device, epochs=20, batch_size=64, lr=3e-4):
    """Train the hierarchical model with word composition."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    
    steps_per_epoch = len(sentences) // batch_size
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, 
                          steps_per_epoch=steps_per_epoch, pct_start=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(sentences)
        
        total_loss = 0
        total_word_bnd = 0
        total_word_pred = 0
        total_char = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences) - batch_size, batch_size), 
                   desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch_sentences = sentences[i:i+batch_size]
            
            # Prepare batch
            char_batch = []
            word_bnd_batch = []
            
            for sent in batch_sentences:
                chars, boundaries = prepare_sentence_data(sent, max_len=64)
                char_batch.append(chars)
                word_bnd_batch.append(boundaries)
            
            char_indices = torch.tensor(char_batch, device=device)
            word_boundaries = torch.tensor(word_bnd_batch, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(char_indices)
                
                # Loss 1: Word boundary detection
                word_bnd_loss = F.binary_cross_entropy_with_logits(
                    outputs['word_boundary_logits'],
                    word_boundaries
                )
                
                # Loss 2: Word embedding prediction (contrastive)
                # Predict next word embedding should be close to actual next word
                word_emb = outputs['word_embeddings']
                word_pred = outputs['word_predictions']
                
                # Shift for next-word prediction
                target_emb = word_emb[:, 1:, :]  # Next positions
                pred_emb = word_pred[:, :-1, :]  # Predictions
                
                # Cosine similarity loss (predicted should match target)
                word_pred_loss = 1 - F.cosine_similarity(
                    pred_emb.reshape(-1, model.d_word),
                    target_emb.reshape(-1, model.d_word),
                    dim=-1
                ).mean()
                
                # Loss 3: Character prediction
                char_targets = char_indices[:, 1:]
                char_preds = outputs['char_predictions'][:, :-1, :]
                char_loss = F.cross_entropy(
                    char_preds.reshape(-1, model.vocab_size),
                    char_targets.reshape(-1),
                    ignore_index=0
                )
                
                loss = word_bnd_loss + word_pred_loss + 0.5 * char_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_word_bnd += word_bnd_loss.item()
            total_word_pred += word_pred_loss.item()
            total_char += char_loss.item()
            num_batches += 1
            
            if num_batches % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'wbnd': f'{total_word_bnd/num_batches:.4f}',
                    'wprd': f'{total_word_pred/num_batches:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        # Test word segmentation
        test_word_segmentation(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hierarchical_v3_best.pth")
            print(f"💾 Saved best model")


def test_word_segmentation(model, device):
    """Test word boundary detection on sample sentences."""
    model.eval()
    
    test_sentences = [
        "Die Katze sitzt auf dem Dach",
        "Der kleine Junge spielt im Garten",
        "Heute ist ein schöner Tag",
        "Ich gehe nach Hause",
    ]
    
    print("\n📊 Word Segmentation Test:")
    
    with torch.no_grad():
        for sent in test_sentences:
            chars, _ = prepare_sentence_data(sent, max_len=len(sent)+5)
            indices = torch.tensor([chars[:len(sent)]], device=device)
            
            # Pad
            if indices.shape[1] < 64:
                pad = torch.zeros(1, 64 - indices.shape[1], dtype=torch.long, device=device)
                indices = torch.cat([indices, pad], dim=1)
            
            outputs = model(indices)
            
            bnd_probs = torch.sigmoid(outputs['word_boundary_logits'][0]).cpu().numpy()
            
            # Reconstruct with word markers
            segmented = ""
            for i, c in enumerate(sent.lower()):
                segmented += c
                if i < len(bnd_probs) and bnd_probs[i] > 0.5 and c != ' ':
                    segmented += "|"
            
            print(f"  '{sent}' → '{segmented}'")
    
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    
    # Create model
    model = HierarchicalGermanV3(
        vocab_size=VOCAB_SIZE,
        d_char=128,
        d_syllable=256,
        d_morpheme=256,
        d_word=512,
        max_len=64
    ).to(device)
    
    # Load v2 weights if available
    v2_path = "checkpoints/hierarchical_v2_best.pth"
    if os.path.exists(v2_path):
        print(f"Loading Level 0-2 weights from {v2_path}")
        v2_state = torch.load(v2_path, map_location=device)
        model_state = model.state_dict()
        loaded = 0
        for k, v in v2_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {loaded} layers from v2")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load data
    sentences = load_german_sentences(max_sentences=50000)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING HIERARCHICAL MODEL (Level 0-3: Chars → Words)")
    print("="*60 + "\n")
    
    train_hierarchical_v3(model, sentences, device, epochs=20, batch_size=64, lr=3e-4)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
