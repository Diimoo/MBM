#!/usr/bin/env python3
"""
Hierarchical German Language Model - Version 2
Level 0: Character Encoder
Level 1: Syllable Detector
Level 2: Morpheme Parser (NEW)

Following human learning: chars → syllables → morphemes → (future: words → ...)
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

VOWELS = set('aeiouäöüAEIOUÄÖÜ')

PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

char_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)

idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# =============================================================================
# GERMAN MORPHOLOGY
# =============================================================================

# Common German prefixes
GERMAN_PREFIXES = [
    'un', 'ver', 'be', 'ent', 'er', 'ge', 'miss', 'zer',
    'ab', 'an', 'auf', 'aus', 'bei', 'durch', 'ein', 'mit',
    'nach', 'vor', 'zu', 'über', 'unter', 'wieder', 'hin', 'her'
]

# Common German suffixes (noun, adjective, verb endings)
GERMAN_SUFFIXES = [
    # Noun suffixes
    'ung', 'heit', 'keit', 'schaft', 'nis', 'tum', 'ling', 'chen', 'lein',
    # Adjective suffixes
    'lich', 'ig', 'isch', 'bar', 'sam', 'haft', 'los', 'voll',
    # Verb/participle suffixes
    'en', 'ieren', 'eln', 'ern',
    # Inflection suffixes
    'er', 'es', 'em', 'en', 'e', 'st', 't', 's'
]

# Morpheme type labels
MORPH_TYPES = {
    'PAD': 0,
    'PREFIX': 1,
    'ROOT': 2,
    'SUFFIX': 3,
    'INFLECTION': 4
}

def analyze_morphemes(word):
    """
    Rule-based German morpheme analysis.
    Returns list of (morpheme, type) tuples.
    """
    word_lower = word.lower()
    morphemes = []
    remaining = word_lower
    
    # Check for prefixes
    for prefix in sorted(GERMAN_PREFIXES, key=len, reverse=True):
        if remaining.startswith(prefix) and len(remaining) > len(prefix) + 2:
            morphemes.append((prefix, 'PREFIX'))
            remaining = remaining[len(prefix):]
            break
    
    # Check for suffixes (from end)
    suffix_found = None
    for suffix in sorted(GERMAN_SUFFIXES, key=len, reverse=True):
        if remaining.endswith(suffix) and len(remaining) > len(suffix) + 2:
            suffix_found = (suffix, 'SUFFIX')
            remaining = remaining[:-len(suffix)]
            break
    
    # What's left is the root
    if remaining:
        morphemes.append((remaining, 'ROOT'))
    
    # Add suffix at the end
    if suffix_found:
        morphemes.append(suffix_found)
    
    return morphemes


def create_morpheme_labels(word, max_len=32):
    """
    Create morpheme boundary and type labels for a word.
    Returns: (char_indices, morpheme_boundaries, morpheme_types)
    """
    morphemes = analyze_morphemes(word)
    
    char_indices = []
    boundaries = []  # 1 = morpheme boundary after this char
    types = []  # type of current morpheme
    
    for morph, morph_type in morphemes:
        type_id = MORPH_TYPES[morph_type]
        for i, c in enumerate(morph):
            char_indices.append(char_to_idx.get(c, char_to_idx[UNK_TOKEN]))
            types.append(type_id)
            # Boundary at end of morpheme
            boundaries.append(1 if i == len(morph) - 1 else 0)
    
    # Pad
    while len(char_indices) < max_len:
        char_indices.append(char_to_idx[PAD_TOKEN])
        boundaries.append(0)
        types.append(MORPH_TYPES['PAD'])
    
    return char_indices[:max_len], boundaries[:max_len], types[:max_len]


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
        self.d_syllable = d_syllable
        
        self.boundary_lstm = nn.LSTM(
            d_char, d_char // 2, 
            num_layers=2, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.1
        )
        self.boundary_head = nn.Sequential(
            nn.Linear(d_char, d_char),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_char, 1)
        )
        
    def forward(self, char_embeddings):
        lstm_out, _ = self.boundary_lstm(char_embeddings)
        boundary_logits = self.boundary_head(lstm_out).squeeze(-1)
        return boundary_logits


# =============================================================================
# LEVEL 2: MORPHEME PARSER (NEW)
# =============================================================================

class MorphemeParser(nn.Module):
    """
    Level 2: Parse syllables into morphemes.
    
    Identifies:
    - Prefixes (un-, ver-, be-, etc.)
    - Roots (semantic core)
    - Suffixes (-ung, -heit, -lich, etc.)
    - Inflections (-en, -er, -es, etc.)
    """
    def __init__(self, d_input=128, d_morpheme=256, num_types=5):
        super().__init__()
        self.d_morpheme = d_morpheme
        self.num_types = num_types
        
        # Morpheme boundary detection
        self.morph_lstm = nn.LSTM(
            d_input, d_input,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # Boundary prediction
        self.boundary_head = nn.Sequential(
            nn.Linear(d_input * 2, d_input),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_input, 1)
        )
        
        # Morpheme type classification (PREFIX, ROOT, SUFFIX, INFLECTION)
        self.type_head = nn.Sequential(
            nn.Linear(d_input * 2, d_input),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_input, num_types)
        )
        
        # Morpheme embedding projection
        self.morph_project = nn.Sequential(
            nn.Linear(d_input * 2, d_morpheme),
            nn.LayerNorm(d_morpheme),
            nn.GELU()
        )
        
    def forward(self, char_embeddings):
        """
        char_embeddings: [batch, seq_len, d_char]
        
        Returns:
            boundary_logits: [batch, seq_len]
            type_logits: [batch, seq_len, num_types]
            morpheme_embeddings: [batch, seq_len, d_morpheme]
        """
        lstm_out, _ = self.morph_lstm(char_embeddings)
        
        boundary_logits = self.boundary_head(lstm_out).squeeze(-1)
        type_logits = self.type_head(lstm_out)
        morpheme_embeddings = self.morph_project(lstm_out)
        
        return boundary_logits, type_logits, morpheme_embeddings


# =============================================================================
# FULL HIERARCHICAL MODEL (Level 0-2)
# =============================================================================

class HierarchicalGermanV2(nn.Module):
    """
    Hierarchical German Model - Version 2
    Levels: Characters → Syllables → Morphemes
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syllable=256, 
                 d_morpheme=256, max_len=128):
        super().__init__()
        
        self.vocab_size = vocab_size
        
        # Level 0: Character encoder
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        
        # Level 1: Syllable detector
        self.syllable_detector = SyllableDetector(d_char, d_syllable)
        
        # Level 2: Morpheme parser
        self.morpheme_parser = MorphemeParser(d_char, d_morpheme, num_types=5)
        
        # Character prediction (auxiliary)
        self.char_predictor = nn.Sequential(
            nn.Linear(d_char, d_char * 2),
            nn.GELU(),
            nn.Linear(d_char * 2, vocab_size)
        )
        
    def forward(self, char_indices):
        # Level 0: Encode characters
        char_embeddings = self.char_encoder(char_indices)
        
        # Level 1: Syllable boundaries
        syllable_logits = self.syllable_detector(char_embeddings)
        
        # Level 2: Morpheme parsing
        morph_boundary_logits, morph_type_logits, morph_embeddings = \
            self.morpheme_parser(char_embeddings)
        
        # Character prediction
        char_predictions = self.char_predictor(char_embeddings)
        
        return {
            'char_embeddings': char_embeddings,
            'syllable_logits': syllable_logits,
            'morph_boundary_logits': morph_boundary_logits,
            'morph_type_logits': morph_type_logits,
            'morph_embeddings': morph_embeddings,
            'char_predictions': char_predictions
        }


# =============================================================================
# TRAINING
# =============================================================================

def load_german_words(max_words=50000):
    """Load German words for morpheme training."""
    print("Loading German words...")
    words = set()
    
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_words * 10):
            text = item.get('text', item.get('story', ''))
            if isinstance(text, str):
                # Extract words
                for word in re.findall(r'[A-Za-zäöüßÄÖÜ]+', text):
                    if 4 <= len(word) <= 20:  # Focus on medium-length words
                        words.add(word.lower())
                        if len(words) >= max_words:
                            break
            if len(words) >= max_words:
                break
    except Exception as e:
        print(f"Error: {e}")
    
    words = list(words)
    print(f"Loaded {len(words)} unique words")
    return words


def train_hierarchical_v2(model, words, device, epochs=20, batch_size=128, lr=3e-4):
    """Train the hierarchical model with morpheme parsing."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    
    steps_per_epoch = len(words) // batch_size
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, 
                          steps_per_epoch=steps_per_epoch, pct_start=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(words)
        
        total_loss = 0
        total_morph_bnd = 0
        total_morph_type = 0
        total_char = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(words) - batch_size, batch_size), 
                   desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch_words = words[i:i+batch_size]
            
            # Create training data
            char_batch = []
            morph_bnd_batch = []
            morph_type_batch = []
            
            for word in batch_words:
                chars, boundaries, types = create_morpheme_labels(word, max_len=32)
                char_batch.append(chars)
                morph_bnd_batch.append(boundaries)
                morph_type_batch.append(types)
            
            char_indices = torch.tensor(char_batch, device=device)
            morph_boundaries = torch.tensor(morph_bnd_batch, dtype=torch.float, device=device)
            morph_types = torch.tensor(morph_type_batch, dtype=torch.long, device=device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(char_indices)
                
                # Loss 1: Morpheme boundary detection
                morph_bnd_loss = F.binary_cross_entropy_with_logits(
                    outputs['morph_boundary_logits'],
                    morph_boundaries
                )
                
                # Loss 2: Morpheme type classification
                morph_type_loss = F.cross_entropy(
                    outputs['morph_type_logits'].view(-1, 5),
                    morph_types.view(-1),
                    ignore_index=0  # Ignore PAD
                )
                
                # Loss 3: Next character prediction
                char_targets = char_indices[:, 1:]
                char_preds = outputs['char_predictions'][:, :-1, :]
                char_loss = F.cross_entropy(
                    char_preds.reshape(-1, model.vocab_size),
                    char_targets.reshape(-1),
                    ignore_index=0
                )
                
                loss = morph_bnd_loss + morph_type_loss + 0.5 * char_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_morph_bnd += morph_bnd_loss.item()
            total_morph_type += morph_type_loss.item()
            total_char += char_loss.item()
            num_batches += 1
            
            if num_batches % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'bnd': f'{total_morph_bnd/num_batches:.4f}',
                    'type': f'{total_morph_type/num_batches:.4f}'
                })
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        # Test morpheme parsing
        test_morpheme_parsing(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hierarchical_v2_best.pth")
            print(f"💾 Saved best model")


def test_morpheme_parsing(model, device):
    """Test morpheme parsing on sample words."""
    model.eval()
    
    test_words = [
        "unmöglich",      # un-mög-lich
        "Freundschaft",   # Freund-schaft
        "Möglichkeit",    # Mög-lich-keit
        "verstehen",      # ver-steh-en
        "Wissenschaft",   # Wissen-schaft
        "bearbeiten",     # be-arbeit-en
        "Kindergarten",   # Kinder-garten
        "unglaublich",    # un-glaub-lich
    ]
    
    type_names = ['PAD', 'PREFIX', 'ROOT', 'SUFFIX', 'INFL']
    
    print("\n📊 Morpheme Parsing Test:")
    
    with torch.no_grad():
        for word in test_words:
            chars, _, _ = create_morpheme_labels(word, max_len=len(word)+5)
            indices = torch.tensor([chars[:len(word)]], device=device)
            
            # Pad to match expected length
            if indices.shape[1] < 32:
                pad = torch.zeros(1, 32 - indices.shape[1], dtype=torch.long, device=device)
                indices = torch.cat([indices, pad], dim=1)
            
            outputs = model(indices)
            
            # Get predictions
            bnd_probs = torch.sigmoid(outputs['morph_boundary_logits'][0]).cpu().numpy()
            type_preds = outputs['morph_type_logits'][0].argmax(dim=-1).cpu().numpy()
            
            # Build parsed word
            parsed = ""
            current_type = None
            for i, c in enumerate(word.lower()):
                t = type_names[type_preds[i]] if i < len(type_preds) else '?'
                if current_type != t:
                    if current_type:
                        parsed += f"({current_type})-"
                    current_type = t
                parsed += c
                if i < len(bnd_probs) and bnd_probs[i] > 0.5:
                    parsed += f"({current_type})-"
                    current_type = None
            if current_type:
                parsed += f"({current_type})"
            
            # Ground truth
            gt = analyze_morphemes(word)
            gt_str = "-".join([f"{m}({t})" for m, t in gt])
            
            print(f"  {word:18} → {parsed[:40]:42} GT: {gt_str}")
    
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    
    # Create model
    model = HierarchicalGermanV2(
        vocab_size=VOCAB_SIZE,
        d_char=128,
        d_syllable=256,
        d_morpheme=256,
        max_len=32
    ).to(device)
    
    # Load v1 weights if available
    v1_path = "checkpoints/hierarchical_v1_best.pth"
    if os.path.exists(v1_path):
        print(f"Loading Level 0-1 weights from {v1_path}")
        v1_state = torch.load(v1_path, map_location=device)
        # Load only matching keys
        model_state = model.state_dict()
        for k, v in v1_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
        model.load_state_dict(model_state, strict=False)
        print("Loaded pre-trained char encoder and syllable detector")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load data
    words = load_german_words(max_words=50000)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING HIERARCHICAL MODEL (Level 0-2: Chars → Syllables → Morphemes)")
    print("="*60 + "\n")
    
    train_hierarchical_v2(model, words, device, epochs=20, batch_size=128, lr=3e-4)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
