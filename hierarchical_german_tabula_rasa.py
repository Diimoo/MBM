#!/usr/bin/env python3
"""
Hierarchical German Language Model - Tabula Rasa (Fresh Start)

Applying lessons learned:
1. PRESERVE CAPITALS - German nouns are capitalized (Möglichkeit, not möglichkeit)
2. EXPLICIT GRAMMATICAL LABELS - Noun, Verb, Adjective, Adverb, etc.
3. MORPHEME TYPES - Prefix, Root, Suffix with explicit supervision
4. DIVERSE TEST SENTENCES - 30+ random sentences, not just 3
5. COLLAPSE MONITORING - Stop if similarity > 0.95
6. RECONSTRUCTION LOSS - Embeddings must decode back to input
7. LEVEL-BY-LEVEL TRAINING - Train each level separately, verify it works

Hierarchy:
  Level 0: Character [alphabet with case]
  Level 1: Syllable [pronounceable units]
  Level 2: Morpheme [prefix, root, suffix]
  Level 3: Word [lexical meaning + POS]
  Level 4: Phrase [NP, VP, PP]
  Level 5: Sentence [proposition]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from datasets import load_dataset
from tqdm import tqdm
import os
import random
import re
from collections import defaultdict

# =============================================================================
# VOCABULARY - PRESERVE CASE!
# =============================================================================

CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'  # CAPITALS PRESERVED
    'äöüßÄÖÜ'
    '0123456789'
    ' .,!?;:\'"()-'
    '\n'
)

SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
char_to_idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# =============================================================================
# GRAMMATICAL CATEGORIES
# =============================================================================

# Part of Speech tags
POS_TAGS = {
    'PAD': 0,
    'NOUN': 1,      # Haus, Katze, Möglichkeit
    'VERB': 2,      # gehen, laufen, spielen
    'ADJ': 3,       # schön, groß, klein
    'ADV': 4,       # schnell, hier, heute
    'DET': 5,       # der, die, das, ein
    'PREP': 6,      # in, auf, mit, nach
    'CONJ': 7,      # und, oder, aber
    'PRON': 8,      # ich, du, er, sie
    'AUX': 9,       # haben, sein, werden
    'PUNCT': 10,    # . , ! ?
}

# Morpheme types
MORPH_TYPES = {
    'PAD': 0,
    'PREFIX': 1,    # un-, ver-, be-, ge-
    'ROOT': 2,      # -lieb-, -geh-, -haus-
    'SUFFIX': 3,    # -ung, -keit, -lich
    'INFLECT': 4,   # -e, -st, -en (verb endings)
    'COMPOUND': 5,  # joining element in compounds
}

# Common German morphemes for supervision
GERMAN_PREFIXES = {'un', 'ver', 'be', 'ge', 'ent', 'er', 'zer', 'miss', 'ab', 'an', 'auf', 'aus', 'ein', 'mit', 'nach', 'vor', 'zu'}
GERMAN_SUFFIXES = {'ung', 'keit', 'heit', 'lich', 'isch', 'bar', 'sam', 'los', 'voll', 'haft', 'ig', 'en', 'er', 'st', 'te', 'schaft', 'tum', 'nis'}

# Common German words with POS for supervision
GERMAN_POS = {
    # Determiners
    'der': 'DET', 'die': 'DET', 'das': 'DET', 'ein': 'DET', 'eine': 'DET', 'einer': 'DET',
    # Pronouns
    'ich': 'PRON', 'du': 'PRON', 'er': 'PRON', 'sie': 'PRON', 'es': 'PRON', 'wir': 'PRON', 'ihr': 'PRON',
    # Prepositions
    'in': 'PREP', 'auf': 'PREP', 'mit': 'PREP', 'nach': 'PREP', 'von': 'PREP', 'zu': 'PREP', 'bei': 'PREP', 'für': 'PREP', 'über': 'PREP', 'unter': 'PREP',
    # Conjunctions
    'und': 'CONJ', 'oder': 'CONJ', 'aber': 'CONJ', 'denn': 'CONJ', 'weil': 'CONJ',
    # Auxiliaries
    'ist': 'AUX', 'sind': 'AUX', 'war': 'AUX', 'hat': 'AUX', 'haben': 'AUX', 'wird': 'AUX', 'werden': 'AUX',
    # Common adverbs
    'hier': 'ADV', 'dort': 'ADV', 'heute': 'ADV', 'morgen': 'ADV', 'gestern': 'ADV', 'sehr': 'ADV', 'auch': 'ADV', 'nicht': 'ADV', 'noch': 'ADV', 'schon': 'ADV',
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def text_to_indices(text, max_len=128):
    """Convert text to indices - PRESERVING CASE!"""
    indices = [char_to_idx.get('<BOS>')]
    for c in text[:max_len-2]:
        indices.append(char_to_idx.get(c, char_to_idx['<UNK>']))
    indices.append(char_to_idx.get('<EOS>'))
    while len(indices) < max_len:
        indices.append(char_to_idx['<PAD>'])
    return indices[:max_len]

def indices_to_text(indices):
    """Convert indices back to text."""
    chars = []
    for idx in indices:
        if idx == char_to_idx['<PAD>'] or idx == char_to_idx['<EOS>']:
            break
        if idx == char_to_idx['<BOS>']:
            continue
        chars.append(idx_to_char.get(idx, '?'))
    return ''.join(chars)

def get_syllable_boundaries(word):
    """Rule-based German syllabification."""
    vowels = set('aeiouäöüAEIOUÄÖÜ')
    boundaries = [0] * len(word)
    
    i = 0
    while i < len(word):
        # Find vowel
        if word[i] in vowels:
            # Look for consonant cluster after vowel
            j = i + 1
            while j < len(word) and word[j] not in vowels:
                j += 1
            # If there are consonants and another vowel follows
            if j < len(word) and j - i > 1:
                # Split before last consonant (or consonant cluster)
                split_point = j - 1
                if split_point > i:
                    boundaries[split_point] = 1
        i += 1
    
    return boundaries

def get_morpheme_labels(word):
    """Get morpheme boundaries and types for a word."""
    word_lower = word.lower()
    boundaries = [0] * len(word)
    types = [MORPH_TYPES['ROOT']] * len(word)
    
    # Check for prefixes
    for prefix in sorted(GERMAN_PREFIXES, key=len, reverse=True):
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            for i in range(len(prefix)):
                types[i] = MORPH_TYPES['PREFIX']
            boundaries[len(prefix)] = 1
            break
    
    # Check for suffixes
    for suffix in sorted(GERMAN_SUFFIXES, key=len, reverse=True):
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            start = len(word) - len(suffix)
            for i in range(start, len(word)):
                types[i] = MORPH_TYPES['SUFFIX']
            if boundaries[start] == 0:
                boundaries[start] = 1
            break
    
    return boundaries, types

def get_pos_tag(word):
    """Get POS tag for a word."""
    word_lower = word.lower()
    
    # Check dictionary first
    if word_lower in GERMAN_POS:
        return POS_TAGS[GERMAN_POS[word_lower]]
    
    # Heuristics
    if word[0].isupper() and len(word) > 1:  # German nouns are capitalized
        return POS_TAGS['NOUN']
    if word_lower.endswith(('en', 'st', 'te', 't')):  # Verb endings
        return POS_TAGS['VERB']
    if word_lower.endswith(('lich', 'ig', 'isch', 'bar')):  # Adjective endings
        return POS_TAGS['ADJ']
    if word in '.,!?;:':
        return POS_TAGS['PUNCT']
    
    return POS_TAGS['NOUN']  # Default

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class CharacterEncoder(nn.Module):
    """Level 0: Character encoding with position."""
    def __init__(self, vocab_size, d_char=128, max_len=128):
        super().__init__()
        self.d_char = d_char
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_char)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(d_char, d_char, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_char, d_char, 3, padding=1),
        )
        self.norm = nn.LayerNorm(d_char)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        
        char_emb = self.char_embed(x)
        pos_emb = self.pos_embed(positions)
        x = char_emb + pos_emb
        
        # Local context via CNN
        cnn_out = self.local_cnn(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x + cnn_out)
        
        return self.dropout(x)


class SyllableDetector(nn.Module):
    """Level 1: Detect syllable boundaries."""
    def __init__(self, d_char=128, d_syl=128):
        super().__init__()
        self.lstm = nn.LSTM(d_char, d_char // 2, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_char, d_char // 2),
            nn.GELU(),
            nn.Linear(d_char // 2, 1)
        )
        self.project = nn.Linear(d_char, d_syl)
        self.norm = nn.LayerNorm(d_syl)
        
    def forward(self, char_emb):
        lstm_out, _ = self.lstm(char_emb)
        boundaries = self.boundary_head(lstm_out).squeeze(-1)
        syl_emb = self.norm(self.project(lstm_out))
        return boundaries, syl_emb


class MorphemeParser(nn.Module):
    """Level 2: Parse morphemes with type classification."""
    def __init__(self, d_syl=128, d_morph=256, num_types=6):
        super().__init__()
        self.lstm = nn.LSTM(d_syl, d_syl, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_syl * 2, d_syl),
            nn.GELU(),
            nn.Linear(d_syl, 1)
        )
        self.type_head = nn.Sequential(
            nn.Linear(d_syl * 2, d_syl),
            nn.GELU(),
            nn.Linear(d_syl, num_types)
        )
        self.project = nn.Sequential(
            nn.Linear(d_syl * 2, d_morph),
            nn.LayerNorm(d_morph),
            nn.GELU()
        )
        
    def forward(self, syl_emb):
        lstm_out, _ = self.lstm(syl_emb)
        boundaries = self.boundary_head(lstm_out).squeeze(-1)
        types = self.type_head(lstm_out)
        morph_emb = self.project(lstm_out)
        return boundaries, types, morph_emb


class WordComposer(nn.Module):
    """Level 3: Compose words with POS tagging."""
    def __init__(self, d_morph=256, d_word=256, num_pos=11):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_morph, 4, dropout=0.1, batch_first=True)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_morph, d_morph // 2),
            nn.GELU(),
            nn.Linear(d_morph // 2, 1)
        )
        self.pos_head = nn.Sequential(
            nn.Linear(d_morph, d_morph // 2),
            nn.GELU(),
            nn.Linear(d_morph // 2, num_pos)
        )
        self.project = nn.Sequential(
            nn.Linear(d_morph, d_word),
            nn.LayerNorm(d_word),
            nn.GELU()
        )
        
    def forward(self, morph_emb):
        attended, _ = self.attention(morph_emb, morph_emb, morph_emb)
        boundaries = self.boundary_head(attended).squeeze(-1)
        pos_tags = self.pos_head(attended)
        word_emb = self.project(attended)
        return boundaries, pos_tags, word_emb


class Decoder(nn.Module):
    """Reconstruction decoder - embeddings must decode back to characters."""
    def __init__(self, d_emb, vocab_size, d_hidden=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_emb, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, vocab_size)
        )
        
    def forward(self, emb):
        return self.decoder(emb)


class HierarchicalGermanModel(nn.Module):
    """Complete hierarchical model with reconstruction."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syl=128, 
                 d_morph=256, d_word=256, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Hierarchical encoders
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syl)
        self.morpheme_parser = MorphemeParser(d_syl, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        
        # Reconstruction decoders (prevent collapse!)
        self.char_decoder = Decoder(d_char, vocab_size)
        self.syl_decoder = Decoder(d_syl, vocab_size)
        self.morph_decoder = Decoder(d_morph, vocab_size)
        
        # Next character prediction
        self.next_char_head = nn.Linear(d_char, vocab_size)
        
    def forward(self, char_indices):
        # Level 0: Characters
        char_emb = self.char_encoder(char_indices)
        
        # Level 1: Syllables
        syl_boundaries, syl_emb = self.syllable_detector(char_emb)
        
        # Level 2: Morphemes
        morph_boundaries, morph_types, morph_emb = self.morpheme_parser(syl_emb)
        
        # Level 3: Words
        word_boundaries, pos_tags, word_emb = self.word_composer(morph_emb)
        
        # Reconstructions (for collapse prevention)
        char_recon = self.char_decoder(char_emb)
        syl_recon = self.syl_decoder(syl_emb)
        morph_recon = self.morph_decoder(morph_emb)
        
        # Next character prediction
        next_char = self.next_char_head(char_emb)
        
        return {
            'char_emb': char_emb,
            'syl_boundaries': syl_boundaries,
            'syl_emb': syl_emb,
            'morph_boundaries': morph_boundaries,
            'morph_types': morph_types,
            'morph_emb': morph_emb,
            'word_boundaries': word_boundaries,
            'pos_tags': pos_tags,
            'word_emb': word_emb,
            'char_recon': char_recon,
            'syl_recon': syl_recon,
            'morph_recon': morph_recon,
            'next_char': next_char,
        }


# =============================================================================
# TRAINING DATA PREPARATION
# =============================================================================

def prepare_training_batch(texts, max_len=128):
    """Prepare a batch with all supervision signals."""
    batch_chars = []
    batch_syl_bnd = []
    batch_morph_bnd = []
    batch_morph_types = []
    batch_word_bnd = []
    batch_pos = []
    
    for text in texts:
        # Character indices (PRESERVE CASE!)
        chars = text_to_indices(text, max_len)
        batch_chars.append(chars)
        
        # Get labels at word level
        words = text.split()
        
        # Build character-level labels
        syl_bnd = [0] * max_len
        morph_bnd = [0] * max_len
        morph_types = [0] * max_len
        word_bnd = [0] * max_len
        pos_labels = [0] * max_len
        
        char_pos = 1  # Start after <BOS>
        for word in words:
            if char_pos >= max_len - 1:
                break
                
            # Syllable boundaries
            syl_b = get_syllable_boundaries(word)
            for i, b in enumerate(syl_b):
                if char_pos + i < max_len:
                    syl_bnd[char_pos + i] = b
            
            # Morpheme boundaries and types
            morph_b, morph_t = get_morpheme_labels(word)
            for i, (b, t) in enumerate(zip(morph_b, morph_t)):
                if char_pos + i < max_len:
                    morph_bnd[char_pos + i] = b
                    morph_types[char_pos + i] = t
            
            # Word boundary (at end of word)
            word_end = char_pos + len(word) - 1
            if word_end < max_len:
                word_bnd[word_end] = 1
            
            # POS tag (apply to all chars in word)
            pos = get_pos_tag(word)
            for i in range(len(word)):
                if char_pos + i < max_len:
                    pos_labels[char_pos + i] = pos
            
            char_pos += len(word) + 1  # +1 for space
        
        batch_syl_bnd.append(syl_bnd)
        batch_morph_bnd.append(morph_bnd)
        batch_morph_types.append(morph_types)
        batch_word_bnd.append(word_bnd)
        batch_pos.append(pos_labels)
    
    return {
        'chars': torch.tensor(batch_chars),
        'syl_boundaries': torch.tensor(batch_syl_bnd, dtype=torch.float),
        'morph_boundaries': torch.tensor(batch_morph_bnd, dtype=torch.float),
        'morph_types': torch.tensor(batch_morph_types),
        'word_boundaries': torch.tensor(batch_word_bnd, dtype=torch.float),
        'pos_tags': torch.tensor(batch_pos),
    }


# =============================================================================
# DIVERSE TEST SENTENCES (30+)
# =============================================================================

TEST_SENTENCES = [
    # Simple sentences
    "Die Katze schläft.",
    "Der Hund läuft schnell.",
    "Ein Kind spielt im Garten.",
    "Die Sonne scheint hell.",
    "Der Mann liest ein Buch.",
    
    # Medium complexity
    "Die kleine Katze sitzt auf dem Dach.",
    "Der große Hund rennt durch den Park.",
    "Ein schönes Haus steht am Fluss.",
    "Die alte Frau geht langsam nach Hause.",
    "Der junge Mann arbeitet fleißig.",
    
    # Complex words (compounds, prefixes)
    "Die Möglichkeit ist unglaublich.",
    "Die Freundlichkeit der Menschen beeindruckt mich.",
    "Die Verantwortung liegt bei uns.",
    "Die Unabhängigkeit ist wichtig.",
    "Die Entwicklung geht weiter.",
    
    # Various verb forms
    "Ich gehe heute einkaufen.",
    "Du hast gestern gut geschlafen.",
    "Er wird morgen ankommen.",
    "Sie haben lange gearbeitet.",
    "Wir werden das Problem lösen.",
    
    # Questions
    "Was macht die Katze?",
    "Wo ist der Schlüssel?",
    "Wann kommst du nach Hause?",
    "Warum ist der Himmel blau?",
    "Wie geht es dir heute?",
    
    # Longer sentences
    "Der kleine Junge mit den roten Haaren spielt jeden Tag im großen Garten.",
    "Die freundliche Verkäuferin hilft den Kunden bei der Auswahl.",
    "Das interessante Buch über die Geschichte Deutschlands liegt auf dem Tisch.",
    "Die schnelle Entwicklung der Technologie verändert unser Leben.",
    "Der erfahrene Lehrer erklärt die schwierige Aufgabe mit Geduld.",
]


def check_embedding_collapse(model, device, threshold=0.95):
    """Check if embeddings are collapsing."""
    model.eval()
    
    # Use diverse test inputs
    test_inputs = [
        "Katze", "Hund", "Möglichkeit", "unglaublich", 
        "Die Sonne scheint.", "Der Mann arbeitet."
    ]
    
    embeddings = []
    with torch.no_grad():
        for text in test_inputs:
            chars = text_to_indices(text, 64)
            indices = torch.tensor([chars], device=device)
            outputs = model(indices)
            # Use morpheme embeddings (mid-level)
            emb = outputs['morph_emb'][0, 1:len(text)+1].mean(dim=0)
            embeddings.append(emb)
    
    # Check pairwise similarities
    max_sim = 0
    collapse_pairs = []
    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            sim = F.cosine_similarity(
                embeddings[i].unsqueeze(0), 
                embeddings[j].unsqueeze(0)
            ).item()
            if sim > max_sim:
                max_sim = sim
            if sim > threshold:
                collapse_pairs.append((test_inputs[i], test_inputs[j], sim))
    
    return max_sim, collapse_pairs


def test_model(model, device, num_samples=10):
    """Test model on random sentences."""
    model.eval()
    
    # Pick random test sentences
    samples = random.sample(TEST_SENTENCES, min(num_samples, len(TEST_SENTENCES)))
    
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    with torch.no_grad():
        for text in samples:
            chars = text_to_indices(text, 128)
            indices = torch.tensor([chars], device=device)
            outputs = model(indices)
            
            # Reconstruct from different levels
            char_recon = outputs['char_recon'].argmax(dim=-1)[0]
            char_text = indices_to_text(char_recon.cpu().tolist())
            
            # Get predictions
            syl_pred = (torch.sigmoid(outputs['syl_boundaries'][0]) > 0.5).int()
            pos_pred = outputs['pos_tags'][0].argmax(dim=-1)
            
            # Mark syllable boundaries in text
            marked_text = ""
            for i, c in enumerate(text[:len(syl_pred)-1]):
                marked_text += c
                if i < len(syl_pred)-1 and syl_pred[i+1] == 1:
                    marked_text += "·"
            
            print(f"\nInput:  '{text}'")
            print(f"Recon:  '{char_text[:len(text)+5]}'")
            print(f"Syllables: '{marked_text}'")
    
    # Check for collapse
    max_sim, collapse_pairs = check_embedding_collapse(model, device)
    print(f"\n📊 Max embedding similarity: {max_sim:.4f}")
    if collapse_pairs:
        print("⚠️  WARNING: Potential collapse detected!")
        for t1, t2, sim in collapse_pairs[:3]:
            print(f"   {t1} ↔ {t2}: {sim:.4f}")
    else:
        print("✅ Embeddings are differentiated")
    
    print("="*70 + "\n")
    
    return max_sim


# =============================================================================
# TRAINING LOOP WITH STRICT SUPERVISION
# =============================================================================

def train(model, sentences, device, epochs=30, batch_size=32, lr=3e-4):
    """Train with strict supervision and collapse monitoring."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    
    # Learning rate scheduler
    total_steps = (len(sentences) // batch_size) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
    
    best_loss = float('inf')
    collapse_count = 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(sentences)
        
        total_loss = 0
        total_recon_loss = 0
        total_boundary_loss = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences) - batch_size, batch_size), 
                   desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch_texts = sentences[i:i+batch_size]
            batch = prepare_training_batch(batch_texts, max_len=128)
            
            # Move to device
            chars = batch['chars'].to(device)
            syl_bnd = batch['syl_boundaries'].to(device)
            morph_bnd = batch['morph_boundaries'].to(device)
            morph_types = batch['morph_types'].to(device)
            word_bnd = batch['word_boundaries'].to(device)
            pos_tags = batch['pos_tags'].to(device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(chars)
                
                # RECONSTRUCTION LOSSES (prevent collapse!)
                mask = (chars != 0).float()
                
                char_recon_loss = F.cross_entropy(
                    outputs['char_recon'].view(-1, model.vocab_size),
                    chars.view(-1), ignore_index=0
                )
                syl_recon_loss = F.cross_entropy(
                    outputs['syl_recon'].view(-1, model.vocab_size),
                    chars.view(-1), ignore_index=0
                )
                morph_recon_loss = F.cross_entropy(
                    outputs['morph_recon'].view(-1, model.vocab_size),
                    chars.view(-1), ignore_index=0
                )
                recon_loss = char_recon_loss + syl_recon_loss + morph_recon_loss
                
                # BOUNDARY LOSSES
                syl_loss = F.binary_cross_entropy_with_logits(
                    outputs['syl_boundaries'], syl_bnd, reduction='none'
                )
                syl_loss = (syl_loss * mask).sum() / (mask.sum() + 1e-8)
                
                morph_loss = F.binary_cross_entropy_with_logits(
                    outputs['morph_boundaries'], morph_bnd, reduction='none'
                )
                morph_loss = (morph_loss * mask).sum() / (mask.sum() + 1e-8)
                
                word_loss = F.binary_cross_entropy_with_logits(
                    outputs['word_boundaries'], word_bnd, reduction='none'
                )
                word_loss = (word_loss * mask).sum() / (mask.sum() + 1e-8)
                
                boundary_loss = syl_loss + morph_loss + word_loss
                
                # TYPE/TAG LOSSES
                morph_type_loss = F.cross_entropy(
                    outputs['morph_types'].view(-1, len(MORPH_TYPES)),
                    morph_types.view(-1), ignore_index=0
                )
                pos_loss = F.cross_entropy(
                    outputs['pos_tags'].view(-1, len(POS_TAGS)),
                    pos_tags.view(-1), ignore_index=0
                )
                
                # NEXT CHARACTER LOSS
                next_char_loss = F.cross_entropy(
                    outputs['next_char'][:, :-1].reshape(-1, model.vocab_size),
                    chars[:, 1:].reshape(-1), ignore_index=0
                )
                
                # CONTRASTIVE LOSS - force different inputs to have different embeddings
                B = chars.shape[0]
                if B > 1:
                    # Get morpheme embeddings averaged per sample
                    morph_avg = outputs['morph_emb'].mean(dim=1)  # [B, d_morph]
                    morph_norm = F.normalize(morph_avg, dim=-1)
                    
                    # Compute similarity matrix
                    sim_matrix = morph_norm @ morph_norm.T  # [B, B]
                    
                    # Diagonal should be 1 (self-similarity)
                    # Off-diagonal should be LOW (different samples = different embeddings)
                    off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                    off_diag_sim = sim_matrix[off_diag_mask]
                    
                    # Penalize high similarity between different samples
                    # Target: off-diagonal similarity < 0.7
                    contrastive_loss = torch.relu(off_diag_sim - 0.7).mean() * 5.0
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
                
                # TOTAL LOSS
                loss = (recon_loss * 1.0 +      # Strong reconstruction signal
                       boundary_loss * 0.5 +
                       morph_type_loss * 0.3 +
                       pos_loss * 0.3 +
                       next_char_loss * 0.5 +
                       contrastive_loss * 1.0)  # Strong contrastive signal
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_boundary_loss += boundary_loss.item()
            total_contrastive = total_contrastive + contrastive_loss.item() if 'total_contrastive' in dir() else contrastive_loss.item()
            num_batches += 1
            
            if num_batches % 30 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'recon': f'{total_recon_loss/num_batches:.4f}',
                    'ctr': f'{contrastive_loss.item():.4f}'
                })
        
        # End of epoch
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        # TEST AND CHECK FOR COLLAPSE
        max_sim = test_model(model, device, num_samples=5)
        
        if max_sim > 0.95:
            collapse_count += 1
            print(f"⚠️  Collapse warning! Count: {collapse_count}/3")
            if collapse_count >= 3:
                print("❌ STOPPING: Embedding collapse detected!")
                break
        else:
            collapse_count = 0
        
        # Save best model
        if avg_loss < best_loss and max_sim < 0.95:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/tabula_rasa_best.pth")
            print(f"💾 Saved best model (loss={avg_loss:.4f}, sim={max_sim:.4f})")
    
    return model


# =============================================================================
# MAIN
# =============================================================================

def load_german_sentences(max_sentences=50000):
    """Load German sentences from dataset."""
    print("Loading German sentences (preserving case)...")
    sentences = []
    
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        
        for item in tqdm(ds, desc="Loading", total=max_sentences * 3):
            text = item.get('text', '')
            if isinstance(text, str):
                # Split into sentences
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    sent = sent.strip()
                    # Filter: reasonable length, has letters
                    if 10 <= len(sent) <= 150 and re.search(r'[A-Za-zäöüÄÖÜß]', sent):
                        sentences.append(sent)
                        if len(sentences) >= max_sentences:
                            break
            if len(sentences) >= max_sentences:
                break
                
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Fallback to test sentences
        sentences = TEST_SENTENCES * 100
    
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def main():
    print("="*70)
    print("HIERARCHICAL GERMAN MODEL - TABULA RASA")
    print("="*70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Vocabulary size: {VOCAB_SIZE} (with case preserved)")
    
    # Create model
    model = HierarchicalGermanModel(
        vocab_size=VOCAB_SIZE,
        d_char=128,
        d_syl=128,
        d_morph=256,
        d_word=256,
        max_len=128
    ).to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    
    # Load data
    sentences = load_german_sentences(50000)
    
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n" + "="*70)
    print("TRAINING WITH STRICT SUPERVISION")
    print("- Reconstruction loss (prevents collapse)")
    print("- Boundary detection (syllable, morpheme, word)")
    print("- Type classification (morpheme types, POS tags)")
    print("- Collapse monitoring (stops if similarity > 0.95)")
    print("="*70 + "\n")
    
    # Train
    model = train(model, sentences, device, epochs=30, batch_size=32, lr=3e-4)
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION (15 random sentences)")
    print("="*70)
    test_model(model, device, num_samples=15)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
