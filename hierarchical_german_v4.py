#!/usr/bin/env python3
"""
Hierarchical German Language Model - Version 4
Level 0: Character Encoder
Level 1: Syllable Detector  
Level 2: Morpheme Parser
Level 3: Word Composer
Level 4: Phrase Chunker (NEW)

Following human learning: chars → syllables → morphemes → words → phrases → sentences
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
# VOCABULARY & CONSTANTS
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

# Phrase types for German
PHRASE_TYPES = {
    'PAD': 0,
    'NP': 1,   # Noun Phrase: "der kleine Hund"
    'VP': 2,   # Verb Phrase: "läuft schnell"
    'PP': 3,   # Prepositional Phrase: "in dem Haus"
    'AP': 4,   # Adjective Phrase: "sehr schön"
    'ADVP': 5, # Adverb Phrase: "sehr schnell"
}

def text_to_indices(text, max_len=128):
    indices = [char_to_idx.get(c, char_to_idx[UNK_TOKEN]) for c in text[:max_len]]
    while len(indices) < max_len:
        indices.append(char_to_idx[PAD_TOKEN])
    return indices


# =============================================================================
# LEVEL 0-3: Previous layers (simplified for inheritance)
# =============================================================================

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, d_char=128, max_len=256):
        super().__init__()
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
        B, L = x.shape
        char_emb = self.char_embed(x)
        pos_emb = self.pos_embed(torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1))
        x = char_emb + pos_emb
        x = self.norm(x + self.local_context(x.transpose(1,2)).transpose(1,2))
        return self.dropout(x)


class SyllableDetector(nn.Module):
    def __init__(self, d_char=128):
        super().__init__()
        self.lstm = nn.LSTM(d_char, d_char//2, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(d_char, d_char), nn.GELU(), nn.Linear(d_char, 1))
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1)


class MorphemeParser(nn.Module):
    def __init__(self, d_input=128, d_morpheme=256):
        super().__init__()
        self.lstm = nn.LSTM(d_input, d_input, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary = nn.Sequential(nn.Linear(d_input*2, d_input), nn.GELU(), nn.Linear(d_input, 1))
        self.types = nn.Sequential(nn.Linear(d_input*2, d_input), nn.GELU(), nn.Linear(d_input, 5))
        self.project = nn.Sequential(nn.Linear(d_input*2, d_morpheme), nn.LayerNorm(d_morpheme), nn.GELU())
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.boundary(out).squeeze(-1), self.types(out), self.project(out)


class WordComposer(nn.Module):
    def __init__(self, d_morpheme=256, d_word=512):
        super().__init__()
        self.type_embed = nn.Embedding(5, d_morpheme)
        self.attn = nn.MultiheadAttention(d_morpheme, 8, dropout=0.1, batch_first=True)
        self.composer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_morpheme, 8, d_morpheme*4, 0.1, batch_first=True), 2
        )
        self.project = nn.Sequential(nn.Linear(d_morpheme, d_word), nn.LayerNorm(d_word), nn.GELU())
        self.boundary = nn.Sequential(nn.Linear(d_morpheme, d_morpheme//2), nn.GELU(), nn.Linear(d_morpheme//2, 1))
        
    def forward(self, morph_emb, morph_types=None):
        if morph_types is not None:
            morph_emb = morph_emb + self.type_embed(morph_types)
        attended, _ = self.attn(morph_emb, morph_emb, morph_emb)
        composed = self.composer(attended)
        return self.project(composed), self.boundary(composed).squeeze(-1)


# =============================================================================
# LEVEL 4: PHRASE CHUNKER (NEW)
# =============================================================================

class PhraseChunker(nn.Module):
    """
    Level 4: Group words into syntactic phrases.
    
    German phrase structure:
    - NP: [Det] [Adj]* Noun [PP]*  e.g., "der kleine rote Ball"
    - VP: [Aux] [Adv]* Verb [NP] [PP]*  e.g., "hat schnell gegessen"
    - PP: Prep NP  e.g., "in dem Haus"
    - AP: [Adv] Adj  e.g., "sehr schön"
    
    Uses attention over word embeddings to identify phrase boundaries and types.
    """
    def __init__(self, d_word=512, d_phrase=512, num_phrase_types=6, num_heads=8):
        super().__init__()
        self.d_phrase = d_phrase
        self.num_phrase_types = num_phrase_types
        
        # Phrase boundary detection
        self.phrase_lstm = nn.LSTM(
            d_word, d_word // 2, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.1
        )
        
        # Boundary prediction (phrase starts after this position)
        self.phrase_boundary_head = nn.Sequential(
            nn.Linear(d_word, d_word // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_word // 2, 1)
        )
        
        # Phrase type classification
        self.phrase_type_head = nn.Sequential(
            nn.Linear(d_word, d_word // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_word // 2, num_phrase_types)
        )
        
        # Phrase composition via transformer
        self.phrase_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_word, nhead=num_heads,
                dim_feedforward=d_word * 4, dropout=0.1,
                batch_first=True
            ),
            num_layers=3
        )
        
        # Project to phrase embedding space
        self.phrase_project = nn.Sequential(
            nn.Linear(d_word, d_phrase),
            nn.LayerNorm(d_phrase),
            nn.GELU()
        )
        
    def forward(self, word_embeddings, word_mask=None):
        """
        word_embeddings: [batch, seq_len, d_word]
        
        Returns:
            phrase_boundary_logits: [batch, seq_len]
            phrase_type_logits: [batch, seq_len, num_types]
            phrase_embeddings: [batch, seq_len, d_phrase]
        """
        # Detect phrase boundaries
        lstm_out, _ = self.phrase_lstm(word_embeddings)
        
        phrase_boundary_logits = self.phrase_boundary_head(lstm_out).squeeze(-1)
        phrase_type_logits = self.phrase_type_head(lstm_out)
        
        # Compose phrases
        phrase_composed = self.phrase_transformer(word_embeddings)
        phrase_embeddings = self.phrase_project(phrase_composed)
        
        return phrase_boundary_logits, phrase_type_logits, phrase_embeddings


# =============================================================================
# LEVEL 5: SENTENCE ENCODER (Preview)
# =============================================================================

class SentenceEncoder(nn.Module):
    """
    Level 5: Compose phrases into sentence-level meaning.
    """
    def __init__(self, d_phrase=512, d_sentence=768, num_heads=8):
        super().__init__()
        self.sentence_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_phrase, nhead=num_heads,
                dim_feedforward=d_phrase * 4, dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )
        self.sentence_project = nn.Sequential(
            nn.Linear(d_phrase, d_sentence),
            nn.LayerNorm(d_sentence),
            nn.GELU()
        )
        # Pool to single sentence vector
        self.sentence_pool = nn.Sequential(
            nn.Linear(d_sentence, d_sentence),
            nn.Tanh()
        )
        
    def forward(self, phrase_embeddings, mask=None):
        composed = self.sentence_transformer(phrase_embeddings)
        sentence_seq = self.sentence_project(composed)
        # Mean pool for sentence vector
        sentence_vector = sentence_seq.mean(dim=1)
        sentence_vector = self.sentence_pool(sentence_vector)
        return sentence_seq, sentence_vector


# =============================================================================
# FULL HIERARCHICAL MODEL (Level 0-5)
# =============================================================================

class HierarchicalGermanV4(nn.Module):
    """
    Hierarchical German Model - Version 4
    Complete hierarchy: Characters → Syllables → Morphemes → Words → Phrases → Sentences
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_morpheme=256, 
                 d_word=512, d_phrase=512, d_sentence=768, max_len=128):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_sentence = d_sentence
        
        # Level 0: Character encoder
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        
        # Level 1: Syllable detector
        self.syllable_detector = SyllableDetector(d_char)
        
        # Level 2: Morpheme parser
        self.morpheme_parser = MorphemeParser(d_char, d_morpheme)
        
        # Level 3: Word composer
        self.word_composer = WordComposer(d_morpheme, d_word)
        
        # Level 4: Phrase chunker
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        
        # Level 5: Sentence encoder
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sentence)
        
        # Character prediction (auxiliary)
        self.char_predictor = nn.Sequential(
            nn.Linear(d_char, d_char * 2), nn.GELU(), nn.Linear(d_char * 2, vocab_size)
        )
        
        # Sentence-level next sentence prediction
        self.sentence_predictor = nn.Sequential(
            nn.Linear(d_sentence, d_sentence),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_sentence, d_sentence)
        )
        
    def forward(self, char_indices):
        # Level 0: Characters
        char_emb = self.char_encoder(char_indices)
        
        # Level 1: Syllables
        syl_logits = self.syllable_detector(char_emb)
        
        # Level 2: Morphemes
        morph_bnd, morph_types, morph_emb = self.morpheme_parser(char_emb)
        
        # Level 3: Words
        word_emb, word_bnd = self.word_composer(morph_emb, morph_types.argmax(-1))
        
        # Level 4: Phrases
        phrase_bnd, phrase_types, phrase_emb = self.phrase_chunker(word_emb)
        
        # Level 5: Sentence
        sentence_seq, sentence_vec = self.sentence_encoder(phrase_emb)
        
        # Predictions
        char_pred = self.char_predictor(char_emb)
        sent_pred = self.sentence_predictor(sentence_vec)
        
        return {
            'char_embeddings': char_emb,
            'syllable_logits': syl_logits,
            'morph_boundary_logits': morph_bnd,
            'morph_type_logits': morph_types,
            'morph_embeddings': morph_emb,
            'word_embeddings': word_emb,
            'word_boundary_logits': word_bnd,
            'phrase_boundary_logits': phrase_bnd,
            'phrase_type_logits': phrase_types,
            'phrase_embeddings': phrase_emb,
            'sentence_sequence': sentence_seq,
            'sentence_vector': sentence_vec,
            'char_predictions': char_pred,
            'sentence_predictions': sent_pred
        }


# =============================================================================
# DATA & TRAINING
# =============================================================================

def prepare_phrase_data(sentence, max_len=64):
    """Prepare sentence with phrase boundary labels (simplified heuristic)."""
    chars = text_to_indices(sentence.lower(), max_len)
    
    # Simple phrase boundary heuristic:
    # Phrases typically end before prepositions and after punctuation
    prepositions = {'in', 'auf', 'an', 'bei', 'mit', 'nach', 'von', 'zu', 'für', 'über', 'unter'}
    words = sentence.lower().split()
    
    phrase_boundaries = []
    char_pos = 0
    
    for word in words:
        # Mark phrase boundary before prepositions
        is_prep = word.strip('.,!?') in prepositions
        for i, c in enumerate(word):
            if char_pos >= max_len:
                break
            # Boundary at end of word before preposition, or after punctuation
            if i == len(word) - 1 and is_prep:
                phrase_boundaries.append(1)
            elif c in '.,!?':
                phrase_boundaries.append(1)
            else:
                phrase_boundaries.append(0)
            char_pos += 1
        # Add space
        if char_pos < max_len:
            phrase_boundaries.append(0)
            char_pos += 1
    
    while len(phrase_boundaries) < max_len:
        phrase_boundaries.append(0)
    
    return chars, phrase_boundaries[:max_len]


def load_german_sentences(max_sentences=30000):
    print("Loading German sentences...")
    sentences = []
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_sentences * 5):
            text = item.get('text', item.get('story', ''))
            if isinstance(text, str):
                for sent in re.split(r'[.!?]+', text):
                    sent = sent.strip()
                    if 15 <= len(sent) <= 100:
                        sentences.append(sent)
                        if len(sentences) >= max_sentences:
                            break
            if len(sentences) >= max_sentences:
                break
    except Exception as e:
        print(f"Error: {e}")
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def train_hierarchical_v4(model, sentences, device, epochs=20, batch_size=32, lr=2e-4):
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
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences) - batch_size, batch_size), 
                   desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch = sentences[i:i+batch_size]
            
            char_batch, phrase_bnd_batch = [], []
            for sent in batch:
                chars, pbnd = prepare_phrase_data(sent, max_len=64)
                char_batch.append(chars)
                phrase_bnd_batch.append(pbnd)
            
            char_indices = torch.tensor(char_batch, device=device)
            phrase_boundaries = torch.tensor(phrase_bnd_batch, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(char_indices)
                
                # Phrase boundary loss
                phrase_loss = F.binary_cross_entropy_with_logits(
                    outputs['phrase_boundary_logits'], phrase_boundaries
                )
                
                # Sentence embedding contrastive loss
                sent_emb = outputs['sentence_vector']
                sent_pred = outputs['sentence_predictions']
                sent_loss = 1 - F.cosine_similarity(sent_emb, sent_pred, dim=-1).mean()
                
                # Char prediction loss
                char_targets = char_indices[:, 1:]
                char_preds = outputs['char_predictions'][:, :-1, :]
                char_loss = F.cross_entropy(
                    char_preds.reshape(-1, model.vocab_size),
                    char_targets.reshape(-1), ignore_index=0
                )
                
                loss = phrase_loss + sent_loss + 0.3 * char_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 20 == 0:
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        test_phrase_chunking(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hierarchical_v4_best.pth")
            print(f"💾 Saved best model")


def test_phrase_chunking(model, device):
    model.eval()
    test_sentences = [
        "Die kleine Katze sitzt auf dem Dach",
        "Der Junge spielt mit dem Ball im Garten",
        "Ich gehe heute nach Hause",
    ]
    
    print("\n📊 Phrase Chunking Test:")
    with torch.no_grad():
        for sent in test_sentences:
            chars, _ = prepare_phrase_data(sent, max_len=64)
            indices = torch.tensor([chars], device=device)
            outputs = model(indices)
            
            bnd = torch.sigmoid(outputs['phrase_boundary_logits'][0]).cpu().numpy()
            types = outputs['phrase_type_logits'][0].argmax(-1).cpu().numpy()
            
            type_names = ['_', 'NP', 'VP', 'PP', 'AP', 'ADV']
            result = ""
            for i, c in enumerate(sent.lower()[:len(bnd)]):
                result += c
                if bnd[i] > 0.5:
                    result += f"[{type_names[types[i]]}]|"
            
            print(f"  '{sent}' → '{result}'")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Vocab: {VOCAB_SIZE}")
    
    model = HierarchicalGermanV4(
        vocab_size=VOCAB_SIZE, d_char=128, d_morpheme=256,
        d_word=512, d_phrase=512, d_sentence=768, max_len=64
    ).to(device)
    
    # Load previous weights
    v3_path = "checkpoints/hierarchical_v3_best.pth"
    if os.path.exists(v3_path):
        print(f"Loading weights from {v3_path}")
        state = torch.load(v3_path, map_location=device)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                loaded += 1
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {loaded} layers")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    
    sentences = load_german_sentences(30000)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING HIERARCHICAL MODEL (Level 0-5: Full Hierarchy)")
    print("="*60 + "\n")
    
    train_hierarchical_v4(model, sentences, device, epochs=20, batch_size=32, lr=2e-4)
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
