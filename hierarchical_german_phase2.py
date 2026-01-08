#!/usr/bin/env python3
"""
Hierarchical German Language Model - Phase 2
Extends tabula rasa with Level 4 (Phrase) and Level 5 (Sentence).
Uses attention pooling (not mean pooling) and contrastive loss.
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

# =============================================================================
# VOCABULARY
# =============================================================================

CHARS = (
    'abcdefghijklmnopqrstuvwxyz'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    'äöüßÄÖÜ'
    '0123456789'
    ' .,!?;:\'"()-\n'
)

SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
char_to_idx = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

POS_TAGS = {'PAD': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'ADV': 4,
            'DET': 5, 'PREP': 6, 'CONJ': 7, 'PRON': 8, 'AUX': 9, 'PUNCT': 10}
MORPH_TYPES = {'PAD': 0, 'PREFIX': 1, 'ROOT': 2, 'SUFFIX': 3, 'INFLECT': 4, 'COMPOUND': 5}
PHRASE_TYPES = {'PAD': 0, 'NP': 1, 'VP': 2, 'PP': 3, 'ADJP': 4, 'ADVP': 5, 'SBAR': 6}

GERMAN_PREFIXES = {'un', 'ver', 'be', 'ge', 'ent', 'er', 'zer', 'ab', 'an', 'auf', 'aus', 'ein', 'mit', 'nach', 'vor', 'zu'}
GERMAN_SUFFIXES = {'ung', 'keit', 'heit', 'lich', 'isch', 'bar', 'sam', 'los', 'ig', 'en', 'er', 'schaft'}
GERMAN_POS = {
    'der': 'DET', 'die': 'DET', 'das': 'DET', 'ein': 'DET', 'eine': 'DET',
    'ich': 'PRON', 'du': 'PRON', 'er': 'PRON', 'sie': 'PRON', 'es': 'PRON', 'wir': 'PRON',
    'in': 'PREP', 'auf': 'PREP', 'mit': 'PREP', 'nach': 'PREP', 'von': 'PREP', 'zu': 'PREP',
    'und': 'CONJ', 'oder': 'CONJ', 'aber': 'CONJ',
    'ist': 'AUX', 'sind': 'AUX', 'hat': 'AUX', 'haben': 'AUX', 'wird': 'AUX',
    'hier': 'ADV', 'dort': 'ADV', 'heute': 'ADV', 'sehr': 'ADV', 'nicht': 'ADV',
}

def text_to_indices(text, max_len=128):
    indices = [char_to_idx.get('<BOS>')]
    for c in text[:max_len-2]:
        indices.append(char_to_idx.get(c, char_to_idx['<UNK>']))
    indices.append(char_to_idx.get('<EOS>'))
    while len(indices) < max_len:
        indices.append(char_to_idx['<PAD>'])
    return indices[:max_len]

def indices_to_text(indices):
    chars = []
    for idx in indices:
        if idx == char_to_idx['<PAD>'] or idx == char_to_idx['<EOS>']:
            break
        if idx == char_to_idx['<BOS>']:
            continue
        chars.append(idx_to_char.get(idx, '?'))
    return ''.join(chars)

def get_syllable_boundaries(word):
    vowels = set('aeiouäöüAEIOUÄÖÜ')
    boundaries = [0] * len(word)
    i = 0
    while i < len(word):
        if word[i] in vowels:
            j = i + 1
            while j < len(word) and word[j] not in vowels:
                j += 1
            if j < len(word) and j - i > 1:
                split_point = j - 1
                if split_point > i:
                    boundaries[split_point] = 1
        i += 1
    return boundaries

def get_morpheme_labels(word):
    word_lower = word.lower()
    boundaries = [0] * len(word)
    types = [MORPH_TYPES['ROOT']] * len(word)
    for prefix in GERMAN_PREFIXES:
        if word_lower.startswith(prefix) and len(word_lower) > len(prefix) + 2:
            boundaries[len(prefix)] = 1
            for i in range(len(prefix)):
                types[i] = MORPH_TYPES['PREFIX']
            break
    for suffix in GERMAN_SUFFIXES:
        if word_lower.endswith(suffix) and len(word_lower) > len(suffix) + 2:
            suffix_start = len(word) - len(suffix)
            boundaries[suffix_start] = 1
            for i in range(suffix_start, len(word)):
                types[i] = MORPH_TYPES['SUFFIX']
            break
    return boundaries, types

def get_pos_tag(word):
    word_lower = word.lower().strip('.,!?;:')
    if word_lower in GERMAN_POS:
        return POS_TAGS[GERMAN_POS[word_lower]]
    if word[0].isupper() and word_lower not in ['der', 'die', 'das', 'ein', 'eine']:
        return POS_TAGS['NOUN']
    if word_lower.endswith(('en', 'st', 'te', 't')):
        return POS_TAGS['VERB']
    return POS_TAGS['NOUN']

# =============================================================================
# LEVEL 0-3: From Phase 1
# =============================================================================

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, d_char=128, max_len=128):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_char)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(d_char, d_char, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_char, d_char, 3, padding=1))
        self.norm = nn.LayerNorm(d_char)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        x = self.char_embed(x) + self.pos_embed(positions)
        cnn_out = self.local_cnn(x.transpose(1, 2)).transpose(1, 2)
        return self.dropout(self.norm(x + cnn_out))

class SyllableDetector(nn.Module):
    def __init__(self, d_char=128, d_syl=128):
        super().__init__()
        self.lstm = nn.LSTM(d_char, d_char // 2, num_layers=2, 
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_char, d_char // 2), nn.GELU(), nn.Linear(d_char // 2, 1))
        self.project = nn.Linear(d_char, d_syl)
        self.norm = nn.LayerNorm(d_syl)
        
    def forward(self, char_emb):
        lstm_out, _ = self.lstm(char_emb)
        boundaries = self.boundary_head(lstm_out).squeeze(-1)
        return boundaries, self.norm(self.project(lstm_out))

class MorphemeParser(nn.Module):
    def __init__(self, d_syl=128, d_morph=256, num_types=6):
        super().__init__()
        self.lstm = nn.LSTM(d_syl, d_syl, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_syl * 2, d_syl), nn.GELU(), nn.Linear(d_syl, 1))
        self.type_head = nn.Sequential(
            nn.Linear(d_syl * 2, d_syl), nn.GELU(), nn.Linear(d_syl, num_types))
        self.project = nn.Sequential(
            nn.Linear(d_syl * 2, d_morph), nn.LayerNorm(d_morph), nn.GELU())
        
    def forward(self, syl_emb):
        lstm_out, _ = self.lstm(syl_emb)
        return (self.boundary_head(lstm_out).squeeze(-1),
                self.type_head(lstm_out), self.project(lstm_out))

class WordComposer(nn.Module):
    def __init__(self, d_morph=256, d_word=256, num_pos=11):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_morph, 4, dropout=0.1, batch_first=True)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_morph, d_morph // 2), nn.GELU(), nn.Linear(d_morph // 2, 1))
        self.pos_head = nn.Sequential(
            nn.Linear(d_morph, d_morph // 2), nn.GELU(), nn.Linear(d_morph // 2, num_pos))
        self.project = nn.Sequential(
            nn.Linear(d_morph, d_word), nn.LayerNorm(d_word), nn.GELU())
        
    def forward(self, morph_emb):
        attended, _ = self.attention(morph_emb, morph_emb, morph_emb)
        return (self.boundary_head(attended).squeeze(-1),
                self.pos_head(attended), self.project(attended))

# =============================================================================
# LEVEL 4-5: NEW IN PHASE 2
# =============================================================================

class PhraseChunker(nn.Module):
    """Level 4: Chunk into phrases with attention pooling."""
    def __init__(self, d_word=256, d_phrase=512, num_types=7):
        super().__init__()
        self.lstm = nn.LSTM(d_word, d_word, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary_head = nn.Sequential(
            nn.Linear(d_word * 2, d_word), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_word, 1))
        self.type_head = nn.Sequential(
            nn.Linear(d_word * 2, d_word), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_word, num_types))
        self.project = nn.Sequential(
            nn.Linear(d_word * 2, d_phrase), nn.LayerNorm(d_phrase), nn.GELU())
        
    def forward(self, word_emb):
        lstm_out, _ = self.lstm(word_emb)
        return (self.boundary_head(lstm_out).squeeze(-1),
                self.type_head(lstm_out), self.project(lstm_out))

class SentenceEncoder(nn.Module):
    """Level 5: Sentence encoding with CLS token (not mean pooling!)."""
    def __init__(self, d_phrase=512, d_sent=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_phrase, nhead=nhead, dim_feedforward=d_phrase * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_phrase) * 0.02)
        self.project = nn.Sequential(nn.Linear(d_phrase, d_sent), nn.LayerNorm(d_sent), nn.GELU())
        
    def forward(self, phrase_emb):
        B = phrase_emb.size(0)
        x = torch.cat([self.cls_token.expand(B, -1, -1), phrase_emb], dim=1)
        x = self.transformer(x)
        return self.project(x[:, 0])  # CLS token

class Decoder(nn.Module):
    def __init__(self, d_emb, vocab_size, d_hidden=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_emb, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, vocab_size))
    def forward(self, emb):
        return self.decoder(emb)

# =============================================================================
# COMPLETE MODEL
# =============================================================================

class HierarchicalGermanPhase2(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syl=128, 
                 d_morph=256, d_word=256, d_phrase=512, d_sent=512, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Level 0-3
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syl)
        self.morpheme_parser = MorphemeParser(d_syl, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        
        # Level 4-5 (NEW)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sent)
        
        # Decoders
        self.char_decoder = Decoder(d_char, vocab_size)
        self.syl_decoder = Decoder(d_syl, vocab_size)
        self.morph_decoder = Decoder(d_morph, vocab_size)
        self.phrase_decoder = Decoder(d_phrase, vocab_size)
        self.sent_decoder = Decoder(d_sent, vocab_size)
        self.next_char_head = nn.Linear(d_char, vocab_size)
        
    def forward(self, char_indices):
        char_emb = self.char_encoder(char_indices)
        syl_boundaries, syl_emb = self.syllable_detector(char_emb)
        morph_boundaries, morph_types, morph_emb = self.morpheme_parser(syl_emb)
        word_boundaries, pos_tags, word_emb = self.word_composer(morph_emb)
        phrase_boundaries, phrase_types, phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        
        return {
            'char_emb': char_emb, 'syl_boundaries': syl_boundaries, 'syl_emb': syl_emb,
            'morph_boundaries': morph_boundaries, 'morph_types': morph_types, 'morph_emb': morph_emb,
            'word_boundaries': word_boundaries, 'pos_tags': pos_tags, 'word_emb': word_emb,
            'phrase_boundaries': phrase_boundaries, 'phrase_types': phrase_types, 'phrase_emb': phrase_emb,
            'sent_emb': sent_emb,
            'char_recon': self.char_decoder(char_emb),
            'syl_recon': self.syl_decoder(syl_emb),
            'morph_recon': self.morph_decoder(morph_emb),
            'phrase_recon': self.phrase_decoder(phrase_emb),
            'sent_recon': self.sent_decoder(sent_emb.unsqueeze(1).expand(-1, char_indices.size(1), -1)),
            'next_char': self.next_char_head(char_emb),
        }
    
    def load_phase1_weights(self, checkpoint_path):
        print(f"Loading Phase 1 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        print(f"  Loading {len(pretrained)}/{len(state_dict)} parameters")
        model_dict.update(pretrained)
        self.load_state_dict(model_dict)
        
        # Freeze Phase 1 layers initially
        for name, param in self.named_parameters():
            if any(x in name for x in ['char_encoder', 'syllable_detector', 
                   'morpheme_parser', 'word_composer', 'char_decoder', 'syl_decoder', 'morph_decoder']):
                param.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"  Frozen: {frozen}, Trainable: {trainable}")

# =============================================================================
# TRAINING
# =============================================================================

def prepare_batch(texts, max_len=128):
    batch_chars, batch_syl, batch_morph, batch_word, batch_phrase = [], [], [], [], []
    for text in texts:
        chars = text_to_indices(text, max_len)
        batch_chars.append(chars)
        syl_bnd = [0] * max_len
        word_bnd = [0] * max_len
        phrase_bnd = [0] * max_len
        pos = 1
        words = text.split()
        for i, word in enumerate(words):
            if pos >= max_len - 1:
                break
            for j, b in enumerate(get_syllable_boundaries(word)):
                if pos + j < max_len:
                    syl_bnd[pos + j] = b
            word_end = min(pos + len(word), max_len - 1)
            word_bnd[word_end] = 1
            if i > 0 and i % 3 == 0:  # Heuristic phrase boundaries
                phrase_bnd[pos] = 1
            pos += len(word) + 1
        batch_syl.append(syl_bnd)
        batch_word.append(word_bnd)
        batch_phrase.append(phrase_bnd)
    return (torch.tensor(batch_chars), torch.tensor(batch_syl, dtype=torch.float),
            torch.tensor(batch_word, dtype=torch.float), torch.tensor(batch_phrase, dtype=torch.float))

def check_collapse(model, device, sentences):
    model.eval()
    embeddings = []
    with torch.no_grad():
        for sent in sentences[:10]:
            chars = torch.tensor([text_to_indices(sent, 128)], device=device)
            outputs = model(chars)
            embeddings.append(outputs['sent_emb'][0])
    
    max_sim = 0.0
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
            max_sim = max(max_sim, sim)
    return max_sim

def test_model(model, device, sentences):
    model.eval()
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    with torch.no_grad():
        for sent in random.sample(sentences, min(5, len(sentences))):
            chars = torch.tensor([text_to_indices(sent, 128)], device=device)
            outputs = model(chars)
            recon = indices_to_text(outputs['char_recon'].argmax(dim=-1)[0].cpu().tolist())
            print(f"Input:  '{sent[:60]}'")
            print(f"Recon:  '{recon[:60]}'")
            print()
    max_sim = check_collapse(model, device, sentences)
    print(f"📊 Max sentence embedding similarity: {max_sim:.4f}")
    if max_sim < 0.95:
        print("✅ Sentence embeddings are differentiated")
    else:
        print("⚠️  WARNING: Potential collapse!")
    print("="*60)
    model.train()  # Switch back to training mode!
    return max_sim

def load_german_sentences(max_sentences=50000):
    print("Loading German sentences...")
    sentences = []
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_sentences):
            text = item.get('text', '')
            if isinstance(text, str):
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    sent = sent.strip()
                    if 10 <= len(sent) <= 120 and re.search(r'[A-Za-zäöüÄÖÜß]', sent):
                        sentences.append(sent)
                        if len(sentences) >= max_sentences:
                            break
            if len(sentences) >= max_sentences:
                break
    except Exception as e:
        print(f"Error: {e}")
    print(f"Loaded {len(sentences)} sentences")
    return sentences

TEST_SENTENCES = [
    "Der große Hund läuft schnell durch den Park.",
    "Die kleine Katze sitzt auf dem Dach.",
    "Ein schönes Haus steht am Fluss.",
    "Die Sonne scheint hell am Himmel.",
    "Der Mann liest ein Buch.",
    "Wir gehen heute einkaufen.",
    "Die Kinder spielen im Garten.",
    "Sie möchte nach Berlin fahren.",
    "Das Wetter ist heute schön.",
    "Die Freundlichkeit der Menschen beeindruckt mich.",
]

def train(model, sentences, device, epochs=20, batch_size=32, lr=3e-4):
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(sentences) // batch_size)
    scaler = GradScaler()
    
    collapse_count = 0
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        random.shuffle(sentences)
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences), batch_size), desc=f"Epoch {epoch}/{epochs}")
        for i in pbar:
            batch = sentences[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            chars, syl_bnd, word_bnd, phrase_bnd = prepare_batch(batch)
            chars = chars.to(device)
            syl_bnd, word_bnd, phrase_bnd = syl_bnd.to(device), word_bnd.to(device), phrase_bnd.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(chars)
                
                # Reconstruction losses
                char_recon_loss = F.cross_entropy(outputs['char_recon'].view(-1, model.vocab_size), chars.view(-1), ignore_index=0)
                phrase_recon_loss = F.cross_entropy(outputs['phrase_recon'].view(-1, model.vocab_size), chars.view(-1), ignore_index=0)
                sent_recon_loss = F.cross_entropy(outputs['sent_recon'].view(-1, model.vocab_size), chars.view(-1), ignore_index=0)
                recon_loss = char_recon_loss + phrase_recon_loss + sent_recon_loss
                
                # Boundary losses
                phrase_bnd_loss = F.binary_cross_entropy_with_logits(outputs['phrase_boundaries'], phrase_bnd)
                
                # Contrastive loss at sentence level
                B = chars.shape[0]
                sent_norm = F.normalize(outputs['sent_emb'], dim=-1)
                sim_matrix = sent_norm @ sent_norm.T
                off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                off_diag_sim = sim_matrix[off_diag_mask]
                contrastive_loss = torch.relu(off_diag_sim - 0.5).mean() * 10.0  # Strong!
                
                loss = recon_loss + phrase_bnd_loss * 0.5 + contrastive_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 30 == 0:
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}', 'ctr': f'{contrastive_loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        max_sim = test_model(model, device, TEST_SENTENCES)
        
        if max_sim > 0.95:
            collapse_count += 1
            print(f"⚠️  Collapse warning! Count: {collapse_count}/3")
            if collapse_count >= 3:
                print("❌ STOPPING: Embedding collapse detected!")
                break
        else:
            collapse_count = 0
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "checkpoints/phase2_best.pth")
                print(f"💾 Saved best model (loss={avg_loss:.4f}, sim={max_sim:.4f})")
    
    print("\n✅ Training complete!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model = HierarchicalGermanPhase2().to(device)
    model.load_phase1_weights("checkpoints/tabula_rasa_best.pth")
    
    sentences = load_german_sentences(50000)
    if not sentences:
        sentences = TEST_SENTENCES * 100
    
    train(model, sentences, device, epochs=20, batch_size=32)

if __name__ == "__main__":
    main()
