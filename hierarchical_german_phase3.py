#!/usr/bin/env python3
"""
Hierarchical German Language Model - Phase 3: Brain Integration

Adds brain-inspired modules:
1. Dopamine - reward signal for correct predictions
2. Hebbian Plasticity - strengthen level associations  
3. Hippocampus - vocabulary memory storage/retrieval
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
# VOCABULARY (same as Phase 2)
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

# =============================================================================
# BRAIN MODULES
# =============================================================================

class DopamineSystem(nn.Module):
    """Reward signal based on prediction accuracy."""
    def __init__(self, baseline_tau=0.99):
        super().__init__()
        self.baseline_tau = baseline_tau
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        self.register_buffer('reward_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def compute_reward(self, predictions, targets, mask=None):
        """Compute accuracy-based reward."""
        pred_ids = predictions.argmax(dim=-1)
        correct = (pred_ids == targets).float()
        if mask is not None:
            correct = correct * mask
            accuracy = correct.sum() / (mask.sum() + 1e-8)
        else:
            accuracy = correct.mean()
        return accuracy
    
    def forward(self, reward):
        """Return reward prediction error (surprise signal)."""
        with torch.no_grad():
            rpe = reward - self.reward_baseline
            self.reward_baseline = self.baseline_tau * self.reward_baseline + (1 - self.baseline_tau) * reward
            idx = self.history_idx.item() % 100
            self.reward_history[idx] = reward
            self.history_idx += 1
        return rpe
    
    def get_learning_rate_multiplier(self, rpe):
        """Higher RPE = higher learning rate (more surprise = learn more)."""
        return 1.0 + torch.clamp(rpe, -0.5, 0.5)


class HebbianPlasticity(nn.Module):
    """Strengthen associations between hierarchical levels."""
    def __init__(self, d_pre, d_post, tau=0.995):
        super().__init__()
        self.tau = tau
        self.register_buffer('association_matrix', torch.zeros(d_pre, d_post))
        self.register_buffer('pre_mean', torch.zeros(d_pre))
        self.register_buffer('post_mean', torch.zeros(d_post))
        
    def update(self, pre_activity, post_activity):
        """Hebbian update: cells that fire together wire together."""
        with torch.no_grad():
            pre = pre_activity.mean(dim=0) if pre_activity.dim() > 1 else pre_activity
            post = post_activity.mean(dim=0) if post_activity.dim() > 1 else post_activity
            
            if pre.dim() > 1:
                pre = pre.mean(dim=0)
            if post.dim() > 1:
                post = post.mean(dim=0)
            
            self.pre_mean = self.tau * self.pre_mean + (1 - self.tau) * pre
            self.post_mean = self.tau * self.post_mean + (1 - self.tau) * post
            
            pre_centered = pre - self.pre_mean
            post_centered = post - self.post_mean
            
            hebbian_update = torch.outer(pre_centered, post_centered)
            self.association_matrix = self.tau * self.association_matrix + (1 - self.tau) * hebbian_update
            
    def get_association_strength(self):
        """Return overall association strength."""
        return self.association_matrix.abs().mean()


class Hippocampus(nn.Module):
    """Vocabulary memory with fast encoding and retrieval."""
    def __init__(self, d_emb, capacity=10000):
        super().__init__()
        self.d_emb = d_emb
        self.capacity = capacity
        self.register_buffer('memory', torch.zeros(capacity, d_emb))
        self.register_buffer('memory_keys', torch.zeros(capacity, d_emb))
        self.register_buffer('count', torch.tensor(0))
        self.register_buffer('importance', torch.zeros(capacity))
        
    def encode(self, embedding, importance=1.0):
        """Store embedding in memory."""
        with torch.no_grad():
            if embedding.dim() > 1:
                embedding = embedding.mean(dim=0)
            
            idx = self.count.item() % self.capacity
            self.memory[idx] = embedding
            self.memory_keys[idx] = F.normalize(embedding, dim=-1)
            self.importance[idx] = importance
            self.count += 1
            
    def retrieve(self, query, topk=5):
        """Retrieve most similar memories."""
        if self.count == 0:
            return query.unsqueeze(0).expand(topk, -1)
        
        with torch.no_grad():
            if query.dim() > 1:
                query = query.mean(dim=0)
            
            query_norm = F.normalize(query, dim=-1)
            n_valid = min(self.count.item(), self.capacity)
            
            similarities = query_norm @ self.memory_keys[:n_valid].T
            similarities = similarities * (self.importance[:n_valid] + 0.1)
            
            k = min(topk, n_valid)
            _, indices = similarities.topk(k)
            return self.memory[indices]
    
    def compute_novelty(self, embedding):
        """How novel is this embedding compared to memory?"""
        if self.count == 0:
            return torch.tensor(1.0, device=embedding.device)
        
        with torch.no_grad():
            if embedding.dim() > 1:
                embedding = embedding.mean(dim=0)
            
            query_norm = F.normalize(embedding, dim=-1)
            n_valid = min(self.count.item(), self.capacity)
            
            similarities = query_norm @ self.memory_keys[:n_valid].T
            max_sim = similarities.max()
            return 1.0 - max_sim.clamp(0, 1)

# =============================================================================
# HIERARCHICAL LAYERS (same as Phase 2)
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

class PhraseChunker(nn.Module):
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
        return self.project(x[:, 0])

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
# COMPLETE BRAIN MODEL
# =============================================================================

class HierarchicalGermanBrain(nn.Module):
    """Hierarchical model with integrated brain modules."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syl=128, 
                 d_morph=256, d_word=256, d_phrase=512, d_sent=512, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Hierarchical encoders
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syl)
        self.morpheme_parser = MorphemeParser(d_syl, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sent)
        
        # Decoders
        self.char_decoder = Decoder(d_char, vocab_size)
        self.syl_decoder = Decoder(d_syl, vocab_size)
        self.morph_decoder = Decoder(d_morph, vocab_size)
        self.phrase_decoder = Decoder(d_phrase, vocab_size)
        self.sent_decoder = Decoder(d_sent, vocab_size)
        self.next_char_head = nn.Linear(d_char, vocab_size)
        
        # BRAIN MODULES
        self.dopamine = DopamineSystem()
        self.hebbian_char_syl = HebbianPlasticity(d_char, d_syl)
        self.hebbian_syl_morph = HebbianPlasticity(d_syl, d_morph)
        self.hebbian_morph_word = HebbianPlasticity(d_morph, d_word)
        self.hebbian_word_phrase = HebbianPlasticity(d_word, d_phrase)
        self.hippocampus = Hippocampus(d_word, capacity=10000)
        
    def forward(self, char_indices, update_brain=True):
        # Hierarchical forward pass
        char_emb = self.char_encoder(char_indices)
        syl_boundaries, syl_emb = self.syllable_detector(char_emb)
        morph_boundaries, morph_types, morph_emb = self.morpheme_parser(syl_emb)
        word_boundaries, pos_tags, word_emb = self.word_composer(morph_emb)
        phrase_boundaries, phrase_types, phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        
        # Brain updates during training
        if update_brain and self.training:
            self.hebbian_char_syl.update(char_emb.mean(dim=1), syl_emb.mean(dim=1))
            self.hebbian_syl_morph.update(syl_emb.mean(dim=1), morph_emb.mean(dim=1))
            self.hebbian_morph_word.update(morph_emb.mean(dim=1), word_emb.mean(dim=1))
            self.hebbian_word_phrase.update(word_emb.mean(dim=1), phrase_emb.mean(dim=1))
            
            # Store word embeddings in hippocampus
            for i in range(word_emb.size(0)):
                novelty = self.hippocampus.compute_novelty(word_emb[i])
                if novelty > 0.3:
                    self.hippocampus.encode(word_emb[i], importance=novelty.item())
        
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
    
    def load_phase2_weights(self, checkpoint_path):
        print(f"Loading Phase 2 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        print(f"  Loading {len(pretrained)}/{len(state_dict)} parameters")
        model_dict.update(pretrained)
        self.load_state_dict(model_dict)
    
    def get_brain_stats(self):
        """Return brain module statistics."""
        return {
            'dopamine_baseline': self.dopamine.reward_baseline.item(),
            'hebbian_char_syl': self.hebbian_char_syl.get_association_strength().item(),
            'hebbian_syl_morph': self.hebbian_syl_morph.get_association_strength().item(),
            'hebbian_morph_word': self.hebbian_morph_word.get_association_strength().item(),
            'hebbian_word_phrase': self.hebbian_word_phrase.get_association_strength().item(),
            'hippocampus_count': self.hippocampus.count.item(),
        }

# =============================================================================
# TRAINING
# =============================================================================

def prepare_batch(texts, max_len=128):
    batch_chars, batch_syl, batch_word, batch_phrase = [], [], [], []
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
            if i > 0 and i % 3 == 0:
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
            outputs = model(chars, update_brain=False)
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
            outputs = model(chars, update_brain=False)
            recon = indices_to_text(outputs['char_recon'].argmax(dim=-1)[0].cpu().tolist())
            print(f"In:  '{sent[:50]}'")
            print(f"Out: '{recon[:50]}'")
            print()
    
    max_sim = check_collapse(model, device, sentences)
    print(f"📊 Sentence similarity: {max_sim:.4f}")
    
    stats = model.get_brain_stats()
    print(f"\n🧠 BRAIN STATS:")
    print(f"   Dopamine baseline: {stats['dopamine_baseline']:.4f}")
    print(f"   Hebbian char→syl: {stats['hebbian_char_syl']:.6f}")
    print(f"   Hebbian syl→morph: {stats['hebbian_syl_morph']:.6f}")
    print(f"   Hebbian morph→word: {stats['hebbian_morph_word']:.6f}")
    print(f"   Hebbian word→phrase: {stats['hebbian_word_phrase']:.6f}")
    print(f"   Hippocampus memories: {stats['hippocampus_count']}")
    
    if max_sim < 0.95:
        print("\n✅ Sentence embeddings differentiated")
    else:
        print("\n⚠️  WARNING: Potential collapse!")
    print("="*60)
    model.train()
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

def train(model, sentences, device, epochs=15, batch_size=32, lr=2e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(sentences) // batch_size)
    scaler = GradScaler()
    
    collapse_count = 0
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        random.shuffle(sentences)
        total_loss = 0
        total_dopamine = 0
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
                outputs = model(chars, update_brain=True)
                
                # Reconstruction losses
                char_recon_loss = F.cross_entropy(outputs['char_recon'].view(-1, model.vocab_size), chars.view(-1), ignore_index=0)
                phrase_recon_loss = F.cross_entropy(outputs['phrase_recon'].view(-1, model.vocab_size), chars.view(-1), ignore_index=0)
                sent_recon_loss = F.cross_entropy(outputs['sent_recon'].view(-1, model.vocab_size), chars.view(-1), ignore_index=0)
                recon_loss = char_recon_loss + phrase_recon_loss + sent_recon_loss
                
                # Boundary loss
                phrase_bnd_loss = F.binary_cross_entropy_with_logits(outputs['phrase_boundaries'], phrase_bnd)
                
                # Contrastive loss
                B = chars.shape[0]
                sent_norm = F.normalize(outputs['sent_emb'], dim=-1)
                sim_matrix = sent_norm @ sent_norm.T
                off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=device)
                off_diag_sim = sim_matrix[off_diag_mask]
                contrastive_loss = torch.relu(off_diag_sim - 0.5).mean() * 10.0
                
                # Dopamine modulation
                reward = model.dopamine.compute_reward(outputs['char_recon'], chars)
                rpe = model.dopamine(reward)
                lr_mult = model.dopamine.get_learning_rate_multiplier(rpe)
                
                loss = (recon_loss + phrase_bnd_loss * 0.5 + contrastive_loss) * lr_mult
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_dopamine += rpe.item()
            num_batches += 1
            
            if num_batches % 30 == 0:
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}', 
                                 'DA': f'{total_dopamine/num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        avg_da = total_dopamine / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Dopamine={avg_da:.4f}")
        
        max_sim = test_model(model, device, TEST_SENTENCES)
        
        if max_sim > 0.95:
            collapse_count += 1
            print(f"⚠️  Collapse warning! Count: {collapse_count}/3")
            if collapse_count >= 3:
                print("❌ STOPPING: Embedding collapse!")
                break
        else:
            collapse_count = 0
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "checkpoints/phase3_best.pth")
                print(f"💾 Saved (loss={avg_loss:.4f}, sim={max_sim:.4f})")
    
    print("\n✅ Training complete!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model = HierarchicalGermanBrain().to(device)
    model.load_phase2_weights("checkpoints/phase2_best.pth")
    
    sentences = load_german_sentences(50000)
    if not sentences:
        sentences = TEST_SENTENCES * 100
    
    train(model, sentences, device, epochs=15, batch_size=32)

if __name__ == "__main__":
    main()
