#!/usr/bin/env python3
"""
Hierarchical German Language Model - Phase 4: Text Generation

Adds generation capabilities:
1. Autoregressive character generation
2. Sentence embedding → text decoding
3. Constrained generation (prompt continuation)
4. Temperature/top-k/top-p sampling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm
import os
import random

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

# =============================================================================
# BRAIN MODULES (from Phase 3)
# =============================================================================

class DopamineSystem(nn.Module):
    def __init__(self, baseline_tau=0.99):
        super().__init__()
        self.baseline_tau = baseline_tau
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        self.register_buffer('reward_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def compute_reward(self, predictions, targets, mask=None):
        pred_ids = predictions.argmax(dim=-1)
        correct = (pred_ids == targets).float()
        if mask is not None:
            correct = correct * mask
            accuracy = correct.sum() / (mask.sum() + 1e-8)
        else:
            accuracy = correct.mean()
        return accuracy
    
    def forward(self, reward):
        with torch.no_grad():
            rpe = reward - self.reward_baseline
            self.reward_baseline = self.baseline_tau * self.reward_baseline + (1 - self.baseline_tau) * reward
            idx = self.history_idx.item() % 100
            self.reward_history[idx] = reward
            self.history_idx += 1
        return rpe
    
    def get_learning_rate_multiplier(self, rpe):
        return 1.0 + torch.clamp(rpe, -0.5, 0.5)


class HebbianPlasticity(nn.Module):
    def __init__(self, d_pre, d_post, tau=0.995):
        super().__init__()
        self.tau = tau
        self.register_buffer('association_matrix', torch.zeros(d_pre, d_post))
        self.register_buffer('pre_mean', torch.zeros(d_pre))
        self.register_buffer('post_mean', torch.zeros(d_post))
        
    def update(self, pre_activity, post_activity):
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
        return self.association_matrix.abs().mean()


class Hippocampus(nn.Module):
    def __init__(self, d_emb, capacity=10000):
        super().__init__()
        self.d_emb = d_emb
        self.capacity = capacity
        self.register_buffer('memory', torch.zeros(capacity, d_emb))
        self.register_buffer('memory_keys', torch.zeros(capacity, d_emb))
        self.register_buffer('count', torch.tensor(0))
        self.register_buffer('importance', torch.zeros(capacity))
        
    def encode(self, embedding, importance=1.0):
        with torch.no_grad():
            if embedding.dim() > 1:
                embedding = embedding.mean(dim=0)
            idx = self.count.item() % self.capacity
            self.memory[idx] = embedding
            self.memory_keys[idx] = F.normalize(embedding, dim=-1)
            self.importance[idx] = importance
            self.count += 1
            
    def retrieve(self, query, topk=5):
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
# HIERARCHICAL LAYERS
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
# AUTOREGRESSIVE GENERATOR (NEW!)
# =============================================================================

class AutoregressiveGenerator(nn.Module):
    """Generate text character by character."""
    def __init__(self, d_sent=512, d_hidden=512, vocab_size=VOCAB_SIZE, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Sentence conditioning
        self.sent_project = nn.Linear(d_sent, d_hidden)
        
        # Character embedding for autoregressive
        self.char_embed = nn.Embedding(vocab_size, d_hidden)
        self.pos_embed = nn.Embedding(max_len, d_hidden)
        
        # Transformer decoder for generation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_hidden, nhead=8, dim_feedforward=d_hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # Output projection
        self.output_head = nn.Linear(d_hidden, vocab_size)
        
        # Causal mask cache
        self.register_buffer('causal_mask', None)
        
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, sent_emb, target_chars=None):
        """
        Training: given sentence embedding and target chars, predict next char.
        """
        B = sent_emb.size(0)
        device = sent_emb.device
        
        # Project sentence embedding as memory
        memory = self.sent_project(sent_emb).unsqueeze(1)  # [B, 1, d_hidden]
        
        if target_chars is not None:
            # Training mode: teacher forcing
            seq_len = target_chars.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            
            char_emb = self.char_embed(target_chars) + self.pos_embed(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            
            output = self.transformer(char_emb, memory, tgt_mask=causal_mask)
            logits = self.output_head(output)
            return logits
        
        return None
    
    @torch.no_grad()
    def generate(self, sent_emb, max_len=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text from sentence embedding."""
        B = sent_emb.size(0)
        device = sent_emb.device
        
        memory = self.sent_project(sent_emb).unsqueeze(1)
        
        # Start with BOS token
        generated = torch.full((B, 1), char_to_idx['<BOS>'], dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            
            char_emb = self.char_embed(generated) + self.pos_embed(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            
            output = self.transformer(char_emb, memory, tgt_mask=causal_mask)
            logits = self.output_head(output[:, -1, :])  # Last position
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if EOS
            if (next_token == char_to_idx['<EOS>']).all():
                break
        
        return generated

# =============================================================================
# COMPLETE MODEL WITH GENERATION
# =============================================================================

class HierarchicalGermanGenerator(nn.Module):
    """Full model with encoding + generation."""
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_syl=128, 
                 d_morph=256, d_word=256, d_phrase=512, d_sent=512, max_len=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        
        # Encoders
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syl)
        self.morpheme_parser = MorphemeParser(d_syl, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sent)
        
        # Decoders (reconstruction)
        self.char_decoder = Decoder(d_char, vocab_size)
        self.syl_decoder = Decoder(d_syl, vocab_size)
        self.morph_decoder = Decoder(d_morph, vocab_size)
        self.phrase_decoder = Decoder(d_phrase, vocab_size)
        self.sent_decoder = Decoder(d_sent, vocab_size)
        self.next_char_head = nn.Linear(d_char, vocab_size)
        
        # Generator (NEW!)
        self.generator = AutoregressiveGenerator(d_sent, d_phrase, vocab_size, max_len)
        
        # Brain modules
        self.dopamine = DopamineSystem()
        self.hebbian_char_syl = HebbianPlasticity(d_char, d_syl)
        self.hebbian_syl_morph = HebbianPlasticity(d_syl, d_morph)
        self.hebbian_morph_word = HebbianPlasticity(d_morph, d_word)
        self.hebbian_word_phrase = HebbianPlasticity(d_word, d_phrase)
        self.hippocampus = Hippocampus(d_word, capacity=10000)
        
    def encode(self, char_indices):
        """Encode text to sentence embedding."""
        char_emb = self.char_encoder(char_indices)
        _, syl_emb = self.syllable_detector(char_emb)
        _, _, morph_emb = self.morpheme_parser(syl_emb)
        _, _, word_emb = self.word_composer(morph_emb)
        _, _, phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        return sent_emb
    
    def forward(self, char_indices, update_brain=True):
        """Full forward pass with all outputs."""
        char_emb = self.char_encoder(char_indices)
        syl_boundaries, syl_emb = self.syllable_detector(char_emb)
        morph_boundaries, morph_types, morph_emb = self.morpheme_parser(syl_emb)
        word_boundaries, pos_tags, word_emb = self.word_composer(morph_emb)
        phrase_boundaries, phrase_types, phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        
        # Generator training
        gen_logits = self.generator(sent_emb, char_indices)
        
        if update_brain and self.training:
            self.hebbian_char_syl.update(char_emb.mean(dim=1), syl_emb.mean(dim=1))
            self.hebbian_syl_morph.update(syl_emb.mean(dim=1), morph_emb.mean(dim=1))
            self.hebbian_morph_word.update(morph_emb.mean(dim=1), word_emb.mean(dim=1))
            self.hebbian_word_phrase.update(word_emb.mean(dim=1), phrase_emb.mean(dim=1))
            for i in range(word_emb.size(0)):
                novelty = self.hippocampus.compute_novelty(word_emb[i])
                if novelty > 0.3:
                    self.hippocampus.encode(word_emb[i], importance=novelty.item())
        
        return {
            'char_emb': char_emb, 'syl_emb': syl_emb, 'morph_emb': morph_emb,
            'word_emb': word_emb, 'phrase_emb': phrase_emb, 'sent_emb': sent_emb,
            'char_recon': self.char_decoder(char_emb),
            'gen_logits': gen_logits,
            'phrase_boundaries': phrase_boundaries,
        }
    
    @torch.no_grad()
    def generate(self, prompt=None, max_len=100, temperature=0.8, top_k=50, top_p=0.9):
        """Generate text, optionally from a prompt."""
        self.eval()
        device = next(self.parameters()).device
        
        if prompt is not None:
            # Encode prompt to get sentence embedding
            chars = torch.tensor([text_to_indices(prompt, self.max_len)], device=device)
            sent_emb = self.encode(chars)
        else:
            # Random sentence embedding
            sent_emb = torch.randn(1, 512, device=device) * 0.5
        
        generated = self.generator.generate(sent_emb, max_len, temperature, top_k, top_p)
        return indices_to_text(generated[0].cpu().tolist())
    
    def load_phase3_weights(self, checkpoint_path):
        print(f"Loading Phase 3 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        print(f"  Loading {len(pretrained)}/{len(state_dict)} parameters")
        model_dict.update(pretrained)
        self.load_state_dict(model_dict)
        
        # Freeze encoder, train generator
        for name, param in self.named_parameters():
            if 'generator' not in name:
                param.requires_grad = False
        frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        trainable = sum(1 for p in self.parameters() if p.requires_grad)
        print(f"  Frozen: {frozen}, Trainable: {trainable} (generator only)")

# =============================================================================
# TRAINING
# =============================================================================

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
                if j - 1 > i:
                    boundaries[j - 1] = 1
        i += 1
    return boundaries

def prepare_batch(texts, max_len=128):
    batch_chars = []
    batch_phrase = []
    for text in texts:
        chars = text_to_indices(text, max_len)
        batch_chars.append(chars)
        phrase_bnd = [0] * max_len
        pos = 1
        for i, word in enumerate(text.split()):
            if pos >= max_len - 1:
                break
            if i > 0 and i % 3 == 0:
                phrase_bnd[pos] = 1
            pos += len(word) + 1
        batch_phrase.append(phrase_bnd)
    return torch.tensor(batch_chars), torch.tensor(batch_phrase, dtype=torch.float)

def load_german_sentences(max_sentences=50000):
    import re
    from datasets import load_dataset
    print("Loading German sentences...")
    sentences = []
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_sentences):
            text = item.get('text', '')
            if isinstance(text, str):
                for sent in re.split(r'(?<=[.!?])\s+', text):
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

def test_generation(model, device):
    model.eval()
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)
    
    # Generate from prompts
    prompts = [
        "Der Hund",
        "Die Katze",
        "Ein Kind",
        "Es war einmal",
        "Heute ist",
    ]
    
    for prompt in prompts:
        generated = model.generate(prompt, max_len=80, temperature=0.8)
        print(f"Prompt: '{prompt}'")
        print(f"Generated: '{generated}'")
        print()
    
    # Random generation
    print("Random generation (no prompt):")
    for i in range(3):
        generated = model.generate(None, max_len=60, temperature=1.0)
        print(f"  {i+1}. '{generated}'")
    
    print("="*60)

def train(model, sentences, device, epochs=10, batch_size=32, lr=3e-4):
    from torch.amp import GradScaler
    
    model.train()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(sentences) // batch_size)
    scaler = GradScaler()
    
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
            
            chars, phrase_bnd = prepare_batch(batch)
            chars = chars.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(chars, update_brain=False)
                
                # Generation loss (main objective)
                gen_logits = outputs['gen_logits']
                targets = chars[:, 1:]  # Shift by 1
                gen_loss = F.cross_entropy(
                    gen_logits[:, :-1].reshape(-1, model.vocab_size),
                    targets.reshape(-1),
                    ignore_index=0
                )
                
                # Reconstruction loss (auxiliary)
                recon_loss = F.cross_entropy(
                    outputs['char_recon'].view(-1, model.vocab_size),
                    chars.view(-1),
                    ignore_index=0
                ) * 0.1
                
                loss = gen_loss + recon_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 30 == 0:
                pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        test_generation(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/phase4_best.pth")
            print(f"💾 Saved (loss={avg_loss:.4f})")
    
    print("\n✅ Training complete!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model = HierarchicalGermanGenerator().to(device)
    model.load_phase3_weights("checkpoints/phase3_best.pth")
    
    sentences = load_german_sentences(50000)
    if not sentences:
        sentences = TEST_SENTENCES * 100
    
    train(model, sentences, device, epochs=10, batch_size=32)
    
    # Final generation demo
    print("\n" + "="*60)
    print("FINAL GENERATION DEMO")
    print("="*60)
    model.load_state_dict(torch.load("checkpoints/phase4_best.pth", map_location=device))
    test_generation(model, device)

if __name__ == "__main__":
    main()
