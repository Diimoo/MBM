#!/usr/bin/env python3
"""
Hierarchical German Language Model - Phase 5: Enhanced Brain + Fine-tuning

New brain modules:
- Serotonin: exploration vs exploitation (temperature modulation)
- Norepinephrine: attention/arousal based on novelty

Extended training with all neuromodulators working together.
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
# ENHANCED BRAIN MODULES
# =============================================================================

class DopamineSystem(nn.Module):
    """Reward prediction error - modulates learning rate."""
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
    
    def get_learning_multiplier(self, rpe):
        """Higher RPE = more learning."""
        return 1.0 + torch.clamp(rpe, -0.5, 0.5)


class SerotoninSystem(nn.Module):
    """
    Serotonin: Controls exploration vs exploitation.
    - High serotonin = more exploitation (lower temperature, more confident)
    - Low serotonin = more exploration (higher temperature, more random)
    
    Modulated by: prediction consistency, reward stability
    """
    def __init__(self, tau=0.95):
        super().__init__()
        self.tau = tau
        self.register_buffer('serotonin_level', torch.tensor(0.5))
        self.register_buffer('reward_variance', torch.tensor(0.1))
        self.register_buffer('recent_rewards', torch.zeros(50))
        self.register_buffer('reward_idx', torch.tensor(0))
        
    def update(self, reward, prediction_confidence):
        """
        Update serotonin based on reward stability and confidence.
        Stable rewards + high confidence = increase serotonin (exploit)
        Variable rewards + low confidence = decrease serotonin (explore)
        """
        with torch.no_grad():
            idx = self.reward_idx.item() % 50
            self.recent_rewards[idx] = reward
            self.reward_idx += 1
            
            if self.reward_idx >= 10:
                n_valid = min(self.reward_idx.item(), 50)
                self.reward_variance = self.recent_rewards[:n_valid].var()
            
            stability = 1.0 / (1.0 + self.reward_variance * 10)
            target_serotonin = (stability + prediction_confidence) / 2
            self.serotonin_level = self.tau * self.serotonin_level + (1 - self.tau) * target_serotonin
            
    def get_temperature(self, base_temp=0.8):
        """
        High serotonin = lower temperature (more exploitation)
        Low serotonin = higher temperature (more exploration)
        """
        temp_modifier = 1.5 - self.serotonin_level.item()
        return base_temp * max(0.5, min(2.0, temp_modifier))
    
    def get_exploration_bonus(self):
        """Bonus for exploring novel outputs when serotonin is low."""
        return (1.0 - self.serotonin_level) * 0.1


class NorepinephrineSystem(nn.Module):
    """
    Norepinephrine: Attention and arousal based on novelty/surprise.
    - High NE = heightened attention, focus on unexpected inputs
    - Low NE = relaxed processing, routine patterns
    
    Modulated by: input novelty, prediction errors
    """
    def __init__(self, tau=0.9):
        super().__init__()
        self.tau = tau
        self.register_buffer('ne_level', torch.tensor(0.5))
        self.register_buffer('baseline_novelty', torch.tensor(0.3))
        
    def update(self, novelty, prediction_error):
        """
        Update NE based on novelty and prediction error.
        High novelty + high error = increase NE (pay attention!)
        Low novelty + low error = decrease NE (routine)
        """
        with torch.no_grad():
            arousal = (novelty + prediction_error) / 2
            self.ne_level = self.tau * self.ne_level + (1 - self.tau) * arousal
            self.baseline_novelty = 0.99 * self.baseline_novelty + 0.01 * novelty
            
    def get_attention_weight(self):
        """
        Higher NE = more attention weight on current input.
        Returns multiplier for attention mechanisms.
        """
        return 0.8 + self.ne_level.item() * 0.4
    
    def get_learning_boost(self):
        """Novel/surprising inputs should be learned more."""
        return 1.0 + self.ne_level.item() * 0.3


class HebbianPlasticity(nn.Module):
    """Strengthen associations between hierarchical levels."""
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


class NeuromodulatorSystem(nn.Module):
    """Unified neuromodulator system combining DA, 5-HT, NE."""
    def __init__(self):
        super().__init__()
        self.dopamine = DopamineSystem()
        self.serotonin = SerotoninSystem()
        self.norepinephrine = NorepinephrineSystem()
        
    def update(self, reward, novelty, prediction_error, confidence):
        """Update all neuromodulators."""
        rpe = self.dopamine(reward)
        self.serotonin.update(reward, confidence)
        self.norepinephrine.update(novelty, prediction_error)
        return rpe
    
    def get_modulation(self):
        """Get combined modulation factors."""
        return {
            'learning_rate': self.dopamine.get_learning_multiplier(
                self.dopamine.reward_baseline - self.dopamine.reward_history.mean()
            ).item() * self.norepinephrine.get_learning_boost(),
            'temperature': self.serotonin.get_temperature(),
            'attention': self.norepinephrine.get_attention_weight(),
            'exploration': self.serotonin.get_exploration_bonus().item(),
        }
    
    def get_stats(self):
        return {
            'dopamine_baseline': self.dopamine.reward_baseline.item(),
            'serotonin_level': self.serotonin.serotonin_level.item(),
            'norepinephrine_level': self.norepinephrine.ne_level.item(),
        }

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
# AUTOREGRESSIVE GENERATOR
# =============================================================================

class AutoregressiveGenerator(nn.Module):
    def __init__(self, d_sent=512, d_hidden=512, vocab_size=VOCAB_SIZE, max_len=128):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.sent_project = nn.Linear(d_sent, d_hidden)
        self.char_embed = nn.Embedding(vocab_size, d_hidden)
        self.pos_embed = nn.Embedding(max_len, d_hidden)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_hidden, nhead=8, dim_feedforward=d_hidden * 4,
            dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.output_head = nn.Linear(d_hidden, vocab_size)
        self.register_buffer('causal_mask', None)
        
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            self.causal_mask = mask
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, sent_emb, target_chars=None):
        B = sent_emb.size(0)
        device = sent_emb.device
        memory = self.sent_project(sent_emb).unsqueeze(1)
        if target_chars is not None:
            seq_len = target_chars.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            char_emb = self.char_embed(target_chars) + self.pos_embed(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.transformer(char_emb, memory, tgt_mask=causal_mask)
            return self.output_head(output)
        return None
    
    @torch.no_grad()
    def generate(self, sent_emb, max_len=100, temperature=0.8, top_k=50, top_p=0.9):
        B = sent_emb.size(0)
        device = sent_emb.device
        memory = self.sent_project(sent_emb).unsqueeze(1)
        generated = torch.full((B, 1), char_to_idx['<BOS>'], dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            seq_len = generated.size(1)
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(B, -1)
            char_emb = self.char_embed(generated) + self.pos_embed(positions)
            causal_mask = self._get_causal_mask(seq_len, device)
            output = self.transformer(char_emb, memory, tgt_mask=causal_mask)
            logits = self.output_head(output[:, -1, :]) / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == char_to_idx['<EOS>']).all():
                break
        
        return generated

# =============================================================================
# COMPLETE MODEL
# =============================================================================

class HierarchicalGermanBrainV2(nn.Module):
    """Full model with enhanced neuromodulators."""
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
        
        # Decoders
        self.char_decoder = Decoder(d_char, vocab_size)
        self.generator = AutoregressiveGenerator(d_sent, d_phrase, vocab_size, max_len)
        
        # Enhanced brain modules
        self.neuromodulators = NeuromodulatorSystem()
        self.hebbian_char_syl = HebbianPlasticity(d_char, d_syl)
        self.hebbian_syl_morph = HebbianPlasticity(d_syl, d_morph)
        self.hebbian_morph_word = HebbianPlasticity(d_morph, d_word)
        self.hebbian_word_phrase = HebbianPlasticity(d_word, d_phrase)
        self.hippocampus = Hippocampus(d_word, capacity=10000)
        
    def encode(self, char_indices):
        char_emb = self.char_encoder(char_indices)
        _, syl_emb = self.syllable_detector(char_emb)
        _, _, morph_emb = self.morpheme_parser(syl_emb)
        _, _, word_emb = self.word_composer(morph_emb)
        _, _, phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
        return sent_emb
    
    def forward(self, char_indices, update_brain=True):
        char_emb = self.char_encoder(char_indices)
        _, syl_emb = self.syllable_detector(char_emb)
        _, _, morph_emb = self.morpheme_parser(syl_emb)
        _, _, word_emb = self.word_composer(morph_emb)
        _, _, phrase_emb = self.phrase_chunker(word_emb)
        sent_emb = self.sentence_encoder(phrase_emb)
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
            'char_emb': char_emb, 'word_emb': word_emb, 'sent_emb': sent_emb,
            'char_recon': self.char_decoder(char_emb),
            'gen_logits': gen_logits,
        }
    
    @torch.no_grad()
    def generate(self, prompt=None, max_len=100, temperature=None, top_k=50, top_p=0.9):
        self.eval()
        device = next(self.parameters()).device
        
        # Use serotonin-modulated temperature if not specified
        if temperature is None:
            temperature = self.neuromodulators.serotonin.get_temperature()
        
        if prompt is not None:
            chars = torch.tensor([text_to_indices(prompt, self.max_len)], device=device)
            sent_emb = self.encode(chars)
        else:
            sent_emb = torch.randn(1, 512, device=device) * 0.5
        
        generated = self.generator.generate(sent_emb, max_len, temperature, top_k, top_p)
        return indices_to_text(generated[0].cpu().tolist())
    
    def load_phase4_weights(self, checkpoint_path):
        print(f"Loading Phase 4 weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model_dict = self.state_dict()
        pretrained = {k: v for k, v in state_dict.items() 
                     if k in model_dict and model_dict[k].shape == v.shape}
        print(f"  Loading {len(pretrained)}/{len(state_dict)} parameters")
        model_dict.update(pretrained)
        self.load_state_dict(model_dict)
    
    def get_brain_stats(self):
        stats = self.neuromodulators.get_stats()
        stats.update({
            'hebbian_char_syl': self.hebbian_char_syl.get_association_strength().item(),
            'hebbian_morph_word': self.hebbian_morph_word.get_association_strength().item(),
            'hippocampus_count': self.hippocampus.count.item(),
        })
        return stats

# =============================================================================
# TRAINING
# =============================================================================

def prepare_batch(texts, max_len=128):
    batch_chars = [text_to_indices(text, max_len) for text in texts]
    return torch.tensor(batch_chars)

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
    "Der Mann liest ein interessantes Buch.",
    "Wir gehen heute in die Stadt.",
    "Die Kinder spielen fröhlich im Garten.",
    "Sie möchte morgen nach Berlin fahren.",
    "Das Wetter ist heute besonders schön.",
    "Die Freundlichkeit der Menschen beeindruckt mich.",
]

def test_generation(model, device):
    model.eval()
    print("\n" + "="*60)
    print("GENERATION TEST")
    print("="*60)
    
    stats = model.get_brain_stats()
    print(f"\n🧠 BRAIN STATE:")
    print(f"   Dopamine: {stats['dopamine_baseline']:.4f}")
    print(f"   Serotonin: {stats['serotonin_level']:.4f} (temp={model.neuromodulators.serotonin.get_temperature():.2f})")
    print(f"   Norepinephrine: {stats['norepinephrine_level']:.4f}")
    print(f"   Hippocampus: {stats['hippocampus_count']} memories\n")
    
    prompts = ["Der Hund", "Die Katze", "Ein Kind", "Es war einmal", "Heute ist"]
    
    for prompt in prompts:
        generated = model.generate(prompt, max_len=80)
        print(f"'{prompt}' → '{generated}'")
    
    print("\nRandom generation:")
    for i in range(3):
        generated = model.generate(None, max_len=60)
        print(f"  {i+1}. '{generated}'")
    
    print("="*60)
    model.train()  # Switch back to training mode!

def train(model, sentences, device, epochs=12, batch_size=32, lr=1e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)  # More regularization
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs * len(sentences) // batch_size)
    scaler = GradScaler()
    
    best_loss = float('inf')
    patience = 3
    no_improve = 0
    
    for epoch in range(1, epochs + 1):
        random.shuffle(sentences)
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences), batch_size), desc=f"Epoch {epoch}/{epochs}")
        for i in pbar:
            batch = sentences[i:i+batch_size]
            if len(batch) < 2:
                continue
            
            chars = prepare_batch(batch).to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(chars, update_brain=True)
                
                # Generation loss with label smoothing
                gen_logits = outputs['gen_logits']
                targets = chars[:, 1:]
                gen_loss = F.cross_entropy(
                    gen_logits[:, :-1].reshape(-1, model.vocab_size),
                    targets.reshape(-1), ignore_index=0, label_smoothing=0.1)
                
                # Reconstruction loss
                recon_loss = F.cross_entropy(
                    outputs['char_recon'].view(-1, model.vocab_size),
                    chars.view(-1), ignore_index=0) * 0.1
                
                # Compute metrics for neuromodulators
                with torch.no_grad():
                    pred_ids = gen_logits[:, :-1].argmax(dim=-1)
                    correct = (pred_ids == targets).float()
                    mask = (targets != 0).float()
                    accuracy = (correct * mask).sum() / (mask.sum() + 1e-8)
                    confidence = F.softmax(gen_logits[:, :-1], dim=-1).max(dim=-1)[0].mean()
                    novelty = model.hippocampus.compute_novelty(outputs['word_emb'].mean(dim=1))
                    if isinstance(novelty, float):
                        novelty = torch.tensor(novelty)
                
                # Update neuromodulators
                rpe = model.neuromodulators.update(
                    accuracy.item(), novelty.mean().item(),
                    gen_loss.item(), confidence.item())
                
                # Get modulation
                mod = model.neuromodulators.get_modulation()
                
                # Apply learning rate modulation
                loss = (gen_loss + recon_loss) * mod['learning_rate']
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches % 30 == 0:
                stats = model.neuromodulators.get_stats()
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'DA': f'{stats["dopamine_baseline"]:.2f}',
                    '5HT': f'{stats["serotonin_level"]:.2f}',
                    'NE': f'{stats["norepinephrine_level"]:.2f}'
                })
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}")
        
        test_generation(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(model.state_dict(), "checkpoints/phase5_best.pth")
            print(f"💾 Saved (loss={avg_loss:.4f})")
        else:
            no_improve += 1
            print(f"⚠️ No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("🛑 Early stopping!")
                break
    
    print("\n✅ Training complete!")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs("checkpoints", exist_ok=True)
    
    model = HierarchicalGermanBrainV2().to(device)
    model.load_phase4_weights("checkpoints/phase4_best.pth")
    
    sentences = load_german_sentences(50000)
    if not sentences:
        sentences = TEST_SENTENCES * 100
    
    print("\n🧠 Training with enhanced neuromodulators (DA + 5-HT + NE)")
    train(model, sentences, device, epochs=20, batch_size=32)
    
    print("\n" + "="*60)
    print("FINAL DEMO")
    print("="*60)
    model.load_state_dict(torch.load("checkpoints/phase5_best.pth", map_location=device))
    test_generation(model, device)

if __name__ == "__main__":
    main()
