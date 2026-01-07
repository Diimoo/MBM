#!/usr/bin/env python3
"""
Hierarchical German Language Model with Brain Integration

Full hierarchy: Chars → Syllables → Morphemes → Words → Phrases → Sentences
Brain integration:
  - Dopamine: reward signal for correct word generation
  - Hebbian: strengthen syllable→morpheme→word associations
  - Hippocampus: vocabulary memory storage and retrieval
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

# Import brain modules
import sys
sys.path.insert(0, '/home/ahmed/Downloads/Kandel')
from digital_brain.modules.plasticity import SynapticPlasticity
from digital_brain.modules.hippocampus import Hippocampus

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

char_to_idx = {'<PAD>': 0, '<UNK>': 1}
for c in CHARS:
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)
idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

def text_to_indices(text, max_len=128):
    indices = [char_to_idx.get(c, 1) for c in text[:max_len]]
    while len(indices) < max_len:
        indices.append(0)
    return indices


# =============================================================================
# HIERARCHICAL LAYERS (Levels 0-5)
# =============================================================================

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, d_char=128, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_char, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_char)
        self.conv = nn.Sequential(
            nn.Conv1d(d_char, d_char, 3, padding=1), nn.GELU(),
            nn.Conv1d(d_char, d_char, 3, padding=1)
        )
        self.norm = nn.LayerNorm(d_char)
        
    def forward(self, x):
        B, L = x.shape
        emb = self.embed(x) + self.pos(torch.arange(L, device=x.device).unsqueeze(0).expand(B,-1))
        return self.norm(emb + self.conv(emb.transpose(1,2)).transpose(1,2))


class SyllableDetector(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.lstm = nn.LSTM(d, d//2, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out).squeeze(-1), out  # Return features too


class MorphemeParser(nn.Module):
    def __init__(self, d_in=128, d_out=256):
        super().__init__()
        self.lstm = nn.LSTM(d_in, d_in, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary = nn.Sequential(nn.Linear(d_in*2, d_in), nn.GELU(), nn.Linear(d_in, 1))
        self.types = nn.Sequential(nn.Linear(d_in*2, d_in), nn.GELU(), nn.Linear(d_in, 5))
        self.project = nn.Sequential(nn.Linear(d_in*2, d_out), nn.LayerNorm(d_out), nn.GELU())
        
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.boundary(out).squeeze(-1), self.types(out), self.project(out)


class WordComposer(nn.Module):
    def __init__(self, d_morph=256, d_word=512):
        super().__init__()
        self.type_emb = nn.Embedding(5, d_morph)
        self.attn = nn.MultiheadAttention(d_morph, 8, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_morph, 8, d_morph*4, 0.1, batch_first=True), 2
        )
        self.project = nn.Sequential(nn.Linear(d_morph, d_word), nn.LayerNorm(d_word), nn.GELU())
        self.boundary = nn.Sequential(nn.Linear(d_morph, d_morph//2), nn.GELU(), nn.Linear(d_morph//2, 1))
        
    def forward(self, morph_emb, types=None):
        if types is not None:
            morph_emb = morph_emb + self.type_emb(types)
        attended, _ = self.attn(morph_emb, morph_emb, morph_emb)
        composed = self.transformer(attended)
        return self.project(composed), self.boundary(composed).squeeze(-1)


class PhraseChunker(nn.Module):
    def __init__(self, d_word=512, d_phrase=512):
        super().__init__()
        self.lstm = nn.LSTM(d_word, d_word//2, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary = nn.Sequential(nn.Linear(d_word, d_word//2), nn.GELU(), nn.Linear(d_word//2, 1))
        self.types = nn.Sequential(nn.Linear(d_word, d_word//2), nn.GELU(), nn.Linear(d_word//2, 6))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_word, 8, d_word*4, 0.1, batch_first=True), 3
        )
        self.project = nn.Sequential(nn.Linear(d_word, d_phrase), nn.LayerNorm(d_phrase), nn.GELU())
        
    def forward(self, word_emb):
        lstm_out, _ = self.lstm(word_emb)
        composed = self.transformer(word_emb)
        return self.boundary(lstm_out).squeeze(-1), self.types(lstm_out), self.project(composed)


class SentenceEncoder(nn.Module):
    def __init__(self, d_phrase=512, d_sent=768):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_phrase, 8, d_phrase*4, 0.1, batch_first=True), 4
        )
        self.project = nn.Sequential(nn.Linear(d_phrase, d_sent), nn.LayerNorm(d_sent), nn.GELU())
        self.pool = nn.Sequential(nn.Linear(d_sent, d_sent), nn.Tanh())
        
    def forward(self, phrase_emb):
        composed = self.transformer(phrase_emb)
        sent_seq = self.project(composed)
        sent_vec = self.pool(sent_seq.mean(dim=1))
        return sent_seq, sent_vec


# =============================================================================
# DOPAMINE MODULE - Reward for correct predictions
# =============================================================================

class DopamineSystem(nn.Module):
    """
    Computes dopamine signal based on prediction accuracy.
    DA = reward - baseline (TD-like error)
    """
    def __init__(self, baseline_tau=0.99):
        super().__init__()
        self.baseline_tau = baseline_tau
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        
    def compute_reward(self, predictions, targets, ignore_idx=0):
        """Compute reward based on prediction accuracy."""
        mask = (targets != ignore_idx).float()
        if mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)
        
        pred_ids = predictions.argmax(dim=-1)
        correct = (pred_ids == targets).float() * mask
        accuracy = correct.sum() / (mask.sum() + 1e-8)
        return accuracy
    
    def forward(self, reward):
        """Compute dopamine signal (reward prediction error)."""
        # Update baseline with exponential moving average
        with torch.no_grad():
            self.reward_baseline = self.baseline_tau * self.reward_baseline + (1 - self.baseline_tau) * reward
        
        # Dopamine = reward - expected reward (RPE)
        dopamine = reward - self.reward_baseline
        return dopamine


# =============================================================================
# VOCABULARY MEMORY (Hippocampus-based)
# =============================================================================

class VocabularyMemory(nn.Module):
    """
    Hippocampus-like memory for learned words.
    Stores word embeddings and retrieves similar ones.
    """
    def __init__(self, d_word=512, capacity=10000):
        super().__init__()
        self.hippocampus = Hippocampus(d_z=d_word, capacity=capacity)
        self.word_to_idx = {}
        self.idx_to_word = {}
        
    def store_word(self, word_embedding, word_text=None):
        """Store a word embedding in memory."""
        self.hippocampus.encode(word_embedding.unsqueeze(0) if word_embedding.dim() == 1 else word_embedding)
        if word_text and word_text not in self.word_to_idx:
            idx = len(self.word_to_idx)
            self.word_to_idx[word_text] = idx
            self.idx_to_word[idx] = word_text
    
    def retrieve_similar(self, query_embedding, topk=5):
        """Retrieve similar words from memory."""
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        return self.hippocampus.retrieve(query_embedding, topk=topk)
    
    def compute_novelty(self, word_embedding):
        """Compute how novel a word is (low similarity = high novelty)."""
        if self.hippocampus.count == 0:
            return torch.ones(word_embedding.shape[0], device=word_embedding.device)
        retrieved = self.retrieve_similar(word_embedding, topk=1)
        similarity = F.cosine_similarity(word_embedding, retrieved, dim=-1)
        novelty = 1.0 - similarity.clamp(0, 1)
        return novelty


# =============================================================================
# FULL BRAIN-INTEGRATED MODEL
# =============================================================================

class HierarchicalGermanBrain(nn.Module):
    """
    Complete hierarchical model with brain integration:
    - Dopamine for reward-based learning
    - Hebbian plasticity for association strengthening
    - Hippocampus for vocabulary memory
    """
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_morph=256,
                 d_word=512, d_phrase=512, d_sent=768, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_word = d_word
        
        # Hierarchical layers
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char)
        self.morpheme_parser = MorphemeParser(d_char, d_morph)
        self.word_composer = WordComposer(d_morph, d_word)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sent)
        
        # Prediction heads
        self.char_predictor = nn.Sequential(
            nn.Linear(d_char, d_char*2), nn.GELU(), nn.Linear(d_char*2, vocab_size)
        )
        self.word_predictor = nn.Sequential(
            nn.Linear(d_word, d_word), nn.GELU(), nn.Linear(d_word, d_word)
        )
        
        # Brain modules
        self.dopamine = DopamineSystem()
        self.plasticity = SynapticPlasticity(tau_e=10.0, learning_rate=1e-4)
        self.vocabulary = VocabularyMemory(d_word=d_word, capacity=10000)
        
        # Hebbian association weights (syllable→word)
        self.register_buffer('syllable_word_trace', torch.zeros(d_char, d_word))
        
    def forward(self, char_indices, train_brain=True):
        # Level 0: Characters
        char_emb = self.char_encoder(char_indices)
        
        # Level 1: Syllables
        syl_logits, syl_features = self.syllable_detector(char_emb)
        
        # Level 2: Morphemes
        morph_bnd, morph_types, morph_emb = self.morpheme_parser(char_emb)
        
        # Level 3: Words
        word_emb, word_bnd = self.word_composer(morph_emb, morph_types.argmax(-1))
        
        # Level 4: Phrases
        phrase_bnd, phrase_types, phrase_emb = self.phrase_chunker(word_emb)
        
        # Level 5: Sentences
        sent_seq, sent_vec = self.sentence_encoder(phrase_emb)
        
        # Predictions
        char_pred = self.char_predictor(char_emb)
        word_pred = self.word_predictor(word_emb)
        
        # Brain computations
        brain_signals = {}
        if train_brain:
            # Compute novelty for each word position
            word_emb_flat = word_emb.view(-1, self.d_word)
            novelty = self.vocabulary.compute_novelty(word_emb_flat)
            brain_signals['novelty'] = novelty.view(word_emb.shape[0], -1)
            
            # Update Hebbian trace (syllable→word association)
            # Average over sequence positions
            syl_avg = syl_features.mean(dim=1)  # [B, d_char]
            word_avg = word_emb.mean(dim=1)     # [B, d_word]
            self.syllable_word_trace = self.plasticity.update_trace(
                self.syllable_word_trace, syl_avg, word_avg
            )
        
        return {
            'char_embeddings': char_emb,
            'syllable_logits': syl_logits,
            'syllable_features': syl_features,
            'morph_boundary_logits': morph_bnd,
            'morph_type_logits': morph_types,
            'morph_embeddings': morph_emb,
            'word_embeddings': word_emb,
            'word_boundary_logits': word_bnd,
            'phrase_boundary_logits': phrase_bnd,
            'phrase_type_logits': phrase_types,
            'phrase_embeddings': phrase_emb,
            'sentence_sequence': sent_seq,
            'sentence_vector': sent_vec,
            'char_predictions': char_pred,
            'word_predictions': word_pred,
            'brain_signals': brain_signals
        }
    
    def compute_brain_loss(self, outputs, char_targets):
        """Compute brain-modulated loss."""
        # Character prediction reward
        char_pred = outputs['char_predictions'][:, :-1, :]
        char_tgt = char_targets[:, 1:]
        reward = self.dopamine.compute_reward(char_pred, char_tgt, ignore_idx=0)
        
        # Dopamine signal (RPE)
        dopamine = self.dopamine(reward)
        
        # Modulate learning based on dopamine
        # Positive dopamine = increase plasticity
        # Negative dopamine = decrease plasticity
        plasticity_scale = torch.sigmoid(dopamine * 5)  # Scale to 0-1
        
        return {
            'reward': reward,
            'dopamine': dopamine,
            'plasticity_scale': plasticity_scale
        }
    
    def store_vocabulary(self, word_embeddings, words=None):
        """Store learned words in vocabulary memory."""
        for i in range(word_embeddings.shape[0]):
            word = words[i] if words else None
            self.vocabulary.store_word(word_embeddings[i], word)


# =============================================================================
# TRAINING
# =============================================================================

def load_sentences(max_sentences=30000):
    print("Loading German sentences...")
    sentences = []
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_sentences * 5):
            text = item.get('text', '')
            if isinstance(text, str):
                for sent in re.split(r'[.!?]+', text):
                    sent = sent.strip()
                    if 15 <= len(sent) <= 80:
                        sentences.append(sent)
                        if len(sentences) >= max_sentences:
                            break
            if len(sentences) >= max_sentences:
                break
    except Exception as e:
        print(f"Error: {e}")
    print(f"Loaded {len(sentences)} sentences")
    return sentences


def train_brain_model(model, sentences, device, epochs=20, batch_size=16, lr=2e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    
    steps = len(sentences) // batch_size
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(sentences)
        
        total_loss = 0
        total_reward = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(sentences) - batch_size, batch_size), desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch = sentences[i:i+batch_size]
            
            # Prepare data
            char_batch = [text_to_indices(s.lower(), 64) for s in batch]
            char_indices = torch.tensor(char_batch, device=device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(char_indices, train_brain=True)
                
                # Standard losses
                char_targets = char_indices[:, 1:]
                char_preds = outputs['char_predictions'][:, :-1, :]
                char_loss = F.cross_entropy(
                    char_preds.reshape(-1, model.vocab_size),
                    char_targets.reshape(-1), ignore_index=0
                )
                
                # Word prediction loss
                word_emb = outputs['word_embeddings']
                word_pred = outputs['word_predictions']
                word_loss = 1 - F.cosine_similarity(
                    word_emb[:, 1:].reshape(-1, model.d_word),
                    word_pred[:, :-1].reshape(-1, model.d_word), dim=-1
                ).mean()
                
                # CONTRASTIVE LOSS - prevent embedding collapse
                # Different sentences should have different embeddings
                sent_vec = outputs['sentence_vector']  # [B, d_sent]
                B = sent_vec.shape[0]
                if B > 1:
                    # Compute pairwise similarities
                    sent_norm = F.normalize(sent_vec, dim=-1)
                    sim_matrix = sent_norm @ sent_norm.T  # [B, B]
                    # Off-diagonal should be low (different sentences = different embeddings)
                    mask = ~torch.eye(B, dtype=torch.bool, device=device)
                    off_diag_sim = sim_matrix[mask].mean()
                    # Penalize high similarity between different samples
                    contrastive_loss = torch.relu(off_diag_sim - 0.5) * 2.0
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)
                
                # VARIANCE LOSS - embeddings should have variance
                emb_var = word_emb.var(dim=0).mean()
                variance_loss = torch.relu(0.1 - emb_var) * 10.0  # Penalize low variance
                
                # Brain-modulated loss
                brain = model.compute_brain_loss(outputs, char_indices)
                
                # Novelty bonus (explore new words)
                if 'novelty' in outputs['brain_signals']:
                    novelty_bonus = outputs['brain_signals']['novelty'].mean() * 0.1
                else:
                    novelty_bonus = 0
                
                # Total loss (modulated by dopamine) + collapse prevention
                base_loss = char_loss + 0.5 * word_loss
                collapse_loss = contrastive_loss + variance_loss
                loss = base_loss * (1 + brain['plasticity_scale'] * 0.5) + collapse_loss - novelty_bonus
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            total_reward += brain['reward'].item()
            num_batches += 1
            
            # Store words in vocabulary memory periodically
            if num_batches % 50 == 0:
                with torch.no_grad():
                    word_emb_sample = outputs['word_embeddings'][:8].mean(dim=1)
                    model.store_vocabulary(word_emb_sample.detach())
            
            if num_batches % 20 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'rwd': f'{total_reward/num_batches:.3f}',
                    'ctr': f'{contrastive_loss.item():.3f}',
                    'var': f'{emb_var.item():.3f}'
                })
        
        avg_loss = total_loss / num_batches
        avg_reward = total_reward / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Reward={avg_reward:.4f}")
        print(f"  Vocabulary size: {model.vocabulary.hippocampus.count}")
        print(f"  Hebbian trace norm: {model.syllable_word_trace.norm():.4f}")
        
        test_generation(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hierarchical_brain_best.pth")
            print(f"💾 Saved best model")


def test_generation(model, device):
    model.eval()
    prompts = ["Die Katze", "Der Junge", "Heute ist"]
    
    print("\n📊 Generation Test:")
    with torch.no_grad():
        for prompt in prompts:
            chars = text_to_indices(prompt.lower(), 64)
            indices = torch.tensor([chars], device=device)
            outputs = model(indices, train_brain=False)
            
            # Get word embeddings
            word_emb = outputs['word_embeddings'][0, :len(prompt)//3+1]
            
            # Retrieve similar words from vocabulary
            if model.vocabulary.hippocampus.count > 0:
                similar = model.vocabulary.retrieve_similar(word_emb.mean(dim=0, keepdim=True))
                sim_score = F.cosine_similarity(word_emb.mean(dim=0, keepdim=True), similar).item()
                print(f"  '{prompt}' → vocab similarity: {sim_score:.4f}")
            else:
                print(f"  '{prompt}' → (vocabulary empty)")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Vocab: {VOCAB_SIZE}")
    
    model = HierarchicalGermanBrain(
        vocab_size=VOCAB_SIZE, d_char=128, d_morph=256,
        d_word=512, d_phrase=512, d_sent=768, max_len=64
    ).to(device)
    
    # Load previous weights if available
    v4_path = "checkpoints/hierarchical_v4_best.pth"
    if os.path.exists(v4_path):
        print(f"Loading weights from {v4_path}")
        state = torch.load(v4_path, map_location=device)
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
    
    sentences = load_sentences(30000)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING HIERARCHICAL MODEL WITH BRAIN INTEGRATION")
    print("  - Dopamine: reward-based learning")
    print("  - Hebbian: syllable→word associations")
    print("  - Hippocampus: vocabulary memory")
    print("="*60 + "\n")
    
    train_brain_model(model, sentences, device, epochs=20, batch_size=16, lr=2e-4)
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
