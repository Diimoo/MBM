#!/usr/bin/env python3
"""
Hierarchical German Model with Brain Integration (v2)
Based on working morpheme-level model (no sentence collapse).

Brain integration at WORD level only:
  - Dopamine: reward for correct character prediction
  - Hebbian: strengthen char→morpheme associations
  - Hippocampus: word/morpheme memory
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

def text_to_indices(text, max_len=64):
    indices = [char_to_idx.get(c, 1) for c in text[:max_len]]
    while len(indices) < max_len:
        indices.append(0)
    return indices


# =============================================================================
# HIERARCHICAL LAYERS (Levels 0-2 only - proven to work)
# =============================================================================

class CharacterEncoder(nn.Module):
    def __init__(self, vocab_size, d_char=128, max_len=64):
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


class MorphemeParser(nn.Module):
    def __init__(self, d_char=128, d_morph=256):
        super().__init__()
        self.lstm = nn.LSTM(d_char, d_char, 2, batch_first=True, bidirectional=True, dropout=0.1)
        self.boundary = nn.Sequential(nn.Linear(d_char*2, d_char), nn.GELU(), nn.Linear(d_char, 1))
        self.types = nn.Sequential(nn.Linear(d_char*2, d_char), nn.GELU(), nn.Linear(d_char, 5))
        self.project = nn.Sequential(nn.Linear(d_char*2, d_morph), nn.LayerNorm(d_morph), nn.GELU())
        
    def forward(self, char_emb):
        lstm_out, _ = self.lstm(char_emb)
        return self.boundary(lstm_out).squeeze(-1), self.types(lstm_out), self.project(lstm_out)


# =============================================================================
# DOPAMINE SYSTEM
# =============================================================================

class DopamineSystem(nn.Module):
    def __init__(self, baseline_tau=0.99):
        super().__init__()
        self.baseline_tau = baseline_tau
        self.register_buffer('reward_baseline', torch.tensor(0.0))
        
    def compute_reward(self, predictions, targets, mask):
        pred_ids = predictions.argmax(dim=-1)
        correct = (pred_ids == targets).float() * mask
        accuracy = correct.sum() / (mask.sum() + 1e-8)
        return accuracy
    
    def forward(self, reward):
        with torch.no_grad():
            self.reward_baseline = self.baseline_tau * self.reward_baseline + (1 - self.baseline_tau) * reward
        return reward - self.reward_baseline


# =============================================================================
# BRAIN-INTEGRATED MODEL (Morpheme Level)
# =============================================================================

class HierarchicalBrainV2(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, d_morph=256, max_len=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_morph = d_morph
        
        # Hierarchical layers
        self.char_encoder = CharacterEncoder(vocab_size, d_char, max_len)
        self.morpheme_parser = MorphemeParser(d_char, d_morph)
        
        # Character prediction head
        self.char_predictor = nn.Sequential(
            nn.Linear(d_char, d_char*2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_char*2, vocab_size)
        )
        
        # Brain modules
        self.dopamine = DopamineSystem()
        self.plasticity = SynapticPlasticity(tau_e=10.0, learning_rate=1e-4)
        self.memory = Hippocampus(d_z=d_morph, capacity=5000)
        
        # Hebbian trace: char→morph associations
        self.register_buffer('char_morph_trace', torch.zeros(d_char, d_morph))
        
    def forward(self, char_indices, update_brain=True):
        # Level 0: Characters
        char_emb = self.char_encoder(char_indices)
        
        # Level 2: Morphemes (skip syllable for simplicity)
        morph_bnd, morph_types, morph_emb = self.morpheme_parser(char_emb)
        
        # Character prediction
        char_pred = self.char_predictor(char_emb)
        
        # Brain computations
        if update_brain and self.training:
            # Update Hebbian trace
            char_avg = char_emb.mean(dim=1)  # [B, d_char]
            morph_avg = morph_emb.mean(dim=1)  # [B, d_morph]
            self.char_morph_trace = self.plasticity.update_trace(
                self.char_morph_trace, char_avg, morph_avg
            )
            
            # Compute novelty
            novelty = self._compute_novelty(morph_emb)
        else:
            novelty = None
        
        return {
            'char_embeddings': char_emb,
            'morph_boundary_logits': morph_bnd,
            'morph_type_logits': morph_types,
            'morph_embeddings': morph_emb,
            'char_predictions': char_pred,
            'novelty': novelty
        }
    
    def _compute_novelty(self, morph_emb):
        if self.memory.count == 0:
            return torch.ones(morph_emb.shape[0], device=morph_emb.device)
        query = morph_emb.mean(dim=1)  # [B, d_morph]
        retrieved = self.memory.retrieve(query, topk=1)
        similarity = F.cosine_similarity(query, retrieved, dim=-1)
        return 1.0 - similarity.clamp(0, 1)
    
    def store_memory(self, morph_emb):
        # Store morpheme embeddings in hippocampus
        avg_emb = morph_emb.mean(dim=1).detach()
        self.memory.encode(avg_emb)


# =============================================================================
# TRAINING
# =============================================================================

def load_german_words(max_words=30000):
    print("Loading German words...")
    words = set()
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="Loading", total=max_words * 10):
            text = item.get('text', '')
            if isinstance(text, str):
                for word in re.findall(r'[A-Za-zäöüßÄÖÜ]+', text):
                    if 3 <= len(word) <= 20:
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


def train(model, words, device, epochs=20, batch_size=64, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = GradScaler()
    
    steps = len(words) // batch_size
    scheduler = OneCycleLR(optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=steps, pct_start=0.1)
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(words)
        
        total_loss = 0
        total_reward = 0
        total_novelty = 0
        num_batches = 0
        
        pbar = tqdm(range(0, len(words) - batch_size, batch_size), desc=f"Epoch {epoch}/{epochs}")
        
        for i in pbar:
            batch = words[i:i+batch_size]
            
            # Prepare data
            max_len = max(len(w) for w in batch) + 2
            max_len = min(max_len, 64)
            char_batch = [text_to_indices(w, max_len) for w in batch]
            char_indices = torch.tensor(char_batch, device=device)
            
            optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(char_indices, update_brain=True)
                
                # Character prediction loss
                char_targets = char_indices[:, 1:]
                char_preds = outputs['char_predictions'][:, :-1, :]
                mask = (char_targets != 0).float()
                
                char_loss = F.cross_entropy(
                    char_preds.reshape(-1, model.vocab_size),
                    char_targets.reshape(-1),
                    ignore_index=0, reduction='none'
                ).view(char_targets.shape)
                char_loss = (char_loss * mask).sum() / (mask.sum() + 1e-8)
                
                # Dopamine reward
                reward = model.dopamine.compute_reward(char_preds, char_targets, mask)
                dopamine = model.dopamine(reward)
                
                # Novelty bonus
                if outputs['novelty'] is not None:
                    novelty = outputs['novelty'].mean()
                    novelty_bonus = novelty * 0.05
                else:
                    novelty = torch.tensor(0.0)
                    novelty_bonus = 0
                
                # Modulate loss by dopamine
                loss = char_loss * (1 + dopamine.clamp(-0.5, 0.5)) - novelty_bonus
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Store in memory periodically
            if num_batches % 20 == 0:
                with torch.no_grad():
                    model.store_memory(outputs['morph_embeddings'][:16])
            
            total_loss += loss.item()
            total_reward += reward.item()
            total_novelty += novelty.item() if isinstance(novelty, torch.Tensor) else novelty
            num_batches += 1
            
            if num_batches % 50 == 0:
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'rwd': f'{total_reward/num_batches:.3f}',
                    'nov': f'{total_novelty/num_batches:.3f}'
                })
        
        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Reward={total_reward/num_batches:.4f}")
        print(f"  Memory size: {model.memory.count}")
        print(f"  Hebbian norm: {model.char_morph_trace.norm():.4f}")
        
        # Test embeddings
        test_embeddings(model, device)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/hierarchical_brain_v2_best.pth")
            print(f"💾 Saved best model")


def test_embeddings(model, device):
    model.eval()
    tests = ['katze', 'hund', 'baum', 'möglichkeit', 'unglaublich']
    embeddings = []
    
    with torch.no_grad():
        for text in tests:
            chars = text_to_indices(text, 32)
            indices = torch.tensor([chars], device=device)
            outputs = model(indices, update_brain=False)
            morph_emb = outputs['morph_embeddings'][0, :len(text)].mean(dim=0)
            embeddings.append(morph_emb)
    
    # Check a few similarities
    pairs = [(0, 1), (0, 3), (3, 4)]
    print("📊 Embedding similarities:")
    for i, j in pairs:
        sim = F.cosine_similarity(embeddings[i].unsqueeze(0), embeddings[j].unsqueeze(0)).item()
        print(f"  {tests[i]} vs {tests[j]}: {sim:.4f}")
    print()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}, Vocab: {VOCAB_SIZE}")
    
    model = HierarchicalBrainV2(
        vocab_size=VOCAB_SIZE, d_char=128, d_morph=256, max_len=64
    ).to(device)
    
    # Load v2 weights if available
    v2_path = "checkpoints/hierarchical_v2_best.pth"
    if os.path.exists(v2_path):
        print(f"Loading base weights from {v2_path}")
        state = torch.load(v2_path, map_location=device)
        model_state = model.state_dict()
        loaded = 0
        for k, v in state.items():
            # Map old keys to new keys
            new_k = k
            if k.startswith('char_encoder.char_embed'):
                new_k = k.replace('char_embed', 'embed')
            elif k.startswith('char_encoder.pos_embed'):
                new_k = k.replace('pos_embed', 'pos')
            elif k.startswith('char_encoder.local_context'):
                new_k = k.replace('local_context', 'conv')
            elif k.startswith('morpheme_parser.morph_lstm'):
                new_k = k.replace('morph_lstm', 'lstm')
            elif k.startswith('morpheme_parser.boundary_head'):
                new_k = k.replace('boundary_head', 'boundary')
            elif k.startswith('morpheme_parser.type_head'):
                new_k = k.replace('type_head', 'types')
            elif k.startswith('morpheme_parser.morph_project'):
                new_k = k.replace('morph_project', 'project')
            
            if new_k in model_state and model_state[new_k].shape == v.shape:
                model_state[new_k] = v
                loaded += 1
        model.load_state_dict(model_state, strict=False)
        print(f"Loaded {loaded} layers")
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,} ({params/1e6:.1f}M)")
    
    words = load_german_words(30000)
    os.makedirs("checkpoints", exist_ok=True)
    
    print("\n" + "="*60)
    print("TRAINING: Brain-Integrated Morpheme Model")
    print("="*60 + "\n")
    
    train(model, words, device, epochs=20, batch_size=64, lr=3e-4)
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
