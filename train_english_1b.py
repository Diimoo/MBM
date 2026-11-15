#!/usr/bin/env python3
"""
Train 1B Parameter Hierarchical English Model.

Scaled up dimensions for ~1 billion parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from tqdm import tqdm
import random
import math

# =============================================================================
# VOCABULARY (same as base model)
# =============================================================================

CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'-\"()")
PAD_TOKEN = 0
UNK_TOKEN = 1
SPECIAL_TOKENS = ["<PAD>", "<UNK>"]

char_to_idx = {c: i + len(SPECIAL_TOKENS) for i, c in enumerate(CHARS)}
char_to_idx["<PAD>"] = PAD_TOKEN
char_to_idx["<UNK>"] = UNK_TOKEN
idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

def text_to_indices(text, max_len=256):
    indices = []
    for c in text[:max_len]:
        indices.append(char_to_idx.get(c, UNK_TOKEN))
    while len(indices) < max_len:
        indices.append(PAD_TOKEN)
    return indices

def indices_to_text(indices):
    chars = []
    for idx in indices:
        if idx == PAD_TOKEN:
            break
        chars.append(idx_to_char.get(idx, "?"))
    return "".join(chars)

# =============================================================================
# CONFIGURATION - 1B PARAMETERS
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Scaled dimensions for ~600M parameters (uses ~12GB of 16GB GPU)
# Original: d_char=128, d_syllable=256, d_morpheme=256, d_word=512, d_phrase=512, d_sentence=768

D_CHAR = 448          # 128 -> 448 (3.5x)
D_SYLLABLE = 896      # 256 -> 896 (3.5x)  
D_MORPHEME = 896      # 256 -> 896 (3.5x)
D_WORD = 1280         # 512 -> 1280 (2.5x)
D_PHRASE = 1280       # 512 -> 1280 (2.5x)
D_SENTENCE = 1792     # 768 -> 1792 (2.3x)
D_DECODER = 1280      # 512 -> 1280 (2.5x)

# Deeper model
NUM_PHRASE_LAYERS = 4   # was 2
NUM_SENTENCE_LAYERS = 5  # was 3
NUM_DECODER_LAYERS = 6   # was 4

# Training config
BATCH_SIZE = 4
NUM_EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
MAX_LEN = 128
NUM_SENTENCES = 200000

# Loss weights
CHAR_LOSS_WEIGHT = 1.0
BOUNDARY_LOSS_WEIGHT = 1.0
DECODER_LOSS_WEIGHT = 2.0


# =============================================================================
# MODEL COMPONENTS (1B Scale)
# =============================================================================

class CharacterEncoder1B(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=D_CHAR, max_len=MAX_LEN):
        super().__init__()
        self.d_char = d_char
        self.max_len = max_len
        
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=PAD_TOKEN)
        self.pos_embed = nn.Embedding(max_len, d_char)
        
        # Deeper CNN
        self.local_context = nn.Sequential(
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        self.norm = nn.LayerNorm(d_char)
    
    def forward(self, char_indices):
        B, L = char_indices.shape
        x = self.char_embed(char_indices)
        positions = torch.arange(L, device=char_indices.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        x = x.transpose(1, 2)
        x = x + self.local_context(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class SyllableDetector1B(nn.Module):
    def __init__(self, d_char=D_CHAR, d_syllable=D_SYLLABLE, max_syllables=64):
        super().__init__()
        self.d_syllable = d_syllable
        self.max_syllables = max_syllables
        
        self.boundary_detector = nn.Sequential(
            nn.Linear(d_char * 3, d_char),
            nn.GELU(),
            nn.Linear(d_char, d_char),
            nn.GELU(),
            nn.Linear(d_char, 1),
        )
        
        self.syllable_lstm = nn.LSTM(
            d_char, d_syllable // 2,
            batch_first=True, bidirectional=True, num_layers=2
        )
        
        self.syllable_proj = nn.Linear(d_syllable, d_syllable)
        self.norm = nn.LayerNorm(d_syllable)
    
    def forward(self, char_embeddings, char_indices=None):
        B, L, D = char_embeddings.shape
        device = char_embeddings.device
        
        padded = F.pad(char_embeddings, (0, 0, 1, 1), mode='constant', value=0)
        prev_ctx = padded[:, :-2, :]
        curr_ctx = padded[:, 1:-1, :]
        next_ctx = padded[:, 2:, :]
        context = torch.cat([prev_ctx, curr_ctx, next_ctx], dim=-1)
        
        boundary_logits = self.boundary_detector(context).squeeze(-1)
        
        if self.training:
            boundary_probs = torch.sigmoid(boundary_logits)
        else:
            boundary_probs = (torch.sigmoid(boundary_logits) > 0.5).float()
        
        boundary_probs = boundary_probs.clone()
        boundary_probs[:, 0] = 1.0
        
        lstm_out, _ = self.syllable_lstm(char_embeddings)
        
        # Simplified pooling for efficiency
        syllable_embeds = lstm_out[:, :self.max_syllables, :]
        if syllable_embeds.shape[1] < self.max_syllables:
            padding = torch.zeros(B, self.max_syllables - syllable_embeds.shape[1], 
                                  self.d_syllable, device=device)
            syllable_embeds = torch.cat([syllable_embeds, padding], dim=1)
        
        syllable_embeds = self.syllable_proj(syllable_embeds)
        syllable_embeds = self.norm(syllable_embeds)
        
        return syllable_embeds, boundary_logits


class MorphemeParser1B(nn.Module):
    def __init__(self, d_syllable=D_SYLLABLE, d_morpheme=D_MORPHEME, max_morphemes=32):
        super().__init__()
        self.d_morpheme = d_morpheme
        self.max_morphemes = max_morphemes
        
        self.type_classifier = nn.Sequential(
            nn.Linear(d_syllable, d_morpheme),
            nn.GELU(),
            nn.Linear(d_morpheme, 3),
        )
        
        self.morpheme_lstm = nn.LSTM(
            d_syllable, d_morpheme // 2,
            batch_first=True, bidirectional=True, num_layers=2
        )
        
        self.norm = nn.LayerNorm(d_morpheme)
    
    def forward(self, syllable_embeddings):
        B, S, D = syllable_embeddings.shape
        device = syllable_embeddings.device
        
        lstm_out, _ = self.morpheme_lstm(syllable_embeddings)
        morpheme_embeds = lstm_out[:, :self.max_morphemes, :]
        
        if morpheme_embeds.shape[1] < self.max_morphemes:
            padding = torch.zeros(B, self.max_morphemes - morpheme_embeds.shape[1], 
                                  self.d_morpheme, device=device)
            morpheme_embeds = torch.cat([morpheme_embeds, padding], dim=1)
        
        morpheme_embeds = self.norm(morpheme_embeds)
        morpheme_types = self.type_classifier(morpheme_embeds)
        
        return morpheme_embeds, morpheme_types


class WordComposer1B(nn.Module):
    def __init__(self, d_morpheme=D_MORPHEME, d_word=D_WORD, max_words=32):
        super().__init__()
        self.d_word = d_word
        self.max_words = max_words
        
        self.word_attention = nn.MultiheadAttention(d_morpheme, num_heads=8, batch_first=True)
        self.word_proj = nn.Linear(d_morpheme, d_word)
        self.norm = nn.LayerNorm(d_word)
    
    def forward(self, morpheme_embeddings, morpheme_types=None):
        B, M, D = morpheme_embeddings.shape
        
        attended, _ = self.word_attention(morpheme_embeddings, morpheme_embeddings, morpheme_embeddings)
        word_embeds = self.word_proj(attended)
        word_embeds = word_embeds[:, :self.max_words, :]
        
        if word_embeds.shape[1] < self.max_words:
            padding = torch.zeros(B, self.max_words - word_embeds.shape[1], 
                                  self.d_word, device=word_embeds.device)
            word_embeds = torch.cat([word_embeds, padding], dim=1)
        
        word_embeds = self.norm(word_embeds)
        return word_embeds


class PhraseChunker1B(nn.Module):
    NUM_PHRASE_TYPES = 6
    
    def __init__(self, d_word=D_WORD, d_phrase=D_PHRASE, max_phrases=16, num_heads=16, num_layers=NUM_PHRASE_LAYERS):
        super().__init__()
        self.d_phrase = d_phrase
        self.max_phrases = max_phrases
        
        self.boundary_detector = nn.Sequential(
            nn.Linear(d_word * 2, d_phrase),
            nn.GELU(),
            nn.Linear(d_phrase, 1),
        )
        
        self.type_classifier = nn.Sequential(
            nn.Linear(d_phrase, d_phrase),
            nn.GELU(),
            nn.Linear(d_phrase, self.NUM_PHRASE_TYPES),
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_word, nhead=num_heads, dim_feedforward=d_phrase * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.phrase_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.phrase_proj = nn.Linear(d_word, d_phrase)
        self.norm = nn.LayerNorm(d_phrase)
    
    def forward(self, word_embeddings):
        B, W, D = word_embeddings.shape
        device = word_embeddings.device
        
        padded = F.pad(word_embeddings, (0, 0, 0, 1), mode='constant', value=0)
        curr = padded[:, :-1, :]
        next_word = padded[:, 1:, :]
        context = torch.cat([curr, next_word], dim=-1)
        
        boundary_logits = self.boundary_detector(context).squeeze(-1)
        
        encoded = self.phrase_encoder(word_embeddings)
        phrase_embeds = self.phrase_proj(encoded)
        phrase_embeds = phrase_embeds[:, :self.max_phrases, :]
        
        if phrase_embeds.shape[1] < self.max_phrases:
            padding = torch.zeros(B, self.max_phrases - phrase_embeds.shape[1], 
                                  self.d_phrase, device=device)
            phrase_embeds = torch.cat([phrase_embeds, padding], dim=1)
        
        phrase_embeds = self.norm(phrase_embeds)
        phrase_types = self.type_classifier(phrase_embeds)
        
        return phrase_embeds, phrase_types, boundary_logits


class SentenceEncoder1B(nn.Module):
    def __init__(self, d_phrase=D_PHRASE, d_sentence=D_SENTENCE, num_heads=16, num_layers=NUM_SENTENCE_LAYERS):
        super().__init__()
        self.d_sentence = d_sentence
        
        self.phrase_proj = nn.Linear(d_phrase, d_sentence)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_sentence, nhead=num_heads, dim_feedforward=d_sentence * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.sentence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_sentence) * 0.02)
        
        self.sentence_proj = nn.Sequential(
            nn.Linear(d_sentence, d_sentence),
            nn.GELU(),
            nn.Linear(d_sentence, d_sentence),
        )
        self.norm = nn.LayerNorm(d_sentence)
    
    def forward(self, phrase_embeddings, phrase_types=None):
        B, P, D = phrase_embeddings.shape
        
        phrase_proj = self.phrase_proj(phrase_embeddings)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        sequence = torch.cat([cls_tokens, phrase_proj], dim=1)
        
        encoded = self.sentence_encoder(sequence)
        
        sentence_repr = encoded[:, 0, :]
        phrase_attended = encoded[:, 1:, :]
        
        sentence_embedding = self.sentence_proj(sentence_repr)
        sentence_embedding = self.norm(sentence_embedding)
        
        return sentence_embedding, phrase_attended


class DecoderLayer1B(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, context, causal_mask=None):
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + x
        
        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(x, context, context)
        x = residual + x
        
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x


class TextDecoder1B(nn.Module):
    def __init__(self, d_sentence=D_SENTENCE, d_phrase=D_PHRASE, d_word=D_WORD, 
                 d_decoder=D_DECODER, num_heads=16, num_layers=NUM_DECODER_LAYERS, 
                 max_len=MAX_LEN, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_decoder = d_decoder
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        self.token_embed = nn.Embedding(vocab_size, d_decoder, padding_idx=PAD_TOKEN)
        self.pos_embed = nn.Embedding(max_len, d_decoder)
        
        self.sentence_proj = nn.Linear(d_sentence, d_decoder)
        self.phrase_proj = nn.Linear(d_phrase, d_decoder)
        self.word_proj = nn.Linear(d_word, d_decoder)
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer1B(d_decoder, num_heads) for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_decoder),
            nn.Linear(d_decoder, vocab_size),
        )
        
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len, device):
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self.causal_mask = mask.bool()
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, target_indices, sentence_embed, phrase_embeds=None, word_embeds=None):
        B, L = target_indices.shape
        device = target_indices.device
        
        x = self.token_embed(target_indices)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        
        sentence_ctx = self.sentence_proj(sentence_embed).unsqueeze(1)
        context = sentence_ctx
        if phrase_embeds is not None:
            phrase_ctx = self.phrase_proj(phrase_embeds)
            context = torch.cat([context, phrase_ctx], dim=1)
        if word_embeds is not None:
            word_ctx = self.word_proj(word_embeds)
            context = torch.cat([context, word_ctx], dim=1)
        
        causal_mask = self._get_causal_mask(L, device)
        
        for layer in self.decoder_layers:
            x = layer(x, context, causal_mask)
        
        logits = self.output_proj(x)
        return logits
    
    def generate(self, sentence_embed, phrase_embeds=None, word_embeds=None, 
                 max_len=None, temperature=1.0, top_k=50):
        if max_len is None:
            max_len = self.max_len
        max_len = min(max_len, self.max_len - 1)
        
        B = sentence_embed.size(0)
        device = sentence_embed.device
        
        generated = torch.full((B, 1), char_to_idx.get(' ', UNK_TOKEN), 
                               dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            input_seq = generated[:, -self.max_len:]
            logits = self.forward(input_seq, sentence_embed, phrase_embeds, word_embeds)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                top_k_clamped = min(top_k, next_logits.size(-1))
                values, indices = next_logits.topk(top_k_clamped)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            next_token = next_token.clamp(0, self.vocab_size - 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token == PAD_TOKEN).all():
                break
        
        return generated


class HierarchicalEnglishModel1B(nn.Module):
    """1 Billion Parameter Hierarchical English Model."""
    
    def __init__(self):
        super().__init__()
        self.max_len = MAX_LEN
        self.d_sentence = D_SENTENCE
        
        # Hierarchical levels
        self.char_encoder = CharacterEncoder1B()
        self.syllable_detector = SyllableDetector1B()
        self.morpheme_parser = MorphemeParser1B()
        self.word_composer = WordComposer1B()
        self.phrase_chunker = PhraseChunker1B()
        self.sentence_encoder = SentenceEncoder1B()
        
        # Text decoder
        self.text_decoder = TextDecoder1B()
        
        # Reconstruction decoders
        self.char_decoder = nn.Linear(D_CHAR, VOCAB_SIZE)
        
    def forward(self, char_indices, return_all_levels=False, target_indices=None):
        # Level 0: Characters
        char_embeds = self.char_encoder(char_indices)
        
        # Level 1: Syllables
        syllable_embeds, syllable_boundaries = self.syllable_detector(char_embeds, char_indices)
        
        # Level 2: Morphemes
        morpheme_embeds, morpheme_types = self.morpheme_parser(syllable_embeds)
        
        # Level 3: Words
        word_embeds = self.word_composer(morpheme_embeds, morpheme_types)
        
        # Level 4: Phrases
        phrase_embeds, phrase_types, phrase_boundaries = self.phrase_chunker(word_embeds)
        
        # Level 5: Sentence
        sentence_embed, phrase_attended = self.sentence_encoder(phrase_embeds, phrase_types)
        
        # Character reconstruction
        char_recon = self.char_decoder(char_embeds)
        
        # Decoder output
        decoder_logits = None
        if target_indices is not None:
            decoder_logits = self.text_decoder(
                target_indices, sentence_embed, phrase_embeds, word_embeds
            )
        
        if return_all_levels:
            return {
                "char_embeds": char_embeds,
                "char_recon": char_recon,
                "syllable_embeds": syllable_embeds,
                "syllable_boundaries": syllable_boundaries,
                "morpheme_embeds": morpheme_embeds,
                "morpheme_types": morpheme_types,
                "word_embeds": word_embeds,
                "phrase_embeds": phrase_embeds,
                "phrase_types": phrase_types,
                "phrase_boundaries": phrase_boundaries,
                "sentence_embed": sentence_embed,
                "decoder_logits": decoder_logits,
            }
        
        return char_recon, syllable_boundaries, decoder_logits
    
    def generate(self, text_or_embedding, max_len=100, temperature=0.8, top_k=50):
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            if isinstance(text_or_embedding, str):
                indices = torch.tensor([text_to_indices(text_or_embedding, self.max_len)], device=device)
                outputs = self.forward(indices, return_all_levels=True)
                sentence_embed = outputs["sentence_embed"]
                phrase_embeds = outputs["phrase_embeds"]
                word_embeds = outputs["word_embeds"]
            else:
                sentence_embed = text_or_embedding
                phrase_embeds = None
                word_embeds = None
            
            generated = self.text_decoder.generate(
                sentence_embed, phrase_embeds, word_embeds,
                max_len=max_len, temperature=temperature, top_k=top_k
            )
            
            text = indices_to_text(generated[0].cpu().tolist())
            return text


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# DATASET
# =============================================================================

class EnglishTextDataset(Dataset):
    def __init__(self, sentences, max_len=MAX_LEN):
        self.sentences = sentences
        self.max_len = max_len
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        text = self.sentences[idx]
        indices = text_to_indices(text, self.max_len)
        indices = torch.tensor(indices, dtype=torch.long)
        
        # Heuristic syllable boundaries
        vowels = set("aeiouAEIOU")
        boundaries = [0.0] * self.max_len
        boundaries[0] = 1.0
        
        text = text[:self.max_len]
        in_word = False
        
        for i, c in enumerate(text):
            if c.isalpha():
                if not in_word:
                    boundaries[i] = 1.0
                    in_word = True
            else:
                in_word = False
                if c == ' ':
                    boundaries[i] = 1.0
        
        boundaries = torch.tensor(boundaries, dtype=torch.float)
        target = indices.clone()
        
        return {"indices": indices, "boundaries": boundaries, "target": target}


def load_dataset(num_sentences=NUM_SENTENCES):
    print("\n📥 Loading HuggingFace dataset...")
    
    try:
        from datasets import load_dataset
        
        ds = load_dataset(
            "alasdairforsythe/text-english-code-fiction-nonfiction",
            split="train",
            streaming=True
        )
        
        sentences = []
        for item in tqdm(ds, desc="Loading", total=num_sentences * 2):
            text = item.get("text", "")
            if not text:
                continue
            
            for sent in text.replace('\n', ' ').split('.'):
                sent = sent.strip()
                if 20 <= len(sent) <= MAX_LEN - 10:
                    sent = ' '.join(sent.split())
                    if sent and sent[0].isupper():
                        sentences.append(sent + '.')
                        if len(sentences) >= num_sentences:
                            break
            
            if len(sentences) >= num_sentences:
                break
        
        print(f"   ✅ Loaded {len(sentences):,} sentences")
        return sentences
        
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return []


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, dataloader, optimizer, scheduler, epoch, scaler):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    
    for batch in pbar:
        indices = batch["indices"].to(DEVICE)
        boundaries = batch["boundaries"].to(DEVICE)
        target = batch["target"].to(DEVICE)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(indices, return_all_levels=True, target_indices=target)
            
            char_recon = outputs["char_recon"]
            syllable_boundaries = outputs["syllable_boundaries"]
            decoder_logits = outputs["decoder_logits"]
            
            mask = (indices != PAD_TOKEN).float()
            
            char_loss = F.cross_entropy(
                char_recon.view(-1, VOCAB_SIZE),
                indices.view(-1),
                ignore_index=PAD_TOKEN
            )
            
            boundary_loss = F.binary_cross_entropy_with_logits(
                syllable_boundaries, boundaries, weight=mask, reduction='sum'
            ) / mask.sum().clamp(min=1)
            
            decoder_target = target[:, 1:]
            decoder_pred = decoder_logits[:, :-1, :]
            
            decoder_loss = F.cross_entropy(
                decoder_pred.reshape(-1, VOCAB_SIZE),
                decoder_target.reshape(-1),
                ignore_index=PAD_TOKEN
            )
            
            loss = (CHAR_LOSS_WEIGHT * char_loss + 
                    BOUNDARY_LOSS_WEIGHT * boundary_loss + 
                    DECODER_LOSS_WEIGHT * decoder_loss)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "dec": f"{decoder_loss.item():.4f}"})
    
    return total_loss / len(dataloader)


def main():
    print("=" * 70)
    print("Training 1B Parameter Hierarchical English Model")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    # Check GPU memory
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {gpu_mem:.1f} GB")
    
    # Create model
    print("\n🔧 Building 1B parameter model...")
    model = HierarchicalEnglishModel1B().to(DEVICE)
    
    total_params = count_parameters(model)
    print(f"\n📊 Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"   Char Encoder:      {count_parameters(model.char_encoder):,}")
    print(f"   Syllable Detector: {count_parameters(model.syllable_detector):,}")
    print(f"   Morpheme Parser:   {count_parameters(model.morpheme_parser):,}")
    print(f"   Word Composer:     {count_parameters(model.word_composer):,}")
    print(f"   Phrase Chunker:    {count_parameters(model.phrase_chunker):,}")
    print(f"   Sentence Encoder:  {count_parameters(model.sentence_encoder):,}")
    print(f"   Text Decoder:      {count_parameters(model.text_decoder):,}")
    
    # Load dataset
    sentences = load_dataset(NUM_SENTENCES)
    if not sentences:
        print("❌ Failed to load dataset")
        return
    
    random.shuffle(sentences)
    val_size = int(len(sentences) * 0.1)
    train_sentences = sentences[val_size:]
    val_sentences = sentences[:val_size]
    
    train_dataset = EnglishTextDataset(train_sentences)
    val_dataset = EnglishTextDataset(val_sentences)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"\nDataset: {len(train_sentences):,} train, {len(val_sentences):,} val")
    
    # Optimizer with gradient scaling for mixed precision
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)
    scaler = torch.cuda.amp.GradScaler()
    
    print("\n" + "=" * 70)
    print("Training (Mixed Precision)")
    print("=" * 70)
    
    best_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, epoch, scaler)
        
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}: train_loss={train_loss:.4f}")
        
        # Save checkpoint
        if train_loss < best_loss:
            best_loss = train_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "english_1b.pth")
            print(f"   💾 Saved checkpoint (loss={best_loss:.4f})")
        
        # Test generation every 2 epochs
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                test = "The quick brown fox jumps over the lazy dog."
                generated = model.generate(test, max_len=60, temperature=0.7)
                print(f"\n   📝 Generation test:")
                print(f"      Input:  '{test}'")
                print(f"      Output: '{generated}'")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
