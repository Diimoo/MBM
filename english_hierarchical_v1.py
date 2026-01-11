#!/usr/bin/env python3
"""
Hierarchical English Language Model - Tabula Rasa
Following the architecture from docs/hierarchical_language_architecture.md

Level 0: Character Encoder
Level 1: Syllable Detector  
Level 2: Morpheme Parser
Level 3: Word Composer
Level 4: Phrase Chunker
Level 5: Sentence Encoder
+ Text Decoder for generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# =============================================================================
# VOCABULARY
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

VOWELS = set("aeiouAEIOU")


def text_to_indices(text, max_len=256):
    """Convert text to character indices."""
    indices = []
    for c in text[:max_len]:
        indices.append(char_to_idx.get(c, UNK_TOKEN))
    # Pad
    while len(indices) < max_len:
        indices.append(PAD_TOKEN)
    return indices


def indices_to_text(indices):
    """Convert indices back to text."""
    chars = []
    for idx in indices:
        if idx == PAD_TOKEN:
            break
        chars.append(idx_to_char.get(idx, "?"))
    return "".join(chars)


# =============================================================================
# LEVEL 0: CHARACTER ENCODER
# =============================================================================

class CharacterEncoder(nn.Module):
    """
    Level 0: Learn character representations.
    Input: Raw text as character indices
    Output: Character embeddings with positional context
    """
    
    def __init__(self, vocab_size=VOCAB_SIZE, d_char=128, max_len=256):
        super().__init__()
        self.d_char = d_char
        self.max_len = max_len
        
        # Character embedding
        self.char_embed = nn.Embedding(vocab_size, d_char, padding_idx=PAD_TOKEN)
        
        # Positional encoding (learned)
        self.pos_embed = nn.Embedding(max_len, d_char)
        
        # Local context via 1D CNN (captures character n-grams)
        self.local_context = nn.Sequential(
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_char, d_char, kernel_size=3, padding=1),
            nn.GELU(),
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(d_char)
    
    def forward(self, char_indices):
        """
        Args:
            char_indices: [batch, seq_len] character indices
        Returns:
            char_embeddings: [batch, seq_len, d_char]
        """
        B, L = char_indices.shape
        
        # Embeddings
        x = self.char_embed(char_indices)  # [B, L, d_char]
        
        # Add positional encoding
        positions = torch.arange(L, device=char_indices.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        
        # Local context (CNN expects [B, C, L])
        x = x.transpose(1, 2)  # [B, d_char, L]
        x = x + self.local_context(x)  # Residual
        x = x.transpose(1, 2)  # [B, L, d_char]
        
        # Normalize
        x = self.norm(x)
        
        return x


# =============================================================================
# LEVEL 1: SYLLABLE DETECTOR
# =============================================================================

class SyllableDetector(nn.Module):
    """
    Level 1: Group characters into pronounceable units (syllables).
    
    English syllable rules:
    - Every syllable has exactly one vowel nucleus (a, e, i, o, u)
    - Consonant clusters follow sonority sequencing
    - Examples: "hap-py", "beau-ti-ful", "strength"
    
    Input: Character embeddings [batch, seq_len, d_char]
    Output: Syllable embeddings [batch, num_syllables, d_syllable]
    """
    
    def __init__(self, d_char=128, d_syllable=256, max_syllables=64):
        super().__init__()
        self.d_char = d_char
        self.d_syllable = d_syllable
        self.max_syllables = max_syllables
        
        # Boundary detection: predict if each character starts a new syllable
        self.boundary_detector = nn.Sequential(
            nn.Linear(d_char * 3, d_char),  # Context: prev, curr, next
            nn.GELU(),
            nn.Linear(d_char, d_char),
            nn.GELU(),
            nn.Linear(d_char, 1),
        )
        
        # Syllable composition: BiLSTM to aggregate chars within syllable
        self.syllable_lstm = nn.LSTM(
            d_char, d_syllable // 2,
            batch_first=True, bidirectional=True
        )
        
        # Syllable projection
        self.syllable_proj = nn.Linear(d_syllable, d_syllable)
        self.norm = nn.LayerNorm(d_syllable)
    
    def forward(self, char_embeddings, char_indices=None):
        """
        Args:
            char_embeddings: [batch, seq_len, d_char]
            char_indices: [batch, seq_len] optional, for masking
        Returns:
            syllable_embeddings: [batch, max_syllables, d_syllable]
            boundary_logits: [batch, seq_len] probability of boundary at each position
        """
        B, L, D = char_embeddings.shape
        device = char_embeddings.device
        
        # Create context windows for boundary detection
        # Pad for boundary detection
        padded = F.pad(char_embeddings, (0, 0, 1, 1), mode='constant', value=0)
        prev_ctx = padded[:, :-2, :]  # [B, L, D]
        curr_ctx = padded[:, 1:-1, :]  # [B, L, D]
        next_ctx = padded[:, 2:, :]  # [B, L, D]
        
        context = torch.cat([prev_ctx, curr_ctx, next_ctx], dim=-1)  # [B, L, 3*D]
        
        # Predict boundaries
        boundary_logits = self.boundary_detector(context).squeeze(-1)  # [B, L]
        
        # Use hard boundaries during inference, soft during training
        if self.training:
            boundary_probs = torch.sigmoid(boundary_logits)
        else:
            boundary_probs = (torch.sigmoid(boundary_logits) > 0.5).float()
        
        # First position is always a boundary
        boundary_probs = boundary_probs.clone()
        boundary_probs[:, 0] = 1.0
        
        # Aggregate characters into syllables using attention-weighted pooling
        # For simplicity, use LSTM over all chars and sample at boundaries
        lstm_out, _ = self.syllable_lstm(char_embeddings)  # [B, L, d_syllable]
        
        # Create syllable embeddings by pooling between boundaries
        syllable_embeds = []
        for b in range(B):
            # Find boundary positions
            if char_indices is not None:
                # Mask out padding
                mask = (char_indices[b] != PAD_TOKEN).float()
                boundaries = (boundary_probs[b] * mask > 0.5).nonzero(as_tuple=True)[0]
            else:
                boundaries = (boundary_probs[b] > 0.5).nonzero(as_tuple=True)[0]
            
            if len(boundaries) == 0:
                boundaries = torch.tensor([0], device=device)
            
            # Pool LSTM outputs between boundaries
            batch_syllables = []
            for i in range(len(boundaries)):
                start = boundaries[i].item()
                end = boundaries[i + 1].item() if i + 1 < len(boundaries) else L
                
                # Skip empty or padding regions
                if char_indices is not None:
                    valid = (char_indices[b, start:end] != PAD_TOKEN).any()
                    if not valid:
                        continue
                
                # Mean pool the LSTM outputs for this syllable
                syllable_repr = lstm_out[b, start:end].mean(dim=0)
                batch_syllables.append(syllable_repr)
            
            if len(batch_syllables) == 0:
                batch_syllables.append(torch.zeros(self.d_syllable, device=device))
            
            # Pad/truncate to max_syllables
            while len(batch_syllables) < self.max_syllables:
                batch_syllables.append(torch.zeros(self.d_syllable, device=device))
            batch_syllables = batch_syllables[:self.max_syllables]
            
            syllable_embeds.append(torch.stack(batch_syllables))
        
        syllable_embeds = torch.stack(syllable_embeds)  # [B, max_syllables, d_syllable]
        
        # Project and normalize
        syllable_embeds = self.syllable_proj(syllable_embeds)
        syllable_embeds = self.norm(syllable_embeds)
        
        return syllable_embeds, boundary_logits


# =============================================================================
# LEVEL 2: MORPHEME PARSER
# =============================================================================

class MorphemeParser(nn.Module):
    """
    Level 2: Identify meaningful sub-word units (roots, prefixes, suffixes).
    
    English morphology:
    - Prefixes: un-, re-, pre-, dis-, mis-, over-, under-
    - Suffixes: -ing, -ed, -er, -est, -ly, -ness, -ment, -tion
    - Roots: The semantic core
    
    Examples:
    - "unhappy" → [un- (negation), happy (root)]
    - "happiness" → [happy (root), -ness (noun)]
    
    Input: Syllable embeddings [batch, num_syllables, d_syllable]
    Output: Morpheme embeddings [batch, num_morphemes, d_morpheme]
    """
    
    def __init__(self, d_syllable=256, d_morpheme=256, max_morphemes=32):
        super().__init__()
        self.d_morpheme = d_morpheme
        self.max_morphemes = max_morphemes
        
        # Morpheme type classification: ROOT, PREFIX, SUFFIX
        self.type_classifier = nn.Sequential(
            nn.Linear(d_syllable, d_morpheme),
            nn.GELU(),
            nn.Linear(d_morpheme, 3),  # 3 types
        )
        
        # Morpheme boundary detection
        self.boundary_detector = nn.Sequential(
            nn.Linear(d_syllable * 2, d_morpheme),
            nn.GELU(),
            nn.Linear(d_morpheme, 1),
        )
        
        # Morpheme composition
        self.morpheme_lstm = nn.LSTM(
            d_syllable, d_morpheme // 2,
            batch_first=True, bidirectional=True
        )
        
        self.norm = nn.LayerNorm(d_morpheme)
    
    def forward(self, syllable_embeddings):
        """
        Args:
            syllable_embeddings: [batch, num_syllables, d_syllable]
        Returns:
            morpheme_embeddings: [batch, max_morphemes, d_morpheme]
            morpheme_types: [batch, max_morphemes, 3] logits for ROOT/PREFIX/SUFFIX
        """
        B, S, D = syllable_embeddings.shape
        device = syllable_embeddings.device
        
        # Morpheme boundary detection
        padded = F.pad(syllable_embeddings, (0, 0, 0, 1), mode='constant', value=0)
        curr = padded[:, :-1, :]
        next_syl = padded[:, 1:, :]
        context = torch.cat([curr, next_syl], dim=-1)
        
        boundary_logits = self.boundary_detector(context).squeeze(-1)  # [B, S]
        boundary_probs = torch.sigmoid(boundary_logits)
        boundary_probs[:, 0] = 1.0  # First position is always a boundary
        
        # LSTM over syllables
        lstm_out, _ = self.morpheme_lstm(syllable_embeddings)  # [B, S, d_morpheme]
        
        # Pool into morphemes (simplified: use syllable boundaries as morpheme boundaries)
        morpheme_embeds = lstm_out[:, :self.max_morphemes, :]  # [B, max_morphemes, d_morpheme]
        
        # Pad if needed
        if morpheme_embeds.shape[1] < self.max_morphemes:
            padding = torch.zeros(B, self.max_morphemes - morpheme_embeds.shape[1], 
                                  self.d_morpheme, device=device)
            morpheme_embeds = torch.cat([morpheme_embeds, padding], dim=1)
        
        morpheme_embeds = self.norm(morpheme_embeds)
        
        # Classify morpheme types
        morpheme_types = self.type_classifier(morpheme_embeds)  # [B, max_morphemes, 3]
        
        return morpheme_embeds, morpheme_types


# =============================================================================
# LEVEL 3: WORD COMPOSER
# =============================================================================

class WordComposer(nn.Module):
    """
    Level 3: Compose morphemes into word-level meaning.
    
    Key insight: Word meaning = f(root_meaning, affix_modifications)
    
    Input: Morpheme embeddings + types
    Output: Word embeddings [batch, num_words, d_word]
    """
    
    def __init__(self, d_morpheme=256, d_word=512, max_words=32):
        super().__init__()
        self.d_word = d_word
        self.max_words = max_words
        
        # Word boundary detection (space-based for English)
        self.boundary_detector = nn.Sequential(
            nn.Linear(d_morpheme * 2, d_word),
            nn.GELU(),
            nn.Linear(d_word, 1),
        )
        
        # Word composition with attention
        self.word_attention = nn.MultiheadAttention(d_morpheme, num_heads=4, batch_first=True)
        self.word_proj = nn.Linear(d_morpheme, d_word)
        self.norm = nn.LayerNorm(d_word)
    
    def forward(self, morpheme_embeddings, morpheme_types=None):
        """
        Args:
            morpheme_embeddings: [batch, num_morphemes, d_morpheme]
            morpheme_types: [batch, num_morphemes, 3] optional
        Returns:
            word_embeddings: [batch, max_words, d_word]
        """
        B, M, D = morpheme_embeddings.shape
        
        # Self-attention over morphemes
        attended, _ = self.word_attention(morpheme_embeddings, morpheme_embeddings, morpheme_embeddings)
        
        # Project to word dimension
        word_embeds = self.word_proj(attended)  # [B, M, d_word]
        
        # Take first max_words (simplified word segmentation)
        word_embeds = word_embeds[:, :self.max_words, :]
        
        # Pad if needed
        if word_embeds.shape[1] < self.max_words:
            padding = torch.zeros(B, self.max_words - word_embeds.shape[1], 
                                  self.d_word, device=word_embeds.device)
            word_embeds = torch.cat([word_embeds, padding], dim=1)
        
        word_embeds = self.norm(word_embeds)
        
        return word_embeds


# =============================================================================
# LEVEL 4: PHRASE CHUNKER
# =============================================================================

class PhraseChunker(nn.Module):
    """
    Level 4: Group words into syntactic phrases (NP, VP, PP).
    
    English phrase types:
    - NP (Noun Phrase): "the big dog", "a beautiful sunset"
    - VP (Verb Phrase): "is running", "has been eating"
    - PP (Prepositional Phrase): "on the table", "under the bridge"
    - ADJP (Adjective Phrase): "very happy"
    - ADVP (Adverb Phrase): "very quickly"
    
    Input: Word embeddings [batch, num_words, d_word]
    Output: Phrase embeddings [batch, num_phrases, d_phrase]
    """
    
    NUM_PHRASE_TYPES = 6  # NP, VP, PP, ADJP, ADVP, OTHER
    
    def __init__(self, d_word=512, d_phrase=512, max_phrases=16, num_heads=8):
        super().__init__()
        self.d_phrase = d_phrase
        self.max_phrases = max_phrases
        
        # Phrase boundary detection
        self.boundary_detector = nn.Sequential(
            nn.Linear(d_word * 2, d_phrase),
            nn.GELU(),
            nn.Linear(d_phrase, 1),
        )
        
        # Phrase type classifier
        self.type_classifier = nn.Sequential(
            nn.Linear(d_phrase, d_phrase),
            nn.GELU(),
            nn.Linear(d_phrase, self.NUM_PHRASE_TYPES),
        )
        
        # Transformer encoder for phrase composition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_word, nhead=num_heads, dim_feedforward=d_phrase * 2,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.phrase_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Phrase projection
        self.phrase_proj = nn.Linear(d_word, d_phrase)
        self.norm = nn.LayerNorm(d_phrase)
    
    def forward(self, word_embeddings):
        """
        Args:
            word_embeddings: [batch, num_words, d_word]
        Returns:
            phrase_embeddings: [batch, max_phrases, d_phrase]
            phrase_types: [batch, max_phrases, NUM_PHRASE_TYPES]
            phrase_boundaries: [batch, num_words]
        """
        B, W, D = word_embeddings.shape
        device = word_embeddings.device
        
        # Phrase boundary detection
        padded = F.pad(word_embeddings, (0, 0, 0, 1), mode='constant', value=0)
        curr = padded[:, :-1, :]
        next_word = padded[:, 1:, :]
        context = torch.cat([curr, next_word], dim=-1)
        
        boundary_logits = self.boundary_detector(context).squeeze(-1)  # [B, W]
        boundary_probs = torch.sigmoid(boundary_logits)
        boundary_probs = boundary_probs.clone()
        boundary_probs[:, 0] = 1.0  # First word starts a phrase
        
        # Encode word sequence with transformer
        encoded = self.phrase_encoder(word_embeddings)  # [B, W, d_word]
        
        # Project to phrase dimension
        phrase_embeds = self.phrase_proj(encoded)  # [B, W, d_phrase]
        
        # Pool words into phrases based on boundaries
        phrase_outputs = []
        for b in range(B):
            boundaries = (boundary_probs[b] > 0.5).nonzero(as_tuple=True)[0]
            if len(boundaries) == 0:
                boundaries = torch.tensor([0], device=device)
            
            batch_phrases = []
            for i in range(len(boundaries)):
                start = boundaries[i].item()
                end = boundaries[i + 1].item() if i + 1 < len(boundaries) else W
                
                # Mean pool words in this phrase
                phrase_repr = phrase_embeds[b, start:end].mean(dim=0)
                batch_phrases.append(phrase_repr)
            
            # Pad to max_phrases
            while len(batch_phrases) < self.max_phrases:
                batch_phrases.append(torch.zeros(self.d_phrase, device=device))
            batch_phrases = batch_phrases[:self.max_phrases]
            phrase_outputs.append(torch.stack(batch_phrases))
        
        phrase_embeddings = torch.stack(phrase_outputs)  # [B, max_phrases, d_phrase]
        phrase_embeddings = self.norm(phrase_embeddings)
        
        # Classify phrase types
        phrase_types = self.type_classifier(phrase_embeddings)
        
        return phrase_embeddings, phrase_types, boundary_logits


# =============================================================================
# LEVEL 5: SENTENCE ENCODER
# =============================================================================

class SentenceEncoder(nn.Module):
    """
    Level 5: Compose phrases into full sentence meaning.
    
    Captures:
    - Subject-Verb-Object structure
    - Clause relationships
    - Sentence-level semantics
    
    Input: Phrase embeddings [batch, num_phrases, d_phrase]
    Output: Sentence embedding [batch, d_sentence]
    """
    
    def __init__(self, d_phrase=512, d_sentence=768, num_heads=8, num_layers=3):
        super().__init__()
        self.d_sentence = d_sentence
        
        # Project phrases to sentence dimension
        self.phrase_proj = nn.Linear(d_phrase, d_sentence)
        
        # Transformer for sentence composition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_sentence, nhead=num_heads, dim_feedforward=d_sentence * 4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.sentence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # [CLS] token for sentence representation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_sentence) * 0.02)
        
        # Final projection
        self.sentence_proj = nn.Sequential(
            nn.Linear(d_sentence, d_sentence),
            nn.GELU(),
            nn.Linear(d_sentence, d_sentence),
        )
        self.norm = nn.LayerNorm(d_sentence)
    
    def forward(self, phrase_embeddings, phrase_types=None):
        """
        Args:
            phrase_embeddings: [batch, num_phrases, d_phrase]
            phrase_types: [batch, num_phrases, NUM_TYPES] optional
        Returns:
            sentence_embedding: [batch, d_sentence]
            phrase_attended: [batch, num_phrases, d_sentence]
        """
        B, P, D = phrase_embeddings.shape
        
        # Project to sentence dimension
        phrase_proj = self.phrase_proj(phrase_embeddings)  # [B, P, d_sentence]
        
        # Add CLS token at the beginning
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, d_sentence]
        sequence = torch.cat([cls_tokens, phrase_proj], dim=1)  # [B, P+1, d_sentence]
        
        # Encode with transformer
        encoded = self.sentence_encoder(sequence)  # [B, P+1, d_sentence]
        
        # Extract sentence representation from CLS token
        sentence_repr = encoded[:, 0, :]  # [B, d_sentence]
        phrase_attended = encoded[:, 1:, :]  # [B, P, d_sentence]
        
        # Final projection
        sentence_embedding = self.sentence_proj(sentence_repr)
        sentence_embedding = self.norm(sentence_embedding)
        
        return sentence_embedding, phrase_attended


# =============================================================================
# TEXT DECODER (for generation)
# =============================================================================

class TextDecoder(nn.Module):
    """
    Autoregressive decoder that generates text from hierarchical representations.
    
    Takes sentence embedding + phrase context and generates character-by-character.
    Uses cross-attention to attend to hierarchical representations.
    """
    
    def __init__(self, d_sentence=768, d_phrase=512, d_word=512, d_decoder=512, 
                 num_heads=8, num_layers=4, max_len=256, vocab_size=VOCAB_SIZE):
        super().__init__()
        self.d_decoder = d_decoder
        self.max_len = max_len
        self.vocab_size = vocab_size
        
        # Token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_decoder, padding_idx=PAD_TOKEN)
        self.pos_embed = nn.Embedding(max_len, d_decoder)
        
        # Project hierarchical representations to decoder dimension
        self.sentence_proj = nn.Linear(d_sentence, d_decoder)
        self.phrase_proj = nn.Linear(d_phrase, d_decoder)
        self.word_proj = nn.Linear(d_word, d_decoder)
        
        # Decoder layers with cross-attention
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_decoder, num_heads) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_decoder),
            nn.Linear(d_decoder, vocab_size),
        )
        
        # Causal mask cache
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len, device):
        """Create causal attention mask."""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            self.causal_mask = mask.bool()
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, target_indices, sentence_embed, phrase_embeds=None, word_embeds=None):
        """
        Args:
            target_indices: [batch, seq_len] target character indices (teacher forcing)
            sentence_embed: [batch, d_sentence] sentence representation
            phrase_embeds: [batch, num_phrases, d_phrase] optional
            word_embeds: [batch, num_words, d_word] optional
        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        B, L = target_indices.shape
        device = target_indices.device
        
        # Token embeddings with positions
        x = self.token_embed(target_indices)
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_embed(positions)
        
        # Project context representations
        sentence_ctx = self.sentence_proj(sentence_embed).unsqueeze(1)  # [B, 1, d_decoder]
        
        # Build context sequence for cross-attention
        context = sentence_ctx
        if phrase_embeds is not None:
            phrase_ctx = self.phrase_proj(phrase_embeds)
            context = torch.cat([context, phrase_ctx], dim=1)
        if word_embeds is not None:
            word_ctx = self.word_proj(word_embeds)
            context = torch.cat([context, word_ctx], dim=1)
        
        # Causal mask for self-attention
        causal_mask = self._get_causal_mask(L, device)
        
        # Decode
        for layer in self.decoder_layers:
            x = layer(x, context, causal_mask)
        
        # Output logits
        logits = self.output_proj(x)
        
        return logits
    
    def generate(self, sentence_embed, phrase_embeds=None, word_embeds=None, 
                 max_len=None, temperature=1.0, top_k=50):
        """
        Autoregressive generation.
        """
        if max_len is None:
            max_len = self.max_len
        
        B = sentence_embed.size(0)
        device = sentence_embed.device
        
        # Start with a space token
        generated = torch.full((B, 1), char_to_idx.get(' ', UNK_TOKEN), 
                               dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.forward(generated, sentence_embed, phrase_embeds, word_embeds)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k > 0:
                values, indices = next_logits.topk(top_k)
                next_logits = torch.full_like(next_logits, float('-inf'))
                next_logits.scatter_(1, indices, values)
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated PAD
            if (next_token == PAD_TOKEN).all():
                break
        
        return generated


class DecoderLayer(nn.Module):
    """Single decoder layer with self-attention and cross-attention."""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_attn_norm = nn.LayerNorm(d_model)
        
        # Cross-attention to hierarchical context
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
    
    def forward(self, x, context, causal_mask=None):
        # Self-attention with causal mask
        residual = x
        x = self.self_attn_norm(x)
        x, _ = self.self_attn(x, x, x, attn_mask=causal_mask)
        x = residual + x
        
        # Cross-attention to context
        residual = x
        x = self.cross_attn_norm(x)
        x, _ = self.cross_attn(x, context, context)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.ffn_norm(x)
        x = residual + self.ffn(x)
        
        return x


# =============================================================================
# FULL HIERARCHICAL MODEL
# =============================================================================

class HierarchicalEnglishModel(nn.Module):
    """
    Full hierarchical model: Chars → Syllables → Morphemes → Words → Phrases → Sentences
    + Text Decoder for generation
    """
    
    def __init__(self, d_char=128, d_syllable=256, d_morpheme=256, d_word=512, 
                 d_phrase=512, d_sentence=768, max_len=256):
        super().__init__()
        self.max_len = max_len
        self.d_sentence = d_sentence
        
        # Hierarchical levels (encoder)
        self.char_encoder = CharacterEncoder(VOCAB_SIZE, d_char, max_len)
        self.syllable_detector = SyllableDetector(d_char, d_syllable)
        self.morpheme_parser = MorphemeParser(d_syllable, d_morpheme)
        self.word_composer = WordComposer(d_morpheme, d_word)
        self.phrase_chunker = PhraseChunker(d_word, d_phrase)
        self.sentence_encoder = SentenceEncoder(d_phrase, d_sentence)
        
        # Text decoder for generation
        self.text_decoder = TextDecoder(
            d_sentence=d_sentence, d_phrase=d_phrase, d_word=d_word,
            d_decoder=512, num_heads=8, num_layers=4, max_len=max_len
        )
        
        # Reconstruction decoders (for training each level)
        self.char_decoder = nn.Linear(d_char, VOCAB_SIZE)
        self.syllable_to_char = nn.Linear(d_syllable, d_char)
        
    def forward(self, char_indices, return_all_levels=False, target_indices=None):
        """
        Forward pass through all levels.
        
        Args:
            char_indices: [batch, seq_len] input character indices
            return_all_levels: if True, return dict with all level outputs
            target_indices: [batch, seq_len] optional, for decoder training
        """
        # Level 0: Characters
        char_embeds = self.char_encoder(char_indices)  # [B, L, d_char]
        
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
        
        # Character reconstruction (for training lower levels)
        char_recon = self.char_decoder(char_embeds)  # [B, L, vocab_size]
        
        # Decoder output (if target provided)
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
    
    def get_syllables(self, text):
        """
        Get syllable segmentation for a text string.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            indices = torch.tensor([text_to_indices(text, self.max_len)], device=device)
            char_embeds = self.char_encoder(indices)
            _, boundary_logits = self.syllable_detector(char_embeds, indices)
            
            boundaries = (torch.sigmoid(boundary_logits[0]) > 0.5).cpu().numpy()
            
            # Build syllable string
            syllables = []
            current = ""
            for i, c in enumerate(text):
                if i > 0 and boundaries[i]:
                    syllables.append(current)
                    current = ""
                current += c
            if current:
                syllables.append(current)
            
            return syllables
    
    def encode(self, text):
        """
        Encode text to sentence embedding.
        """
        self.eval()
        device = next(self.parameters()).device
        with torch.no_grad():
            indices = torch.tensor([text_to_indices(text, self.max_len)], device=device)
            outputs = self.forward(indices, return_all_levels=True)
            return outputs["sentence_embed"]
    
    def generate(self, text_or_embedding, max_len=100, temperature=0.8, top_k=50):
        """
        Generate text from an input text or sentence embedding.
        
        Args:
            text_or_embedding: Either a string or a sentence embedding tensor
            max_len: Maximum length to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: Top-k sampling parameter
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            if isinstance(text_or_embedding, str):
                # Encode the text first
                indices = torch.tensor([text_to_indices(text_or_embedding, self.max_len)], device=device)
                outputs = self.forward(indices, return_all_levels=True)
                sentence_embed = outputs["sentence_embed"]
                phrase_embeds = outputs["phrase_embeds"]
                word_embeds = outputs["word_embeds"]
            else:
                # Use provided embedding directly
                sentence_embed = text_or_embedding
                phrase_embeds = None
                word_embeds = None
            
            # Generate
            generated = self.text_decoder.generate(
                sentence_embed, phrase_embeds, word_embeds,
                max_len=max_len, temperature=temperature, top_k=top_k
            )
            
            # Convert to text
            text = indices_to_text(generated[0].cpu().tolist())
            return text


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("=" * 60)
    print("Hierarchical English Model - Full Architecture Test")
    print("=" * 60)
    
    model = HierarchicalEnglishModel()
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print(f"Vocabulary size: {VOCAB_SIZE}")
    
    # Test forward pass
    test_text = "The beautiful butterfly landed on the flower."
    indices = torch.tensor([text_to_indices(test_text)])
    
    outputs = model(indices, return_all_levels=True)
    
    print(f"\nInput: '{test_text}'")
    print(f"\n📊 Hierarchical Representations:")
    print(f"   Level 0 - Char embeddings:     {outputs['char_embeds'].shape}")
    print(f"   Level 1 - Syllable embeddings: {outputs['syllable_embeds'].shape}")
    print(f"   Level 2 - Morpheme embeddings: {outputs['morpheme_embeds'].shape}")
    print(f"   Level 3 - Word embeddings:     {outputs['word_embeds'].shape}")
    print(f"   Level 4 - Phrase embeddings:   {outputs['phrase_embeds'].shape}")
    print(f"   Level 5 - Sentence embedding:  {outputs['sentence_embed'].shape}")
    
    # Test syllable detection
    syllables = model.get_syllables(test_text)
    print(f"\n📝 Syllables: {'-'.join(syllables)}")
    
    # Test decoder (forward pass with target)
    target = indices.clone()
    outputs_with_decoder = model(indices, return_all_levels=True, target_indices=target)
    print(f"\n🔤 Decoder logits: {outputs_with_decoder['decoder_logits'].shape}")
    
    # Test generation (untrained, will be random)
    print(f"\n🎲 Generation test (untrained model - will be random):")
    generated = model.generate(test_text, max_len=50, temperature=1.0)
    print(f"   Generated: '{generated[:50]}...'")
    
    print("\n✅ Full architecture test passed!")
