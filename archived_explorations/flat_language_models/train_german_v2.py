#!/usr/bin/env python3
"""
German Brain Language Training v2
- Char-level encoding (not bytes)
- 4096 semantic dimension
- ~300M parameters
- No reconstruction loss
- Word boundary weighting (2.0x)
- Next-char prediction only ("learning by doing")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import OneCycleLR
from datasets import load_dataset
from tqdm import tqdm
import time
import json
import random
import os

# =============================================================================
# CHAR-LEVEL VOCABULARY
# =============================================================================

# German character vocabulary (comprehensive)
CHARS = (
    # Lowercase
    'abcdefghijklmnopqrstuvwxyz'
    # Uppercase  
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # German special
    'äöüßÄÖÜ'
    # Numbers
    '0123456789'
    # Punctuation & common symbols
    ' .,!?;:\'"()-–—/\\@#$%&*+=<>[]{}|~`^_'
    # Newlines and tabs
    '\n\t'
)

# Add padding and unknown tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# Build vocabulary
char_to_idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for i, c in enumerate(CHARS):
    if c not in char_to_idx:
        char_to_idx[c] = len(char_to_idx)

idx_to_char = {v: k for k, v in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

print(f"Vocabulary size: {VOCAB_SIZE} characters")

def text_to_tensor(text, max_len=128):
    """Convert text to char indices."""
    indices = []
    for c in text[:max_len]:
        indices.append(char_to_idx.get(c, char_to_idx[UNK_TOKEN]))
    # Pad if needed
    while len(indices) < max_len:
        indices.append(char_to_idx[PAD_TOKEN])
    return torch.tensor(indices, dtype=torch.long)

def tensor_to_text(tensor):
    """Convert tensor back to text."""
    chars = []
    for idx in tensor.cpu().numpy():
        if idx == char_to_idx[PAD_TOKEN]:
            break
        chars.append(idx_to_char.get(idx, '?'))
    return ''.join(chars)

def is_word_boundary(char_idx):
    """Check if character is a word boundary (space, punctuation)."""
    char = idx_to_char.get(char_idx, '')
    return char in ' .,!?;:\n\t-–—'

# =============================================================================
# LARGER MODEL ARCHITECTURE (~300M params)
# =============================================================================

class CharPatternDetector(nn.Module):
    """Detects character patterns at multiple scales."""
    def __init__(self, vocab_size, d_embed=512, d_pattern=1024):
        super().__init__()
        self.char_embed = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        
        # Multi-scale convolutions
        self.conv2 = nn.Conv1d(d_embed, d_pattern//4, kernel_size=2, padding=1)
        self.conv3 = nn.Conv1d(d_embed, d_pattern//4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(d_embed, d_pattern//4, kernel_size=4, padding=2)
        self.conv5 = nn.Conv1d(d_embed, d_pattern//4, kernel_size=5, padding=2)
        
        self.norm = nn.LayerNorm(d_pattern)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x: [batch, seq_len]
        embed = self.char_embed(x)  # [batch, seq_len, d_embed]
        embed = embed.transpose(1, 2)  # [batch, d_embed, seq_len]
        
        # Multi-scale patterns
        p2 = F.gelu(self.conv2(embed))[:, :, :x.size(1)]
        p3 = F.gelu(self.conv3(embed))[:, :, :x.size(1)]
        p4 = F.gelu(self.conv4(embed))[:, :, :x.size(1)]
        p5 = F.gelu(self.conv5(embed))[:, :, :x.size(1)]
        
        # Combine
        patterns = torch.cat([p2, p3, p4, p5], dim=1)  # [batch, d_pattern, seq_len]
        patterns = patterns.transpose(1, 2)  # [batch, seq_len, d_pattern]
        
        return self.dropout(self.norm(patterns))


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence understanding."""
    def __init__(self, d_model=1024, nhead=8, num_layers=6, d_ff=4096, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask=None):
        return self.norm(self.encoder(x, src_key_padding_mask=mask))


class SemanticCompressor(nn.Module):
    """Compress sequence to semantic vector."""
    def __init__(self, d_model=1024, d_semantic=4096):
        super().__init__()
        self.attention_pool = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.project = nn.Sequential(
            nn.Linear(d_model, d_semantic),
            nn.GELU(),
            nn.LayerNorm(d_semantic),
            nn.Dropout(0.1)
        )
        
    def forward(self, x, mask=None):
        # x: [batch, seq_len, d_model]
        batch_size = x.size(0)
        query = self.query.expand(batch_size, -1, -1)
        
        # Attention pooling
        pooled, _ = self.attention_pool(query, x, x, key_padding_mask=mask)
        pooled = pooled.squeeze(1)  # [batch, d_model]
        
        return self.project(pooled)  # [batch, d_semantic]


class CharGenerator(nn.Module):
    """Generate characters from semantic vector."""
    def __init__(self, vocab_size, d_semantic=4096, d_hidden=2048, num_layers=3):
        super().__init__()
        self.d_hidden = d_hidden
        self.num_layers = num_layers
        
        # Project semantic to initial hidden state
        self.semantic_to_h = nn.Linear(d_semantic, d_hidden * num_layers)
        self.semantic_to_c = nn.Linear(d_semantic, d_hidden * num_layers)
        
        # Character embedding for autoregressive generation
        self.char_embed = nn.Embedding(vocab_size, d_hidden)
        
        # LSTM generator
        self.lstm = nn.LSTM(d_hidden, d_hidden, num_layers=num_layers, 
                           batch_first=True, dropout=0.1)
        
        # Output projection with temperature control
        self.to_logits = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, vocab_size)
        )
        
    def forward(self, semantic, target_chars=None, max_length=128):
        batch_size = semantic.size(0)
        device = semantic.device
        
        # Initialize hidden state from semantic
        h = self.semantic_to_h(semantic).view(batch_size, self.num_layers, self.d_hidden)
        h = h.transpose(0, 1).contiguous()  # [num_layers, batch, d_hidden]
        c = self.semantic_to_c(semantic).view(batch_size, self.num_layers, self.d_hidden)
        c = c.transpose(0, 1).contiguous()
        
        if target_chars is not None:
            # Teacher forcing: use target chars as input
            # Shift right: input is [PAD, c1, c2, ...], target is [c1, c2, c3, ...]
            input_chars = torch.zeros_like(target_chars)
            input_chars[:, 1:] = target_chars[:, :-1]
            
            char_embeds = self.char_embed(input_chars)  # [batch, seq_len, d_hidden]
            outputs, _ = self.lstm(char_embeds, (h, c))
            logits = self.to_logits(outputs)  # [batch, seq_len, vocab_size]
            return logits
        else:
            # Autoregressive generation
            outputs = []
            current_char = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            hidden = (h, c)
            
            for _ in range(max_length):
                char_embed = self.char_embed(current_char)
                out, hidden = self.lstm(char_embed, hidden)
                logits = self.to_logits(out)  # [batch, 1, vocab_size]
                
                # Sample with temperature
                probs = F.softmax(logits[:, 0, :] / 0.8, dim=-1)
                current_char = torch.multinomial(probs, 1)
                outputs.append(current_char)
            
            return torch.cat(outputs, dim=1)  # [batch, max_length]


class GermanBrainV2(nn.Module):
    """
    Large German language model with char-level encoding.
    Target: ~300M parameters
    """
    def __init__(self, vocab_size, d_embed=512, d_pattern=1024, d_model=1024, 
                 d_semantic=4096, d_hidden=2048, num_encoder_layers=8, num_decoder_layers=3):
        super().__init__()
        
        # Comprehension pathway (Wernicke-inspired)
        self.pattern_detector = CharPatternDetector(vocab_size, d_embed, d_pattern)
        self.pattern_to_model = nn.Linear(d_pattern, d_model)
        self.encoder = TransformerEncoder(d_model, nhead=8, num_layers=num_encoder_layers)
        self.compressor = SemanticCompressor(d_model, d_semantic)
        
        # Production pathway (Broca-inspired)  
        self.generator = CharGenerator(vocab_size, d_semantic, d_hidden, num_decoder_layers)
        
        # Store dimensions
        self.d_semantic = d_semantic
        self.vocab_size = vocab_size
        
    def comprehend(self, chars, mask=None):
        """Encode character sequence to semantic vector."""
        patterns = self.pattern_detector(chars)
        features = self.pattern_to_model(patterns)
        encoded = self.encoder(features, mask)
        semantic = self.compressor(encoded, mask)
        return semantic
    
    def produce(self, semantic, target_chars=None, max_length=128):
        """Generate characters from semantic vector."""
        return self.generator(semantic, target_chars, max_length)
    
    def forward(self, chars, mask=None):
        """Full forward pass: comprehend then produce."""
        semantic = self.comprehend(chars, mask)
        logits = self.produce(semantic, target_chars=chars)
        return logits, semantic


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_german_data(max_sentences=500000):
    """Load German sentences from TinyStories."""
    print("Loading German dataset...")
    
    sentences = []
    
    # TinyStories German
    try:
        ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
        for item in tqdm(ds, desc="TinyStories", total=max_sentences):
            text = item.get('text', item.get('story', ''))
            if isinstance(text, str) and len(text) > 20:
                # Split into sentences
                for sent in text.replace('\n', ' ').split('. '):
                    sent = sent.strip()
                    if 20 < len(sent) < 200:
                        sentences.append(sent + '.')
                        if len(sentences) >= max_sentences:
                            break
            if len(sentences) >= max_sentences:
                break
    except Exception as e:
        print(f"Error loading TinyStories: {e}")
    
    print(f"Loaded {len(sentences)} sentences")
    return sentences


# =============================================================================
# TRAINING
# =============================================================================

def train_epoch(model, sentences, optimizer, scaler, scheduler, device, 
                batch_size=64, seq_len=128):
    """Train for one epoch with word boundary weighting."""
    model.train()
    
    random.shuffle(sentences)
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(range(0, len(sentences) - batch_size, batch_size), desc="Training")
    
    for i in pbar:
        batch_sentences = sentences[i:i+batch_size]
        
        # Convert to tensors
        batch_tensors = []
        for sent in batch_sentences:
            tensor = text_to_tensor(sent, seq_len)
            batch_tensors.append(tensor)
        
        chars = torch.stack(batch_tensors).to(device)
        
        # Create padding mask
        mask = (chars == char_to_idx[PAD_TOKEN])
        
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=torch.float16):
            # Forward pass
            logits, semantic = model(chars, mask)
            
            # Compute loss with word boundary weighting
            # Shift for next-char prediction
            targets = chars[:, 1:]  # [batch, seq_len-1]
            predictions = logits[:, :-1, :]  # [batch, seq_len-1, vocab]
            
            # Base cross-entropy (no reduction)
            loss_per_pos = F.cross_entropy(
                predictions.reshape(-1, model.vocab_size),
                targets.reshape(-1),
                ignore_index=char_to_idx[PAD_TOKEN],
                reduction='none'
            ).view(targets.shape)
            
            # Word boundary weights
            weights = torch.ones_like(loss_per_pos, dtype=torch.float32)
            for b in range(targets.size(0)):
                for t in range(targets.size(1)):
                    if is_word_boundary(targets[b, t].item()):
                        # Weight the position BEFORE word boundary (end of word)
                        if t > 0:
                            weights[b, t-1] = 2.0
                        weights[b, t] = 2.0
            
            # Weighted loss
            weighted_loss = (loss_per_pos * weights).sum() / weights.sum()
        
        scaler.scale(weighted_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += weighted_loss.item()
        num_batches += 1
        
        if num_batches % 100 == 0:
            pbar.set_postfix({'loss': f'{total_loss/num_batches:.4f}'})
    
    return total_loss / max(num_batches, 1)


def test_generation(model, device, prompts=None):
    """Test model generation."""
    model.eval()
    
    if prompts is None:
        prompts = [
            "Die Katze",
            "Es war einmal",
            "Der kleine Junge",
            "Heute ist ein",
            "Die Sonne scheint",
        ]
    
    print("\n📝 Generation Test:")
    with torch.no_grad():
        for prompt in prompts:
            chars = text_to_tensor(prompt, max_len=len(prompt)+1).unsqueeze(0).to(device)
            semantic = model.comprehend(chars)
            output = model.produce(semantic, max_length=60)
            generated = tensor_to_text(output[0])
            print(f"  '{prompt}' → '{generated[:50]}...'")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model config for ~300M params
    model = GermanBrainV2(
        vocab_size=VOCAB_SIZE,
        d_embed=512,
        d_pattern=1024,
        d_model=1024,
        d_semantic=4096,        # Increased as requested
        d_hidden=2048,
        num_encoder_layers=12,  # More layers for capacity
        num_decoder_layers=4,
    ).to(device)
    
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Verify ~300M
    if num_params < 200_000_000:
        print("⚠️  Model smaller than target, increasing size...")
        model = GermanBrainV2(
            vocab_size=VOCAB_SIZE,
            d_embed=768,
            d_pattern=1536,
            d_model=1536,
            d_semantic=4096,
            d_hidden=2048,
            num_encoder_layers=12,
            num_decoder_layers=4,
        ).to(device)
        num_params = count_parameters(model)
        print(f"Updated model parameters: {num_params:,} ({num_params/1e6:.1f}M)")
    
    # Load data
    sentences = load_german_data(max_sentences=500000)
    
    # Training setup
    BATCH_SIZE = 32  # Smaller batch for larger model
    EPOCHS = 50
    LR = 3e-4
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = GradScaler()
    
    steps_per_epoch = len(sentences) // BATCH_SIZE
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1
    )
    
    print(f"\nTraining config:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Steps/epoch: {steps_per_epoch}")
    print(f"  Learning rate: {LR}")
    
    # Training loop
    os.makedirs("checkpoints", exist_ok=True)
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{EPOCHS}")
        print(f"{'='*60}")
        
        start_time = time.time()
        loss = train_epoch(model, sentences, optimizer, scaler, scheduler, 
                          device, BATCH_SIZE)
        epoch_time = time.time() - start_time
        
        print(f"\n📊 Epoch {epoch}: Loss = {loss:.4f} ({epoch_time/60:.1f} min)")
        
        # Test generation
        test_generation(model, device)
        
        # Save checkpoint
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), "checkpoints/german_v2_best.pth")
            print(f"  💾 Saved best model (loss={loss:.4f})")
        
        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, f"checkpoints/german_v2_e{epoch}.pth")
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
