#!/usr/bin/env python3
"""
Analyze trained German model:
1. Test generation interactively
2. Analyze dataset for noise
3. Check semantic embedding diversity
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import Counter
import re

from digital_brain.modules.brain_language import (
    BrainLanguageSystem,
    create_brain_language_config
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# 1. LOAD MODEL
# =============================================================================
print("\n" + "="*60)
print("1. LOADING MODEL")
print("="*60)

config = create_brain_language_config()
config['char_embedding_dim'] = 512
config['d_pattern'] = 512
config['d_semantic'] = 1024
config['d_hidden'] = 1024
config['num_concepts'] = 5000  # Match checkpoint
config['memory_capacity'] = 50000  # Match checkpoint

brain = BrainLanguageSystem(config).to(device)

checkpoint = torch.load("checkpoints/german_latest.pth", map_location=device)

# Handle both formats: direct state_dict or wrapped
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    brain.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint.get('epoch', 'unknown')
    loss = checkpoint.get('loss', 'unknown')
else:
    # Direct state_dict
    brain.load_state_dict(checkpoint)
    epoch = 21  # From log
    loss = 4.60

brain.eval()

print(f"✅ Model loaded from epoch {epoch}")
print(f"   Loss: {loss}")

# =============================================================================
# 2. TEST GENERATION
# =============================================================================
print("\n" + "="*60)
print("2. GENERATION TESTS")
print("="*60)

def safe_text_to_tensor(text, max_len=128):
    """Convert text to tensor safely."""
    bytes_list = list(text.encode('utf-8', errors='replace'))[:max_len]
    bytes_list = [min(max(b, 0), 255) for b in bytes_list]
    return torch.tensor(bytes_list, dtype=torch.long).unsqueeze(0).to(device)

def generate_from_prompt(brain, prompt, max_len=100):
    """Generate text from a prompt."""
    with torch.no_grad():
        input_tensor = safe_text_to_tensor(prompt)
        semantic, _ = brain.comprehend(input_tensor)
        
        # Generate (no temperature param in this API)
        output = brain.produce(semantic, max_length=max_len)
        if isinstance(output, torch.Tensor):
            output_bytes = output[0].cpu().numpy()
            text = bytes(output_bytes).decode('utf-8', errors='replace')
            return text
        return str(output)

prompts = [
    "Die Katze",
    "Es war einmal",
    "Der kleine",
    "Heute ist",
    "Ich bin",
    "Das Haus",
    "Ein Mann",
    "Die Sonne",
]

print("\nGeneration results:")
for prompt in prompts:
    result = generate_from_prompt(brain, prompt)
    # Show first 60 chars
    display = result[:60].replace('\n', '↵')
    print(f"  '{prompt}' → '{display}...'")

# =============================================================================
# 3. ANALYZE DATASET FOR NOISE
# =============================================================================
print("\n" + "="*60)
print("3. DATASET NOISE ANALYSIS")
print("="*60)

# Load sample sentences from Hugging Face datasets
import os
try:
    from datasets import load_dataset
    print("Loading sample from TinyStories German...")
    ds = load_dataset("SkySyrup/tinystories_german", split="train", streaming=True)
    sentences = []
    for i, item in enumerate(ds):
        text = item.get('text', item.get('story', str(item)))
        if isinstance(text, str) and len(text) > 10:
            # Split into sentences
            for sent in text.replace('\n', ' ').split('. '):
                if len(sent) > 10:
                    sentences.append(sent.strip())
                    if len(sentences) >= 10000:
                        break
        if len(sentences) >= 10000:
            break
    print(f"Loaded {len(sentences)} sentences")
except Exception as e:
    print(f"Could not load dataset: {e}")
    sentences = []

print(f"Analyzing {len(sentences)} sample sentences...")

# Check for problematic patterns
noise_patterns = {
    'numbers_only': r'^[\d\s\.,]+$',
    'long_numbers': r'\d{6,}',
    'urls': r'https?://',
    'emails': r'@\w+\.',
    'special_chars_heavy': r'[^\w\s]{5,}',
    'non_german_chars': r'[^\x00-\x7F\xC0-\xFF\u00C0-\u017F]',
    'very_short': lambda s: len(s) < 10,
    'very_long': lambda s: len(s) > 500,
}

noise_counts = Counter()
noise_examples = {}

for sent in sentences:
    for name, pattern in noise_patterns.items():
        if callable(pattern):
            if pattern(sent):
                noise_counts[name] += 1
                if name not in noise_examples:
                    noise_examples[name] = sent[:100]
        else:
            if re.search(pattern, sent):
                noise_counts[name] += 1
                if name not in noise_examples:
                    noise_examples[name] = sent[:100]

print("\nNoise pattern frequency:")
for name, count in noise_counts.most_common():
    pct = 100 * count / len(sentences)
    print(f"  {name}: {count} ({pct:.1f}%)")
    if name in noise_examples:
        print(f"    Example: '{noise_examples[name][:80]}...'")

# Character frequency analysis
all_chars = ''.join(sentences)
char_freq = Counter(all_chars)
print(f"\nTotal characters: {len(all_chars)}")
print(f"Unique characters: {len(char_freq)}")
print("\nMost common characters:")
for char, count in char_freq.most_common(20):
    pct = 100 * count / len(all_chars)
    display_char = repr(char) if char in '\n\t\r ' else char
    print(f"  {display_char}: {count} ({pct:.2f}%)")

print("\nLeast common characters (potential noise):")
for char, count in char_freq.most_common()[-20:]:
    if count < 100:
        display_char = repr(char) if ord(char) < 32 or ord(char) > 126 else char
        print(f"  {display_char} (ord={ord(char)}): {count}")

# =============================================================================
# 4. CHECK SEMANTIC EMBEDDING DIVERSITY
# =============================================================================
print("\n" + "="*60)
print("4. SEMANTIC EMBEDDING DIVERSITY")
print("="*60)

# Encode various sentences and check diversity
test_sentences = [
    "Die Katze schläft.",
    "Der Hund bellt.",
    "Das Auto fährt.",
    "Die Sonne scheint.",
    "Der Mann liest.",
    "Die Frau kocht.",
    "Das Kind spielt.",
    "Der Vogel singt.",
    "Die Blume blüht.",
    "Das Wasser fließt.",
    # Similar sentences
    "Die Katze schläft ruhig.",
    "Die Katze schläft laut.",
    "Die Katze rennt.",
    # Very different
    "Heute ist Montag.",
    "1234567890",
    "Hello world!",
]

embeddings = []
labels = []

with torch.no_grad():
    for sent in test_sentences:
        input_tensor = safe_text_to_tensor(sent)
        semantic, _ = brain.comprehend(input_tensor)
        embeddings.append(semantic.cpu().numpy().flatten())
        labels.append(sent[:30])

embeddings = np.array(embeddings)
print(f"Embedding shape: {embeddings.shape}")

# Statistics
print("\n📊 Embedding Statistics:")
print(f"  Mean: {embeddings.mean():.4f}")
print(f"  Std:  {embeddings.std():.4f}")
print(f"  Min:  {embeddings.min():.4f}")
print(f"  Max:  {embeddings.max():.4f}")

# Check for collapse - are all embeddings similar?
print("\n📊 Pairwise Cosine Similarities:")

# Normalize for cosine similarity
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
normalized = embeddings / (norms + 1e-8)

similarity_matrix = normalized @ normalized.T

print(f"  Mean similarity: {similarity_matrix.mean():.4f}")
print(f"  Min similarity:  {similarity_matrix.min():.4f}")
print(f"  Max similarity:  {similarity_matrix.max():.4f}")

# Check diagonal dominance (should be 1.0)
diag_mean = np.diag(similarity_matrix).mean()
off_diag = similarity_matrix[~np.eye(similarity_matrix.shape[0], dtype=bool)]
off_diag_mean = off_diag.mean()

print(f"\n  Diagonal mean (self-similarity): {diag_mean:.4f}")
print(f"  Off-diagonal mean (cross-similarity): {off_diag_mean:.4f}")

if off_diag_mean > 0.95:
    print("\n  ⚠️  WARNING: Embeddings are nearly identical (COLLAPSE detected)")
elif off_diag_mean > 0.8:
    print("\n  ⚠️  WARNING: Embeddings are very similar (partial collapse)")
elif off_diag_mean > 0.5:
    print("\n  ℹ️  Embeddings have moderate diversity")
else:
    print("\n  ✅ Embeddings are diverse")

# Show some specific similarities
print("\n📊 Specific Pair Similarities:")
pairs = [
    (0, 1),   # Katze vs Hund
    (0, 10),  # Katze schläft vs Katze schläft ruhig
    (0, 12),  # Katze schläft vs Katze rennt
    (0, 13),  # Katze vs Heute ist Montag
    (0, 14),  # Katze vs numbers
    (0, 15),  # Katze vs English
]

for i, j in pairs:
    sim = similarity_matrix[i, j]
    print(f"  '{labels[i]}' vs '{labels[j]}': {sim:.4f}")

# Variance per dimension
dim_variance = embeddings.var(axis=0)
print(f"\n📊 Per-dimension Variance:")
print(f"  Mean: {dim_variance.mean():.6f}")
print(f"  Max:  {dim_variance.max():.6f}")
print(f"  Min:  {dim_variance.min():.6f}")
print(f"  Dims with variance < 0.001: {(dim_variance < 0.001).sum()} / {len(dim_variance)}")

# =============================================================================
# 5. RECONSTRUCTION TEST
# =============================================================================
print("\n" + "="*60)
print("5. RECONSTRUCTION TEST")
print("="*60)

def test_reconstruction(brain, text):
    """Test if model can reconstruct input from semantic."""
    with torch.no_grad():
        input_tensor = safe_text_to_tensor(text)
        semantic, _ = brain.comprehend(input_tensor)
        
        # Try to reconstruct
        recon_logits = brain.reconstruct_input(semantic)
        recon_bytes = recon_logits.argmax(dim=-1)[0].cpu().numpy()
        recon_text = bytes(recon_bytes).decode('utf-8', errors='replace')
        
        return recon_text

print("\nReconstruction tests:")
for text in ["Die Katze", "Es war einmal", "Hallo Welt", "123456"]:
    recon = test_reconstruction(brain, text)
    print(f"  Original: '{text}'")
    print(f"  Reconstructed: '{recon[:len(text)+20]}'")
    print()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
