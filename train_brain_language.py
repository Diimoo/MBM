"""
Brain-Grounded Language Training

Trains the BrainLanguageSystem using:
1. Character-level processing (no tokenization)
2. Next-byte prediction (like language modeling)
3. Semantic comprehension and production
4. Hebbian associative learning

Phase 1: English semantic learning
Phase 2: German transfer (semantic alignment)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from typing import List, Tuple, Dict
from datetime import datetime

from digital_brain.modules.brain_language import (
    BrainLanguageSystem, 
    create_brain_language_config,
    text_to_bytes,
    bytes_to_text
)


# Sample training data (expandable with real corpus)
ENGLISH_SENTENCES = [
    "the cat sat on the mat",
    "the dog ran in the park",
    "the bird flew in the sky",
    "the fish swam in the water",
    "the sun shone bright today",
    "the moon glows at night",
    "red apples grow on trees",
    "blue water flows in rivers",
    "green grass covers the field",
    "the quick brown fox jumps",
    "a lazy dog sleeps well",
    "the happy child plays games",
    "big houses have many rooms",
    "small cars use less fuel",
    "old books tell great stories",
    "new ideas change the world",
    "hot coffee warms the body",
    "cold ice cools the drink",
    "soft music calms the mind",
    "loud thunder scares the cat",
]

GERMAN_SENTENCES = [
    "die katze saß auf der matte",
    "der hund rannte im park",
    "der vogel flog im himmel",
    "der fisch schwamm im wasser",
    "die sonne schien heute hell",
    "der mond leuchtet in der nacht",
    "rote äpfel wachsen auf bäumen",
    "blaues wasser fließt in flüssen",
    "grünes gras bedeckt das feld",
    "der schnelle braune fuchs springt",
    "ein fauler hund schläft gut",
    "das glückliche kind spielt spiele",
    "große häuser haben viele zimmer",
    "kleine autos verbrauchen weniger",
    "alte bücher erzählen geschichten",
    "neue ideen verändern die welt",
    "heißer kaffee wärmt den körper",
    "kaltes eis kühlt das getränk",
    "sanfte musik beruhigt den geist",
    "lauter donner erschreckt die katze",
]

# Translation pairs for semantic alignment
TRANSLATION_PAIRS = list(zip(ENGLISH_SENTENCES, GERMAN_SENTENCES))


def create_byte_dataset(sentences: List[str], max_len: int = 128) -> torch.Tensor:
    """Convert sentences to byte tensors."""
    byte_sequences = []
    for sent in sentences:
        bytes_tensor = text_to_bytes(sent, max_len)
        byte_sequences.append(bytes_tensor)
    return torch.stack(byte_sequences)


def train_english_semantics(
    brain: BrainLanguageSystem,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = 'cuda'
) -> Dict:
    """
    Phase 1: Train brain to understand English semantics.
    Uses next-byte prediction (language modeling objective).
    """
    print("=" * 70)
    print("PHASE 1: ENGLISH SEMANTIC LEARNING")
    print("=" * 70)
    print("Training on character-level patterns (no tokenization)")
    print(f"Corpus: {len(ENGLISH_SENTENCES)} sentences")
    print("=" * 70)
    
    brain.train()
    optimizer = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Create dataset
    dataset = create_byte_dataset(ENGLISH_SENTENCES, max_len=64).to(device)
    num_samples = dataset.shape[0]
    
    losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle
        perm = torch.randperm(num_samples)
        dataset = dataset[perm]
        
        for i in range(0, num_samples, batch_size):
            batch = dataset[i:i+batch_size]
            if batch.shape[0] < 2:
                continue
            
            # Input: all bytes except last
            input_bytes = batch[:, :-1]
            target_bytes = batch[:, 1:]
            
            # COMPREHENSION: Understand input
            semantic, concepts = brain.comprehend(input_bytes, store_in_memory=True)
            
            # PRODUCTION: Generate continuation (teacher forcing)
            logits = brain.produce(semantic, target_bytes=input_bytes)
            
            # Loss: Next-byte prediction
            loss = F.cross_entropy(
                logits.reshape(-1, 256),
                target_bytes.reshape(-1),
                ignore_index=0  # Ignore padding
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: Loss={avg_loss:.4f}")
            
            # Test generation
            if epoch % 20 == 0:
                test_english_generation(brain, device)
    
    print("-" * 70)
    print(f"Final Loss: {losses[-1]:.4f}")
    
    return {'losses': losses, 'final_loss': losses[-1]}


def test_english_generation(brain: BrainLanguageSystem, device: str):
    """Test English text generation."""
    brain.eval()
    
    test_prompts = ["the cat", "red apple", "the sun"]
    
    print("\n  Generation test:")
    for prompt in test_prompts:
        prompt_bytes = text_to_bytes(prompt, 64).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Comprehend prompt
            semantic, _ = brain.comprehend(prompt_bytes, store_in_memory=False)
            
            # Generate continuation
            generated = brain.produce(semantic, max_length=32)
            generated_text = bytes_to_text(generated)
        
        print(f"    '{prompt}' → '{generated_text}'")
    
    brain.train()
    print()


def train_german_transfer(
    brain: BrainLanguageSystem,
    epochs: int = 30,
    batch_size: int = 4,
    lr: float = 5e-4,
    device: str = 'cuda'
) -> Dict:
    """
    Phase 2: Teach German by semantic alignment.
    English and German map to SAME semantic representation.
    
    Key insight: Freeze English comprehension, train German production.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: GERMAN SEMANTIC TRANSFER")
    print("=" * 70)
    print("Aligning German to English semantic space")
    print(f"Translation pairs: {len(TRANSLATION_PAIRS)}")
    print("=" * 70)
    
    # Freeze Wernicke (preserve English understanding)
    for param in brain.wernicke.parameters():
        param.requires_grad = False
    
    # Train Broca (learn German production)
    for param in brain.broca.parameters():
        param.requires_grad = True
    
    # Also train a German comprehension adapter
    brain.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, brain.parameters()),
        lr=lr, weight_decay=0.01
    )
    
    # Create paired dataset
    en_dataset = create_byte_dataset([p[0] for p in TRANSLATION_PAIRS], 64).to(device)
    de_dataset = create_byte_dataset([p[1] for p in TRANSLATION_PAIRS], 64).to(device)
    num_pairs = len(TRANSLATION_PAIRS)
    
    losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle pairs together
        perm = torch.randperm(num_pairs)
        en_dataset = en_dataset[perm]
        de_dataset = de_dataset[perm]
        
        for i in range(0, num_pairs, batch_size):
            en_batch = en_dataset[i:i+batch_size]
            de_batch = de_dataset[i:i+batch_size]
            
            if en_batch.shape[0] < 2:
                continue
            
            # Get English semantic representation (frozen)
            with torch.no_grad():
                en_semantic, _ = brain.comprehend(en_batch, store_in_memory=False)
            
            # Generate German from SAME semantic (train production)
            de_input = de_batch[:, :-1]
            de_target = de_batch[:, 1:]
            
            de_logits = brain.produce(en_semantic, target_bytes=de_input)
            
            # Loss: German reconstruction from English semantics
            loss = F.cross_entropy(
                de_logits.reshape(-1, 256),
                de_target.reshape(-1),
                ignore_index=0
            )
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / max(num_batches, 1)
        losses.append(avg_loss)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}: German Loss={avg_loss:.4f}")
            
            if epoch % 15 == 0:
                test_cross_lingual(brain, device)
    
    # Unfreeze for future training
    for param in brain.wernicke.parameters():
        param.requires_grad = True
    
    print("-" * 70)
    print(f"Final German Loss: {losses[-1]:.4f}")
    
    return {'losses': losses, 'final_loss': losses[-1]}


def test_cross_lingual(brain: BrainLanguageSystem, device: str):
    """Test cross-lingual semantic similarity."""
    brain.eval()
    
    test_pairs = [
        ("the cat", "die katze"),
        ("red apple", "roter apfel"),
        ("the sun", "die sonne"),
    ]
    
    print("\n  Cross-lingual test:")
    
    for en, de in test_pairs:
        en_bytes = text_to_bytes(en, 64).unsqueeze(0).to(device)
        de_bytes = text_to_bytes(de, 64).unsqueeze(0).to(device)
        
        with torch.no_grad():
            en_semantic, _ = brain.comprehend(en_bytes, store_in_memory=False)
            de_semantic, _ = brain.comprehend(de_bytes, store_in_memory=False)
            
            # Cosine similarity
            similarity = F.cosine_similarity(en_semantic, de_semantic).item()
            
            # Generate German from English semantic
            de_generated = brain.produce(en_semantic, max_length=32)
            de_text = bytes_to_text(de_generated)
        
        print(f"    EN: '{en}' ↔ DE: '{de}'")
        print(f"    Similarity: {similarity:.3f} | Generated: '{de_text}'")
    
    brain.train()
    print()


def run_experiments(brain: BrainLanguageSystem, device: str) -> Dict:
    """
    Run all validation experiments.
    """
    print("\n" + "#" * 70)
    print("# BRAIN-LIKE LANGUAGE EXPERIMENTS")
    print("#" * 70)
    
    results = {}
    
    # Experiment 1: Abstraction Control (lATL Dimmer)
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: ABSTRACTION CONTROL (lATL DIMMER)")
    print("=" * 70)
    
    brain.eval()
    test_text = "the animal was large and gray"
    test_bytes = text_to_bytes(test_text, 64).unsqueeze(0).to(device)
    
    abstraction_results = []
    levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    for level in levels:
        with torch.no_grad():
            brain.set_abstraction_level(level)
            semantic, concepts = brain.comprehend(test_bytes, store_in_memory=False)
            generated = brain.produce(semantic, max_length=40)
            text = bytes_to_text(generated)
        
        abstraction_results.append({'level': level, 'output': text})
        print(f"Level {level:.2f}: '{text}'")
    
    results['abstraction'] = abstraction_results
    
    # Experiment 2: N400 Surprise Detection
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: N400 SURPRISE DETECTION")
    print("=" * 70)
    
    n400_cases = [
        ("the cat sat on the", "mat", "Expected"),
        ("the cat sat on the", "fish", "Unexpected"),
        ("the bird flew in the", "sky", "Expected"),
        ("the bird flew in the", "table", "Unexpected"),
    ]
    
    n400_results = []
    
    for context, word, expectation in n400_cases:
        context_bytes = text_to_bytes(context, 64).unsqueeze(0).to(device)
        word_bytes = text_to_bytes(word, 64).unsqueeze(0).to(device)
        
        with torch.no_grad():
            surprise = brain.test_n400(context_bytes, word_bytes).item()
        
        n400_results.append({
            'context': context,
            'word': word,
            'expectation': expectation,
            'surprise': surprise
        })
        print(f"'{context}' + '{word}' ({expectation}): Surprise={surprise:.4f}")
    
    # Check if unexpected words have higher surprise
    expected_surprises = [r['surprise'] for r in n400_results if r['expectation'] == 'Expected']
    unexpected_surprises = [r['surprise'] for r in n400_results if r['expectation'] == 'Unexpected']
    
    avg_expected = np.mean(expected_surprises) if expected_surprises else 0
    avg_unexpected = np.mean(unexpected_surprises) if unexpected_surprises else 0
    
    n400_works = avg_unexpected > avg_expected
    print(f"\nExpected avg: {avg_expected:.4f}, Unexpected avg: {avg_unexpected:.4f}")
    print(f"N400 effect: {'✓ DETECTED' if n400_works else '✗ NOT DETECTED'}")
    
    results['n400'] = {
        'cases': n400_results,
        'effect_detected': n400_works,
        'expected_avg': avg_expected,
        'unexpected_avg': avg_unexpected
    }
    
    # Experiment 3: Associative Activation
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ASSOCIATIVE ACTIVATION")
    print("=" * 70)
    
    test_words = ["cat", "sun", "water"]
    association_results = []
    
    for word in test_words:
        word_bytes = text_to_bytes(word, 64).unsqueeze(0).to(device)
        
        with torch.no_grad():
            semantic, concept_activation = brain.comprehend(word_bytes, store_in_memory=False)
            
            # Get top activated concepts
            top_k = torch.topk(concept_activation[0], k=5)
            activation_strength = top_k.values.mean().item()
        
        association_results.append({
            'word': word,
            'activation_strength': activation_strength
        })
        print(f"'{word}': Top-5 activation strength = {activation_strength:.4f}")
    
    results['associations'] = association_results
    
    # Experiment 4: Cross-lingual Semantic Similarity
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: CROSS-LINGUAL SEMANTICS")
    print("=" * 70)
    
    cross_pairs = [
        ("the cat", "die katze"),
        ("red apple", "roter apfel"),
        ("big house", "großes haus"),
        ("the sun shines", "die sonne scheint"),
    ]
    
    cross_results = []
    
    for en, de in cross_pairs:
        en_bytes = text_to_bytes(en, 64).unsqueeze(0).to(device)
        de_bytes = text_to_bytes(de, 64).unsqueeze(0).to(device)
        
        with torch.no_grad():
            en_semantic, _ = brain.comprehend(en_bytes, store_in_memory=False)
            de_semantic, _ = brain.comprehend(de_bytes, store_in_memory=False)
            
            similarity = F.cosine_similarity(en_semantic, de_semantic).item()
        
        cross_results.append({
            'english': en,
            'german': de,
            'similarity': similarity
        })
        print(f"'{en}' ↔ '{de}': {similarity:.3f}")
    
    avg_similarity = np.mean([r['similarity'] for r in cross_results])
    print(f"\nAverage cross-lingual similarity: {avg_similarity:.3f}")
    
    results['cross_lingual'] = {
        'pairs': cross_results,
        'average_similarity': avg_similarity
    }
    
    brain.train()
    
    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"✓ Abstraction control: {len(abstraction_results)} levels tested")
    print(f"{'✓' if n400_works else '✗'} N400 effect: {'detected' if n400_works else 'not detected'}")
    print(f"✓ Associative activation: avg strength = {np.mean([r['activation_strength'] for r in association_results]):.4f}")
    print(f"✓ Cross-lingual: avg similarity = {avg_similarity:.3f}")
    
    return results


def main():
    """Main training and experiment pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Brain-Grounded Language Training')
    parser.add_argument('--epochs-en', type=int, default=50, help='English training epochs')
    parser.add_argument('--epochs-de', type=int, default=30, help='German transfer epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip-training', action='store_true', help='Skip training, run experiments only')
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create brain
    config = create_brain_language_config()
    brain = BrainLanguageSystem(config).to(device)
    
    total_params = sum(p.numel() for p in brain.parameters())
    print(f"Brain parameters: {total_params:,}")
    
    results = {'config': config}
    
    if not args.skip_training:
        # Phase 1: English
        en_results = train_english_semantics(brain, epochs=args.epochs_en, device=device)
        results['english'] = en_results
        
        # Phase 2: German transfer
        de_results = train_german_transfer(brain, epochs=args.epochs_de, device=device)
        results['german'] = de_results
        
        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(brain.state_dict(), 'checkpoints/brain_language.pth')
        print("\nCheckpoint saved to checkpoints/brain_language.pth")
    
    # Run experiments
    exp_results = run_experiments(brain, device)
    results['experiments'] = exp_results
    
    # Save results
    os.makedirs('experiments', exist_ok=True)
    
    # Convert to JSON-serializable
    results_json = {
        'config': results['config'],
        'experiments': {
            'n400': {
                'effect_detected': bool(results['experiments']['n400']['effect_detected']),
                'expected_avg': float(results['experiments']['n400']['expected_avg']),
                'unexpected_avg': float(results['experiments']['n400']['unexpected_avg']),
            },
            'cross_lingual': {
                'average_similarity': float(results['experiments']['cross_lingual']['average_similarity']),
            },
        }
    }
    
    with open('experiments/brain_language_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print("\nResults saved to experiments/brain_language_results.json")
    
    return brain, results


if __name__ == '__main__':
    brain, results = main()
