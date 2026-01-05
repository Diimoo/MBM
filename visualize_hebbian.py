#!/usr/bin/env python3
"""
Visualize Hebbian associations learned by the brain language system.
Shows which concepts fire together and wire together.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from digital_brain.modules.brain_language import (
    BrainLanguageSystem,
    create_brain_language_config,
    text_to_bytes
)


def load_trained_brain(checkpoint_path: str = 'checkpoints/brain_language.pth'):
    """Load trained brain from checkpoint."""
    config = create_brain_language_config()
    brain = BrainLanguageSystem(config)
    
    if Path(checkpoint_path).exists():
        brain.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"No checkpoint found at {checkpoint_path}, using random weights")
    
    brain.eval()
    return brain


def visualize_association_matrix(brain: BrainLanguageSystem, save_path: str = 'figures/hebbian_associations.png'):
    """Visualize the Hebbian association matrix as a heatmap."""
    # Get association matrix
    assoc_matrix = brain.wernicke.associative_network.associations.detach().cpu().numpy()
    
    print(f"Association matrix shape: {assoc_matrix.shape}")
    print(f"Value range: [{assoc_matrix.min():.3f}, {assoc_matrix.max():.3f}]")
    print(f"Mean: {assoc_matrix.mean():.3f}, Std: {assoc_matrix.std():.3f}")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Full matrix overview (downsampled)
    ax1 = axes[0]
    step = max(1, assoc_matrix.shape[0] // 100)
    downsampled = assoc_matrix[::step, ::step]
    im1 = ax1.imshow(downsampled, cmap='RdBu_r', aspect='auto', 
                     vmin=-2, vmax=2)
    ax1.set_title(f'Full Association Matrix\n(downsampled {step}x)')
    ax1.set_xlabel('Concept Index')
    ax1.set_ylabel('Concept Index')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Top-left corner detail
    ax2 = axes[1]
    corner_size = min(50, assoc_matrix.shape[0])
    corner = assoc_matrix[:corner_size, :corner_size]
    im2 = ax2.imshow(corner, cmap='RdBu_r', aspect='auto',
                     vmin=-2, vmax=2)
    ax2.set_title(f'First {corner_size}x{corner_size} Concepts')
    ax2.set_xlabel('Concept Index')
    ax2.set_ylabel('Concept Index')
    plt.colorbar(im2, ax=ax2)
    
    # 3. Distribution of association weights
    ax3 = axes[2]
    weights_flat = assoc_matrix.flatten()
    ax3.hist(weights_flat, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', label='Zero')
    ax3.axvline(x=weights_flat.mean(), color='green', linestyle='--', 
                label=f'Mean ({weights_flat.mean():.2f})')
    ax3.set_title('Distribution of Association Weights')
    ax3.set_xlabel('Weight Value')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved association matrix visualization to {save_path}")
    plt.close()


def find_strongest_associations(brain: BrainLanguageSystem, top_k: int = 5):
    """Find and print the strongest positive and negative associations."""
    assoc_matrix = brain.wernicke.associative_network.associations.detach().cpu().numpy()
    
    print("\n" + "=" * 60)
    print("STRONGEST HEBBIAN ASSOCIATIONS")
    print("=" * 60)
    
    # Find top positive associations (excluding self-connections)
    np.fill_diagonal(assoc_matrix, -np.inf)  # Mask diagonal
    
    # Flatten and find top indices
    flat_idx = np.argsort(assoc_matrix.flatten())[::-1]
    
    print(f"\nTop {top_k} POSITIVE associations (concepts that fire together):")
    for i in range(top_k):
        idx = flat_idx[i]
        row = idx // assoc_matrix.shape[1]
        col = idx % assoc_matrix.shape[1]
        weight = assoc_matrix[row, col]
        print(f"  Concept {row} ↔ Concept {col}: weight = {weight:.4f}")
    
    print(f"\nTop {top_k} NEGATIVE associations (inhibitory connections):")
    for i in range(top_k):
        idx = flat_idx[-(i+1)]
        row = idx // assoc_matrix.shape[1]
        col = idx % assoc_matrix.shape[1]
        weight = assoc_matrix[row, col]
        print(f"  Concept {row} ↔ Concept {col}: weight = {weight:.4f}")


def analyze_concept_activations(brain: BrainLanguageSystem, device: str = 'cpu'):
    """Analyze which concepts activate for different inputs."""
    test_words = [
        "cat", "dog", "bird", "fish",
        "red", "blue", "green",
        "big", "small", "fast", "slow",
        "run", "walk", "swim", "fly"
    ]
    
    print("\n" + "=" * 60)
    print("CONCEPT ACTIVATIONS FOR DIFFERENT WORDS")
    print("=" * 60)
    
    activations = {}
    
    for word in test_words:
        word_bytes = text_to_bytes(word, 64).unsqueeze(0).to(device)
        
        with torch.no_grad():
            _, concept_activation = brain.comprehend(word_bytes, store_in_memory=False)
        
        # Get top-5 activated concepts
        top_values, top_indices = torch.topk(concept_activation[0], 5)
        activations[word] = {
            'indices': top_indices.cpu().numpy(),
            'values': top_values.cpu().numpy()
        }
        
        print(f"\n'{word}':")
        for idx, val in zip(top_indices.cpu().numpy(), top_values.cpu().numpy()):
            print(f"  Concept {idx}: {val:.4f}")
    
    # Find shared concepts between related words
    print("\n" + "-" * 40)
    print("SHARED CONCEPTS BETWEEN RELATED WORDS:")
    print("-" * 40)
    
    pairs = [
        ("cat", "dog"),
        ("red", "blue"),
        ("run", "walk"),
        ("big", "small"),
    ]
    
    for word1, word2 in pairs:
        idx1 = set(activations[word1]['indices'])
        idx2 = set(activations[word2]['indices'])
        shared = idx1 & idx2
        print(f"'{word1}' & '{word2}': {len(shared)} shared concepts - {shared}")


def visualize_concept_similarity(brain: BrainLanguageSystem, device: str = 'cpu',
                                  save_path: str = 'figures/concept_similarity.png'):
    """Visualize semantic similarity between words based on concept activations."""
    test_words = [
        "cat", "dog", "bird", "fish",
        "red", "blue", "green", "yellow",
        "big", "small", "fast", "slow",
        "happy", "sad", "good", "bad"
    ]
    
    # Get embeddings for all words
    embeddings = []
    for word in test_words:
        word_bytes = text_to_bytes(word, 64).unsqueeze(0).to(device)
        with torch.no_grad():
            semantic, _ = brain.comprehend(word_bytes, store_in_memory=False)
        embeddings.append(semantic[0].cpu().numpy())
    
    embeddings = np.array(embeddings)
    
    # Compute cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(embeddings)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, xticklabels=test_words, yticklabels=test_words,
                cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                square=True, linewidths=0.5)
    plt.title('Semantic Similarity Between Words\n(Based on Learned Embeddings)')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved concept similarity visualization to {save_path}")
    plt.close()


def main():
    print("=" * 60)
    print("HEBBIAN ASSOCIATION VISUALIZATION")
    print("=" * 60)
    
    # Load trained brain
    brain = load_trained_brain()
    device = 'cpu'
    
    # 1. Visualize association matrix
    visualize_association_matrix(brain)
    
    # 2. Find strongest associations
    find_strongest_associations(brain, top_k=10)
    
    # 3. Analyze concept activations
    analyze_concept_activations(brain, device)
    
    # 4. Visualize concept similarity
    try:
        visualize_concept_similarity(brain, device)
    except ImportError:
        print("\nSkipping similarity visualization (sklearn not available)")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
