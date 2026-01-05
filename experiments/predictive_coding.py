"""
Predictive Coding Analysis (N400-like Response)

Test whether the Digital Brain shows prediction error spikes on semantic violations,
similar to the human N400 ERP response.

Normal sentence: "The ball rolled down the hill"
Violation: "The ball rolled down the salad"

Expected: Surprise spikes at violation word ("salad") but not at normal word ("hill")
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import json

import sys
sys.path.insert(0, '/home/ahmed/Downloads/Kandel')

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs


class PredictiveLanguageAnalyzer:
    """Analyze prediction errors in language processing."""
    
    def __init__(self, brain: DigitalBrain, device: str = 'cuda'):
        self.brain = brain
        self.device = device
        
        # Simple vocabulary for test sentences
        self.vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'the': 3, 'ball': 4, 'rolled': 5, 'down': 6,
            'hill': 7, 'salad': 8, 'cat': 9, 'sat': 10,
            'on': 11, 'mat': 12, 'fish': 13, 'swam': 14,
            'in': 15, 'water': 16, 'fire': 17, 'bird': 18,
            'flew': 19, 'sky': 20, 'rock': 21, 'dog': 22,
            'ran': 23, 'park': 24, 'moon': 25, 'sun': 26,
            'shone': 27, 'bright': 28, 'cloud': 29, 'rain': 30,
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
    def tokenize(self, sentence: str) -> List[int]:
        """Convert sentence to token IDs."""
        tokens = [self.vocab['<START>']]
        for word in sentence.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens.append(self.vocab['<END>'])
        return tokens
    
    def compute_surprise(self, tokens: List[int]) -> List[float]:
        """
        Compute surprise (prediction error) for each token in sequence.
        
        Surprise = -log P(word | context)
        Approximated by reconstruction error from cortex.
        """
        self.brain.eval()
        surprise_log = []
        
        with torch.no_grad():
            # Initialize brain state
            self.brain.reset(1)
            
            # Create dummy observation (we're testing language, not vision)
            dummy_obs = torch.zeros(1, self.brain.config['d_obs'], device=self.device)
            obs_wrapped = Obs(x=dummy_obs)
            
            prev_reward = torch.zeros(1, device=self.device)
            prev_done = torch.zeros(1, dtype=torch.bool, device=self.device)
            
            # Process each token and measure prediction error
            for i, token in enumerate(tokens):
                # Create token tensor
                token_seq = torch.tensor([tokens[:i+1]], dtype=torch.long, device=self.device)
                # Pad to fixed length
                padded = torch.zeros(1, 15, dtype=torch.long, device=self.device)
                padded[0, :min(i+1, 15)] = token_seq[0, :min(i+1, 15)]
                
                # Forward pass through brain
                action, log_prob, value, state, log, entropy = self.brain.step(
                    obs_wrapped, prev_reward, prev_done,
                    learn=False, instruction=padded
                )
                
                # Surprise approximated by prediction error from cortex
                pred_error = log.pred_error if hasattr(log, 'pred_error') else 0.0
                
                # Also use entropy as proxy for uncertainty
                ent_value = entropy.mean().item() if entropy is not None else 0.0
                
                # Combine metrics (higher = more surprising)
                surprise = pred_error + 0.5 * ent_value
                surprise_log.append(surprise)
        
        self.brain.train()
        return surprise_log
    
    def analyze_sentence_pair(self, normal: str, violation: str, 
                              violation_position: int) -> Dict:
        """
        Compare surprise patterns between normal and violation sentences.
        
        Args:
            normal: Normal/expected sentence
            violation: Sentence with semantic violation
            violation_position: Word index where violation occurs (0-indexed)
        """
        normal_tokens = self.tokenize(normal)
        violation_tokens = self.tokenize(violation)
        
        normal_surprise = self.compute_surprise(normal_tokens)
        violation_surprise = self.compute_surprise(violation_tokens)
        
        # Get words for labels
        normal_words = ['<START>'] + normal.lower().split() + ['<END>']
        violation_words = ['<START>'] + violation.lower().split() + ['<END>']
        
        # Compute spike at violation point (adjusted for <START> token)
        viol_idx = violation_position + 1  # +1 for <START>
        
        normal_at_viol = normal_surprise[viol_idx] if viol_idx < len(normal_surprise) else 0
        violation_at_viol = violation_surprise[viol_idx] if viol_idx < len(violation_surprise) else 0
        
        spike = violation_at_viol - normal_at_viol
        
        return {
            'normal_sentence': normal,
            'violation_sentence': violation,
            'normal_surprise': normal_surprise,
            'violation_surprise': violation_surprise,
            'normal_words': normal_words,
            'violation_words': violation_words,
            'violation_position': viol_idx,
            'surprise_at_violation_normal': normal_at_viol,
            'surprise_at_violation_anomaly': violation_at_viol,
            'spike_magnitude': spike,
        }


def run_n400_analysis(brain: DigitalBrain, device: str = 'cuda') -> Dict:
    """
    Run N400-like analysis on multiple sentence pairs.
    """
    analyzer = PredictiveLanguageAnalyzer(brain, device)
    
    # Test sentence pairs (normal, violation, violation_word_position)
    test_pairs = [
        ("the ball rolled down the hill", "the ball rolled down the salad", 4),
        ("the cat sat on the mat", "the cat sat on the fish", 4),
        ("the bird flew in the sky", "the bird flew in the rock", 4),
        ("the dog ran in the park", "the dog ran in the moon", 4),
        ("the sun shone bright", "the sun shone cloud", 2),
    ]
    
    results = []
    spikes = []
    
    print("\n" + "=" * 70)
    print("N400-LIKE PREDICTION ERROR ANALYSIS")
    print("=" * 70)
    
    for normal, violation, viol_pos in test_pairs:
        result = analyzer.analyze_sentence_pair(normal, violation, viol_pos)
        results.append(result)
        spikes.append(result['spike_magnitude'])
        
        print(f"\nNormal: '{normal}'")
        print(f"Violation: '{violation}'")
        print(f"  Surprise at '{result['violation_words'][result['violation_position']]}' (violation point):")
        print(f"    Normal:    {result['surprise_at_violation_normal']:.4f}")
        print(f"    Violation: {result['surprise_at_violation_anomaly']:.4f}")
        print(f"    Spike:     {result['spike_magnitude']:.4f}")
    
    # Statistics
    mean_spike = np.mean(spikes)
    std_spike = np.std(spikes)
    positive_spikes = sum(1 for s in spikes if s > 0)
    
    print("\n" + "-" * 70)
    print("SUMMARY STATISTICS")
    print("-" * 70)
    print(f"Mean spike magnitude: {mean_spike:.4f} ± {std_spike:.4f}")
    print(f"Positive spikes: {positive_spikes}/{len(spikes)} ({100*positive_spikes/len(spikes):.0f}%)")
    
    # Success: majority of violations should show positive spike
    success = positive_spikes > len(spikes) / 2
    
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA:")
    print(f"  Majority positive spikes: {'✓ PASS' if success else '✗ FAIL'}")
    print("=" * 70)
    
    return {
        'results': results,
        'spikes': spikes,
        'mean_spike': mean_spike,
        'std_spike': std_spike,
        'positive_ratio': positive_spikes / len(spikes),
        'success': success
    }


def create_n400_figure(results: Dict, save_path: str = 'figures/n400_analysis.png'):
    """Create publication-quality figure for N400 analysis."""
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for i, result in enumerate(results['results'][:5]):
        ax = axes[i]
        
        normal_s = result['normal_surprise']
        viol_s = result['violation_surprise']
        words = result['normal_words']
        viol_pos = result['violation_position']
        
        x = range(len(normal_s))
        ax.plot(x, normal_s, 'b-o', label='Normal', markersize=4)
        ax.plot(x, viol_s, 'r-s', label='Violation', markersize=4)
        ax.axvline(x=viol_pos, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(words[:len(x)], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Surprise')
        ax.set_title(f"Spike: {result['spike_magnitude']:.3f}")
        ax.legend(fontsize=8)
    
    # Summary in last subplot
    ax = axes[5]
    spikes = results['spikes']
    colors = ['green' if s > 0 else 'red' for s in spikes]
    ax.bar(range(len(spikes)), spikes, color=colors)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Sentence Pair')
    ax.set_ylabel('Spike Magnitude')
    ax.set_title(f"N400-like Spikes (Mean: {results['mean_spike']:.3f})")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nFigure saved to {save_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, 
                       help='Path to trained brain checkpoint')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Create or load brain
    config = {
        'd_obs': 9,
        'd_z': 64,
        'd_sel': 32,
        'd_act': 4,
        'use_language': True,
        'vocab_size': 50,
        'd_lang_embed': 32,
        'd_lang_hidden': 64,
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': False,
        'use_cerebellum': True,
        'device': args.device,
    }
    
    device = args.device if torch.cuda.is_available() else 'cpu'
    brain = DigitalBrain(config).to(device)
    
    if args.checkpoint:
        brain.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print(f"Loaded checkpoint: {args.checkpoint}")
    
    # Run analysis
    results = run_n400_analysis(brain, device)
    
    # Create figure
    create_n400_figure(results)
    
    # Save results
    results_clean = {k: v for k, v in results.items() if k != 'results'}
    results_clean['spikes'] = [float(s) for s in results_clean['spikes']]
    
    with open('experiments/n400_results.json', 'w') as f:
        json.dump(results_clean, f, indent=2)
    print("Results saved to experiments/n400_results.json")
