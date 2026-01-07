#!/usr/bin/env python3
"""
Diagnostic script to identify which seeds fail and why.
Run this BEFORE any training to identify fundamentally broken seeds.
"""
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.insert(0, os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs


def diagnose_seed(seed, steps=100, config=None, verbose=True):
    """Run seed for N steps, log everything."""
    if verbose:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
    
    if config is None:
        config = {
            'd_obs': 9, 'd_z': 256, 'd_sel': 32, 'd_act': 4,
            'layer_sizes': [128, 256, 128],
            'use_hippocampus': True,
            'use_plasticity': True,
            'use_memory_policy': True,
            'use_cerebellum': True,
            'sparse_cortex': False,
            'hip_confidence_threshold': 0.5,
        }
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    brain = DigitalBrain(config)
    brain.reset(batch_size=1)
    
    # CHECK INITIAL WEIGHTS
    if verbose:
        print(f"\nInitial Conditions:")
        if hasattr(brain.cortex.microcircuit, 'layers'):
            for i, layer in enumerate(brain.cortex.microcircuit.layers):
                print(f"  Layer {i} W_ee: norm={layer.W_ee.norm():.4f}, max={layer.W_ee.abs().max():.4f}")
        else:
            print(f"  W_ee: norm={brain.cortex.microcircuit.W_ee.norm():.4f}, max={brain.cortex.microcircuit.W_ee.abs().max():.4f}")
        print(f"  Policy weight norm: {brain.bg.policy_head.weight.norm():.4f}")
        print(f"  Value weight norm: {brain.bg.value_head.weight.norm():.4f}")
    
    failed = False
    failure_step = -1
    failure_reason = ""
    
    # Random obs sequence
    for step in range(steps):
        obs_data = torch.randn(1, config['d_obs']) * 0.5
        obs = Obs(x=obs_data)
        
        # Random reward/done
        reward = torch.zeros(1, 1)
        if step > 0 and np.random.random() < 0.1:
            reward = torch.tensor([[1.0]]) if np.random.random() < 0.5 else torch.tensor([[-0.1]])
        done = torch.zeros(1, 1, dtype=torch.bool)
        
        try:
            action, log_prob, value, state, log, entropy = brain.step(obs, reward, done, learn=True)
            
            # CHECK FOR PROBLEMS
            if torch.isnan(value).any():
                if verbose:
                    print(f"\n❌ NaN in value at step {step}")
                failed = True
                failure_step = step
                failure_reason = "NaN in value"
                break
            
            if torch.isnan(log_prob).any():
                if verbose:
                    print(f"\n❌ NaN in log_prob at step {step}")
                failed = True
                failure_step = step
                failure_reason = "NaN in log_prob"
                break
            
            if value.abs().max() > 1000:
                if verbose:
                    print(f"\n⚠️ Value explosion at step {step}: {value.item():.2f}")
                failed = True
                failure_step = step
                failure_reason = f"Value explosion: {value.item():.2f}"
                break
            
            if verbose and step % 20 == 0:
                print(f"  Step {step:3d}: value={value.item():7.2f}, DA={log.rpe:6.3f}, novelty={log.novelty:.3f}")
                
        except Exception as e:
            if verbose:
                print(f"\n❌ Exception at step {step}: {e}")
            failed = True
            failure_step = step
            failure_reason = str(e)
            break
    
    # Final conditions
    if verbose:
        print(f"\nFinal Conditions:")
        if hasattr(brain.cortex.microcircuit, 'layers'):
            for i, layer in enumerate(brain.cortex.microcircuit.layers):
                print(f"  Layer {i} W_ee: norm={layer.W_ee.norm():.4f}, max={layer.W_ee.abs().max():.4f}")
        else:
            print(f"  W_ee: norm={brain.cortex.microcircuit.W_ee.norm():.4f}, max={brain.cortex.microcircuit.W_ee.abs().max():.4f}")
    
    if failed:
        if verbose:
            print(f"\n💥 FAILED at step {failure_step}: {failure_reason}")
        return False, failure_step, failure_reason
    else:
        if verbose:
            print(f"\n✅ SURVIVED {steps} steps")
        return True, steps, "OK"


def run_diagnostics(n_seeds=10, steps=100):
    """Test multiple seeds and summarize results."""
    print("="*60)
    print("MBM STABILITY DIAGNOSTICS")
    print("="*60)
    print(f"Testing {n_seeds} seeds for {steps} steps each...")
    
    results = {}
    for seed in range(n_seeds):
        survived, last_step, reason = diagnose_seed(seed, steps=steps, verbose=True)
        results[seed] = {'survived': survived, 'last_step': last_step, 'reason': reason}
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    good_seeds = []
    bad_seeds = []
    
    for seed, result in results.items():
        if result['survived']:
            status = "✅ OK"
            good_seeds.append(seed)
        else:
            status = f"❌ FAILED at step {result['last_step']}: {result['reason'][:40]}"
            bad_seeds.append(seed)
        print(f"Seed {seed}: {status}")
    
    print(f"\n✅ Good seeds ({len(good_seeds)}): {good_seeds}")
    print(f"❌ Bad seeds ({len(bad_seeds)}): {bad_seeds}")
    print(f"\nSuccess rate: {len(good_seeds)}/{n_seeds} ({100*len(good_seeds)/n_seeds:.0f}%)")
    
    return results


def stress_test_seed(seed, steps=500, verbose=True):
    """Longer stress test for a specific seed."""
    print(f"\n{'='*60}")
    print(f"STRESS TEST: Seed {seed} for {steps} steps")
    print(f"{'='*60}")
    
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
        'layer_sizes': [256, 512, 256],
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': True,
        'use_cerebellum': True,
        'sparse_cortex': False,
        'hip_confidence_threshold': 0.5,
    }
    
    return diagnose_seed(seed, steps=steps, config=config, verbose=verbose)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to test")
    parser.add_argument("--steps", type=int, default=100, help="Steps per seed")
    parser.add_argument("--stress", type=int, default=None, help="Stress test specific seed")
    args = parser.parse_args()
    
    if args.stress is not None:
        stress_test_seed(args.stress, steps=500)
    else:
        run_diagnostics(n_seeds=args.seeds, steps=args.steps)
