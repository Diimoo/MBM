#!/usr/bin/env python3
"""
CORRECT Continual Learning Protocol for measuring catastrophic forgetting.

Protocol:
1. Train on Task A → measure SR_A_before
2. Train on Task B → measure SR_B  
3. Test on Task A again → measure SR_A_after
4. CFI = (SR_A_before - SR_A_after) / SR_A_before

This is the CORRECT way to measure forgetting, not forward transfer.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
import json
from datetime import datetime

sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_mbm, train_ppo_baseline, evaluate_vectorized


class SimplePPO(nn.Module):
    """Simple PPO baseline for comparison."""
    def __init__(self, d_obs, d_hidden, d_act):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(d_obs, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(d_hidden, d_act)
        self.value_head = nn.Linear(d_hidden, 1)
    
    def forward(self, x):
        h = self.shared(x)
        return self.policy_head(h), self.value_head(h)
    
    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)


def measure_forgetting(model, env_A, env_B, config, device, optimizer, 
                       train_fn, is_mbm=False, verbose=True):
    """
    Measure catastrophic forgetting with CORRECT protocol.
    
    Returns dict with:
    - sr_A_before: SR on task A after training on A
    - sr_B: SR on task B after training on B
    - sr_A_after: SR on task A after training on B (key metric!)
    - cfi: Catastrophic Forgetting Index
    - forward_transfer: How well B training helps A
    """
    # Phase 1: Train on Task A
    if verbose:
        print("  Phase 1: Training on Task A...")
    
    if is_mbm:
        model.reset(config['num_envs'], device=device)
    
    train_fn(env_A, model, optimizer, config, device, verbose=False, eval_env=env_A)
    
    # Measure performance on A after A training
    sr_A_before = evaluate_vectorized(model, env_A, device, episodes=128)
    if verbose:
        print(f"    SR_A (after A training): {sr_A_before:.3f}")
    
    # Phase 2: Train on Task B
    if verbose:
        print("  Phase 2: Training on Task B...")
    
    train_fn(env_B, model, optimizer, config, device, verbose=False, eval_env=env_B)
    
    # Measure performance on B
    sr_B = evaluate_vectorized(model, env_B, device, episodes=128)
    if verbose:
        print(f"    SR_B (after B training): {sr_B:.3f}")
    
    # Phase 3: RE-TEST on Task A (THE KEY STEP!)
    sr_A_after = evaluate_vectorized(model, env_A, device, episodes=128)
    if verbose:
        print(f"    SR_A (after B training): {sr_A_after:.3f}")
    
    # Compute CFI
    if sr_A_before > 0:
        cfi = (sr_A_before - sr_A_after) / sr_A_before
    else:
        cfi = 0.0
    
    forward_transfer = sr_A_after - sr_A_before  # Negative = forgetting
    
    if verbose:
        print(f"    CFI: {cfi:.3f} (0=no forgetting, 1=total forgetting)")
        print(f"    Forward Transfer: {forward_transfer:+.3f}")
    
    return {
        'sr_A_before': sr_A_before,
        'sr_B': sr_B,
        'sr_A_after': sr_A_after,
        'cfi': cfi,
        'forward_transfer': forward_transfer
    }


def run_continual_learning_experiment(n_seeds=5, updates_per_task=100):
    """Run the correct continual learning protocol across multiple seeds."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{'='*70}")
    print(f"CORRECT CONTINUAL LEARNING EXPERIMENT")
    print(f"Protocol: Train A → Test A → Train B → Test A again (measure forgetting)")
    print(f"Device: {device}, Seeds: {n_seeds}, Updates/task: {updates_per_task}")
    print(f"{'='*70}\n")
    
    # Task configs
    tasks = [
        ('5x5', 5),
        ('7x7', 7),
        ('10x10', 10)
    ]
    
    results = {
        'mbm': [],
        'ppo': [],
        'config': {
            'n_seeds': n_seeds,
            'updates_per_task': updates_per_task,
            'tasks': [t[0] for t in tasks]
        }
    }
    
    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        
        # Set random seeds
        torch.manual_seed(seed + 100)  # Offset to avoid overlap with debug
        np.random.seed(seed + 100)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 100)
        
        # --- MBM Setup ---
        mbm_config = {
            'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
            'layer_sizes': [256, 512, 256],
            'lr': 3e-4, 'seed': seed, 'num_envs': 64, 'num_steps': 64,
            'ppo_epochs': 4, 'mini_batch_size': 2048, 'gamma': 0.99, 'gae_lambda': 0.95,
            'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'total_updates': updates_per_task,
            'use_hippocampus': True, 'use_plasticity': True,
            'use_memory_policy': True, 'use_cerebellum': True,
            'sparse_cortex': False
        }
        
        mbm = DigitalBrain(mbm_config).to(device)
        mbm_opt = optim.Adam(mbm.parameters(), lr=mbm_config['lr'])
        
        # --- PPO Baseline Setup (same hidden dim for fairness) ---
        ppo_config = mbm_config.copy()
        ppo = SimplePPO(d_obs=9, d_hidden=512, d_act=4).to(device)
        ppo_opt = optim.Adam(ppo.parameters(), lr=ppo_config['lr'])
        
        # Create environments for all tasks
        envs = {
            name: TorchVectorPOMDP(num_envs=64, size=sz, device=device, seed=seed)
            for name, sz in tasks
        }
        
        seed_results_mbm = {'seed': seed, 'experiments': []}
        seed_results_ppo = {'seed': seed, 'experiments': []}
        
        # Run A→B forgetting experiments for each task pair
        for i, (task_A_name, task_A_size) in enumerate(tasks[:-1]):
            for task_B_name, task_B_size in tasks[i+1:]:
                print(f"\n--- Experiment: {task_A_name} → {task_B_name} ---")
                
                # Reset models for each experiment
                torch.manual_seed(seed + 100)
                mbm_fresh = DigitalBrain(mbm_config).to(device)
                mbm_opt_fresh = optim.Adam(mbm_fresh.parameters(), lr=mbm_config['lr'])
                
                ppo_fresh = SimplePPO(d_obs=9, d_hidden=512, d_act=4).to(device)
                ppo_opt_fresh = optim.Adam(ppo_fresh.parameters(), lr=ppo_config['lr'])
                
                # Fresh envs
                env_A = TorchVectorPOMDP(num_envs=64, size=task_A_size, device=device, seed=seed)
                env_B = TorchVectorPOMDP(num_envs=64, size=task_B_size, device=device, seed=seed)
                
                print(f"\n  [MBM]")
                mbm_result = measure_forgetting(
                    mbm_fresh, env_A, env_B, mbm_config, device, mbm_opt_fresh,
                    train_fn=train_mbm, is_mbm=True, verbose=True
                )
                mbm_result['task_A'] = task_A_name
                mbm_result['task_B'] = task_B_name
                seed_results_mbm['experiments'].append(mbm_result)
                
                print(f"\n  [PPO Baseline]")
                ppo_result = measure_forgetting(
                    ppo_fresh, env_A, env_B, ppo_config, device, ppo_opt_fresh,
                    train_fn=train_ppo_baseline, is_mbm=False, verbose=True
                )
                ppo_result['task_A'] = task_A_name
                ppo_result['task_B'] = task_B_name
                seed_results_ppo['experiments'].append(ppo_result)
        
        results['mbm'].append(seed_results_mbm)
        results['ppo'].append(seed_results_ppo)
    
    # Aggregate results
    print(f"\n{'='*70}")
    print("AGGREGATED RESULTS")
    print(f"{'='*70}")
    
    def aggregate_metric(results_list, metric):
        values = []
        for seed_data in results_list:
            for exp in seed_data['experiments']:
                values.append(exp[metric])
        return np.mean(values), np.std(values)
    
    # CFI comparison
    mbm_cfi_mean, mbm_cfi_std = aggregate_metric(results['mbm'], 'cfi')
    ppo_cfi_mean, ppo_cfi_std = aggregate_metric(results['ppo'], 'cfi')
    
    print(f"\nCatastrophic Forgetting Index (CFI):")
    print(f"  MBM: {mbm_cfi_mean:.3f} ± {mbm_cfi_std:.3f}")
    print(f"  PPO: {ppo_cfi_mean:.3f} ± {ppo_cfi_std:.3f}")
    
    if ppo_cfi_mean > 0:
        ratio = ppo_cfi_mean / max(mbm_cfi_mean, 0.001)
        print(f"  PPO has {ratio:.1f}x more forgetting than MBM")
    
    # Per-experiment breakdown
    print(f"\nPer-Experiment Breakdown:")
    experiment_types = set()
    for seed_data in results['mbm']:
        for exp in seed_data['experiments']:
            experiment_types.add((exp['task_A'], exp['task_B']))
    
    for task_A, task_B in sorted(experiment_types):
        mbm_cfis = []
        ppo_cfis = []
        for seed_data in results['mbm']:
            for exp in seed_data['experiments']:
                if exp['task_A'] == task_A and exp['task_B'] == task_B:
                    mbm_cfis.append(exp['cfi'])
        for seed_data in results['ppo']:
            for exp in seed_data['experiments']:
                if exp['task_A'] == task_A and exp['task_B'] == task_B:
                    ppo_cfis.append(exp['cfi'])
        
        print(f"  {task_A} → {task_B}:")
        print(f"    MBM CFI: {np.mean(mbm_cfis):.3f} ± {np.std(mbm_cfis):.3f}")
        print(f"    PPO CFI: {np.mean(ppo_cfis):.3f} ± {np.std(ppo_cfis):.3f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiments/continual_learning_proper_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--updates", type=int, default=100, help="Updates per task")
    args = parser.parse_args()
    
    run_continual_learning_experiment(n_seeds=args.seeds, updates_per_task=args.updates)
