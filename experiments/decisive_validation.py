#!/usr/bin/env python3
"""
Decisive Validation Experiment: Stabilized MBM vs PPO

This experiment determines whether stabilization fixes (homeostatic plasticity,
selective hippocampal retrieval, metaplasticity) reduce MBM variance enough
to be competitive with PPO.

Decision Criteria:
- If MBM variance < 0.20: Continue Path A (stabilization worked)
- If MBM variance > 0.30: Pivot to Path B (tradeoff story)
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
                       train_fn, is_mbm=False):
    """Measure CFI with correct protocol: Train A → Test A → Train B → Test A again."""
    
    if is_mbm:
        model.reset(config['num_envs'], device=device)
    
    # Phase 1: Train on Task A
    train_fn(env_A, model, optimizer, config, device, verbose=False, eval_env=env_A)
    sr_A_before = evaluate_vectorized(model, env_A, device, episodes=128)
    
    # Phase 2: Train on Task B
    train_fn(env_B, model, optimizer, config, device, verbose=False, eval_env=env_B)
    sr_B = evaluate_vectorized(model, env_B, device, episodes=128)
    
    # Phase 3: RE-TEST on Task A
    sr_A_after = evaluate_vectorized(model, env_A, device, episodes=128)
    
    # Compute CFI
    if sr_A_before > 0.01:
        cfi = (sr_A_before - sr_A_after) / sr_A_before
    else:
        cfi = 0.0
    
    return {
        'sr_A_before': sr_A_before,
        'sr_B': sr_B,
        'sr_A_after': sr_A_after,
        'cfi': cfi,
    }


def run_decisive_validation(n_seeds=10, updates_per_task=100):
    """
    Run decisive validation with 10 seeds.
    Determines: Path A (stabilization worked) or Path B (pivot story).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("="*70)
    print("DECISIVE VALIDATION EXPERIMENT")
    print("="*70)
    print(f"Device: {device}")
    print(f"Seeds: {n_seeds}")
    print(f"Updates per task: {updates_per_task}")
    print()
    print("Stabilization Features Enabled:")
    print("  ✓ Homeostatic weight regulation (adaptive LR)")
    print("  ✓ Soft weight clipping (tanh-based)")
    print("  ✓ Metaplasticity (history-based LR scaling)")
    print("  ✓ Selective hippocampal retrieval (confidence=0.5)")
    print("="*70)
    
    # Task pairs to test
    task_pairs = [
        ('5x5', 5, '7x7', 7),
        ('5x5', 5, '10x10', 10),
        ('7x7', 7, '10x10', 10),
    ]
    
    results = {
        'mbm_stabilized': [],
        'ppo_baseline': [],
        'config': {
            'n_seeds': n_seeds,
            'updates_per_task': updates_per_task,
            'stabilization': {
                'homeostatic_plasticity': True,
                'soft_clipping': True,
                'metaplasticity': True,
                'hip_confidence_threshold': 0.5,
            }
        }
    }
    
    for seed in range(n_seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed}")
        print(f"{'='*50}")
        
        # Set random seeds
        torch.manual_seed(seed + 200)
        np.random.seed(seed + 200)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + 200)
        
        seed_results_mbm = {'seed': seed, 'experiments': []}
        seed_results_ppo = {'seed': seed, 'experiments': []}
        
        for task_A_name, task_A_size, task_B_name, task_B_size in task_pairs:
            print(f"\n--- {task_A_name} → {task_B_name} ---")
            
            # --- Stabilized MBM ---
            torch.manual_seed(seed + 200)
            mbm_config = {
                'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
                'layer_sizes': [256, 512, 256],
                'lr': 3e-4, 'seed': seed, 'num_envs': 64, 'num_steps': 64,
                'ppo_epochs': 4, 'mini_batch_size': 2048, 'gamma': 0.99, 'gae_lambda': 0.95,
                'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
                'total_updates': updates_per_task,
                'use_hippocampus': True, 'use_plasticity': True,
                'use_memory_policy': True, 'use_cerebellum': True,
                'sparse_cortex': False,
                # Stabilization config
                'hip_confidence_threshold': 0.5,
            }
            
            mbm = DigitalBrain(mbm_config).to(device)
            mbm_opt = optim.Adam(mbm.parameters(), lr=mbm_config['lr'])
            
            env_A = TorchVectorPOMDP(num_envs=64, size=task_A_size, device=device, seed=seed)
            env_B = TorchVectorPOMDP(num_envs=64, size=task_B_size, device=device, seed=seed)
            
            mbm_result = measure_forgetting(
                mbm, env_A, env_B, mbm_config, device, mbm_opt,
                train_fn=train_mbm, is_mbm=True
            )
            mbm_result['task_A'] = task_A_name
            mbm_result['task_B'] = task_B_name
            seed_results_mbm['experiments'].append(mbm_result)
            print(f"  MBM: SR_A={mbm_result['sr_A_before']:.3f}→{mbm_result['sr_A_after']:.3f}, CFI={mbm_result['cfi']:+.3f}")
            
            # --- PPO Baseline ---
            torch.manual_seed(seed + 200)
            ppo_config = mbm_config.copy()
            ppo = SimplePPO(d_obs=9, d_hidden=512, d_act=4).to(device)
            ppo_opt = optim.Adam(ppo.parameters(), lr=ppo_config['lr'])
            
            env_A = TorchVectorPOMDP(num_envs=64, size=task_A_size, device=device, seed=seed)
            env_B = TorchVectorPOMDP(num_envs=64, size=task_B_size, device=device, seed=seed)
            
            ppo_result = measure_forgetting(
                ppo, env_A, env_B, ppo_config, device, ppo_opt,
                train_fn=train_ppo_baseline, is_mbm=False
            )
            ppo_result['task_A'] = task_A_name
            ppo_result['task_B'] = task_B_name
            seed_results_ppo['experiments'].append(ppo_result)
            print(f"  PPO: SR_A={ppo_result['sr_A_before']:.3f}→{ppo_result['sr_A_after']:.3f}, CFI={ppo_result['cfi']:+.3f}")
        
        results['mbm_stabilized'].append(seed_results_mbm)
        results['ppo_baseline'].append(seed_results_ppo)
    
    # Aggregate and analyze
    print("\n" + "="*70)
    print("AGGREGATED RESULTS")
    print("="*70)
    
    def get_all_cfis(results_list):
        cfis = []
        for seed_data in results_list:
            for exp in seed_data['experiments']:
                cfis.append(exp['cfi'])
        return np.array(cfis)
    
    mbm_cfis = get_all_cfis(results['mbm_stabilized'])
    ppo_cfis = get_all_cfis(results['ppo_baseline'])
    
    mbm_mean, mbm_std = np.mean(mbm_cfis), np.std(mbm_cfis)
    ppo_mean, ppo_std = np.mean(ppo_cfis), np.std(ppo_cfis)
    
    print(f"\nCatastrophic Forgetting Index (CFI):")
    print(f"  Stabilized MBM: {mbm_mean:+.3f} ± {mbm_std:.3f}")
    print(f"  PPO Baseline:   {ppo_mean:+.3f} ± {ppo_std:.3f}")
    
    # Per-task breakdown
    print(f"\nPer-Task Breakdown:")
    for task_A, _, task_B, _ in task_pairs:
        mbm_task_cfis = [exp['cfi'] for sd in results['mbm_stabilized'] 
                         for exp in sd['experiments'] 
                         if exp['task_A'] == task_A and exp['task_B'] == task_B]
        ppo_task_cfis = [exp['cfi'] for sd in results['ppo_baseline'] 
                         for exp in sd['experiments'] 
                         if exp['task_A'] == task_A and exp['task_B'] == task_B]
        
        print(f"  {task_A} → {task_B}:")
        print(f"    MBM: {np.mean(mbm_task_cfis):+.3f} ± {np.std(mbm_task_cfis):.3f}")
        print(f"    PPO: {np.mean(ppo_task_cfis):+.3f} ± {np.std(ppo_task_cfis):.3f}")
    
    # Decision
    print("\n" + "="*70)
    print("DECISION")
    print("="*70)
    
    variance_ratio = mbm_std / max(ppo_std, 0.01)
    
    if mbm_std < 0.20:
        decision = "PATH_A"
        print(f"✅ STABILIZATION SUCCESSFUL (variance = {mbm_std:.3f} < 0.20)")
        print("   → Continue with original hypothesis")
        print("   → MBM can claim improved stability")
    elif mbm_std < 0.30:
        decision = "PATH_A_MARGINAL"
        print(f"⚠️ MARGINAL SUCCESS (variance = {mbm_std:.3f}, target < 0.20)")
        print("   → Consider more stabilization work OR")
        print("   → Proceed with caveats about variance")
    else:
        decision = "PATH_B"
        print(f"❌ STABILIZATION INSUFFICIENT (variance = {mbm_std:.3f} > 0.30)")
        print("   → Pivot to stability-plasticity tradeoff story")
        print("   → Focus on conditions where MBM excels")
    
    print(f"\nVariance Ratio (MBM/PPO): {variance_ratio:.2f}x")
    
    if mbm_mean < ppo_mean:
        print(f"MBM shows LESS forgetting on average ({mbm_mean:+.3f} vs {ppo_mean:+.3f})")
    else:
        print(f"PPO shows LESS forgetting on average ({ppo_mean:+.3f} vs {mbm_mean:+.3f})")
    
    # Save results
    results['summary'] = {
        'mbm_cfi_mean': float(mbm_mean),
        'mbm_cfi_std': float(mbm_std),
        'ppo_cfi_mean': float(ppo_mean),
        'ppo_cfi_std': float(ppo_std),
        'variance_ratio': float(variance_ratio),
        'decision': decision,
    }
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"experiments/decisive_validation_{timestamp}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument("--updates", type=int, default=100, help="Updates per task")
    args = parser.parse_args()
    
    run_decisive_validation(n_seeds=args.seeds, updates_per_task=args.updates)
