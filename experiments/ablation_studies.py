import torch
import torch.optim as optim
import numpy as np
import os
import sys
import json

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_mbm, evaluate_vectorized
from experiments.compare_mbm_vs_ppo import get_env

def run_ablations(n_seeds=3, device="cuda"):
    """Run ablation studies to determine component importance."""
    print(f"Starting ablation studies over {n_seeds} seeds...")
    
    ablation_configs = {
        'mbm_full': {
            'use_hippocampus': True,
            'use_plasticity': True,
            'use_memory_policy': True,
            'use_cerebellum': True,
        },
        'ablation_no_hippo': {
            'use_hippocampus': False,
            'use_plasticity': True,
            'use_memory_policy': False,
            'use_cerebellum': True,
        },
        'ablation_no_plasticity': {
            'use_hippocampus': True,
            'use_plasticity': False,
            'use_memory_policy': True,
            'use_cerebellum': True,
        },
        'ablation_no_cerebellum': {
            'use_hippocampus': True,
            'use_plasticity': True,
            'use_memory_policy': True,
            'use_cerebellum': False,
        },
        'ablation_minimal': {
            'use_hippocampus': False,
            'use_plasticity': False,
            'use_memory_policy': False,
            'use_cerebellum': False,
        },
    }
    
    task_name = 'gridworld_5x5'
    num_envs = 512
    eval_envs = 64
    n_updates = 100
    
    all_results = {}
    
    for config_name, flags in ablation_configs.items():
        print(f"\n--- Running Ablation: {config_name} ---")
        srs = []
        for seed in range(n_seeds):
            train_env, d_obs, d_act = get_env(task_name, num_envs, device)
            eval_env, _, _ = get_env(task_name, eval_envs, device)
            
            brain_config = {
                'd_obs': d_obs, 'd_z': 512, 'd_sel': 64, 'd_act': d_act,
                'lr': 3e-4, 'seed': seed,
                'sparse_cortex': False,
                'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
                'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
                'ppo_epochs': 4, 'mini_batch_size': 16384,
                'num_envs': num_envs, 'num_steps': 128,
                'total_updates': n_updates,
                **flags
            }
            
            brain = DigitalBrain(brain_config).to(device)
            optimizer = optim.Adam(brain.parameters(), lr=brain_config['lr'])
            
            train_mbm(train_env, brain, optimizer, brain_config, device, verbose=False)
            sr = evaluate_vectorized(brain, eval_env, device)
            srs.append(sr)
            print(f"Seed {seed} SR: {sr:.3f}")
            
        all_results[config_name] = {
            'mean': float(np.mean(srs)),
            'std': float(np.std(srs)),
            'all': srs
        }
        print(f"Result for {config_name}: {all_results[config_name]['mean']:.3f} +/- {all_results[config_name]['std']:.3f}")

    with open("experiments/ablation_results.json", "w") as f:
        json.dump(all_results, f)
    print("\nAblation studies complete. Results saved to experiments/ablation_results.json")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_ablations(n_seeds=3, device=device)
