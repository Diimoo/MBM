import torch
import torch.optim as optim
import numpy as np
import os
import sys
import json
from tqdm import tqdm

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_mbm, evaluate_vectorized
from eval_current_state import eval_on_task

def run_hierarchical_consolidation(n_seeds=5, updates_per_task=150):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Hierarchical MBM Consolidation Validation across {n_seeds} seeds...")
    
    task_names = ['5x5', '7x7', '10x10']
    task_sizes = [5, 7, 10]
    
    results = {
        'mbm_hier': [] # list of matrices (3x3)
    }
    
    for seed in range(n_seeds):
        print(f"\n--- SEED {seed} ---")
        
        # --- Hierarchical MBM Setup ---
        # 3-layer architecture as suggested in PRODUCTION_PLAN.md
        mbm_config = {
            'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
            'layer_sizes': [256, 512, 256], # L1 -> L2 -> L3
            'lr': 3e-4, 'seed': seed, 'num_envs': 128, 'num_steps': 128,
            'ppo_epochs': 4, 'mini_batch_size': 4096, 'gamma': 0.99, 'gae_lambda': 0.95,
            'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'total_updates': updates_per_task,
            'use_hippocampus': True, 'use_plasticity': True, 'use_memory_policy': True, 'use_cerebellum': True,
            'sparse_cortex': False
        }
        
        # Note: d_z in config will be overridden by the last layer size in HierarchicalCortex
        brain = DigitalBrain(mbm_config).to(device)
        optimizer = optim.Adam(brain.parameters(), lr=mbm_config['lr'])
        
        # Environments
        envs = {
            name: TorchVectorPOMDP(num_envs=mbm_config['num_envs'], size=sz, device=device, seed=seed)
            for name, sz in zip(task_names, task_sizes)
        }
        
        matrix = []
        
        for phase, train_task in enumerate(task_names):
            print(f"Phase {phase+1}: Training on {train_task}...")
            
            # Train Hierarchical MBM
            train_mbm(envs[train_task], brain, optimizer, mbm_config, device, verbose=False)
            
            # Evaluate on all tasks
            phase_sr = [eval_on_task(brain, f'gridworld_{t}', device, episodes=64) for t in task_names]
            matrix.append(phase_sr)
            print(f"  SRs: {phase_sr}")
            
        results['mbm_hier'].append(matrix)
        
    # Save results
    with open("experiments/hierarchical_results.json", "w") as f:
        json.dump(results, f)
    
    print("\nHierarchical MBM Validation complete. Results saved to experiments/hierarchical_results.json")

if __name__ == "__main__":
    run_hierarchical_consolidation(n_seeds=5, updates_per_task=150)
