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
from experiments.train_utils import train_mbm, train_ppo_baseline
from experiments.compare_mbm_vs_ppo import PPOBaseline
from eval_current_state import eval_on_task

def run_consolidation_study(n_seeds=10, updates_per_task=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Consolidation Effect (Backward Transfer) Validation across {n_seeds} seeds...")
    
    task_names = ['5x5', '7x7', '10x10']
    task_sizes = [5, 7, 10]
    
    results = {
        'mbm': [], # list of matrices (3x3)
        'ppo': []  # list of matrices (3x3)
    }
    
    for seed in range(n_seeds):
        print(f"\n--- SEED {seed} ---")
        
        # --- MBM Setup ---
        mbm_config = {
            'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
            'lr': 3e-4, 'seed': seed, 'num_envs': 128, 'num_steps': 128,
            'ppo_epochs': 4, 'mini_batch_size': 4096, 'gamma': 0.99, 'gae_lambda': 0.95,
            'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'total_updates': updates_per_task,
            'use_hippocampus': True, 'use_plasticity': True, 'use_memory_policy': True, 'use_cerebellum': True,
            'sparse_cortex': False
        }
        mbm_brain = DigitalBrain(mbm_config).to(device)
        mbm_opt = optim.Adam(mbm_brain.parameters(), lr=mbm_config['lr'])
        
        # --- PPO Setup ---
        ppo_config = mbm_config.copy()
        ppo_model = PPOBaseline(ppo_config['d_obs'], 512, ppo_config['d_act']).to(device)
        ppo_opt = optim.Adam(ppo_model.parameters(), lr=ppo_config['lr'])
        
        # Environments
        envs = {
            name: TorchVectorPOMDP(num_envs=mbm_config['num_envs'], size=sz, device=device, seed=seed)
            for name, sz in zip(task_names, task_sizes)
        }
        
        mbm_matrix = []
        ppo_matrix = []
        
        for phase, train_task in enumerate(task_names):
            print(f"Phase {phase+1}: Training on {train_task}...")
            
            # Train MBM
            train_mbm(envs[train_task], mbm_brain, mbm_opt, mbm_config, device, verbose=False)
            # Evaluate MBM on all tasks
            mbm_phase_sr = [eval_on_task(mbm_brain, f'gridworld_{t}', device, episodes=64) for t in task_names]
            mbm_matrix.append(mbm_phase_sr)
            
            # Train PPO
            train_ppo_baseline(envs[train_task], ppo_model, ppo_opt, ppo_config, device, verbose=False)
            # Evaluate PPO on all tasks
            ppo_phase_sr = [eval_on_task(ppo_model, f'gridworld_{t}', device, episodes=64) for t in task_names]
            ppo_matrix.append(ppo_phase_sr)
            
            print(f"  MBM SRs: {mbm_phase_sr}")
            print(f"  PPO SRs: {ppo_phase_sr}")
            
        results['mbm'].append(mbm_matrix)
        results['ppo'].append(ppo_matrix)
        
    # Save results
    with open("experiments/consolidation_results.json", "w") as f:
        json.dump(results, f)
    
    print("\nValidation complete. Results saved to experiments/consolidation_results.json")

if __name__ == "__main__":
    run_consolidation_study(n_seeds=10, updates_per_task=100)
