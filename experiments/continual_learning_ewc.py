import torch
import torch.optim as optim
import numpy as np
import os
import sys
import json
from tqdm import tqdm

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_ppo_baseline, compute_fisher, evaluate_vectorized
from experiments.compare_mbm_vs_ppo import PPOBaseline
from eval_current_state import eval_on_task

def run_ewc_continual_learning(n_seeds=5, updates_per_task=100, ewc_lambda=1000.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting PPO+EWC Continual Learning Baseline (Lambda: {ewc_lambda}) across {n_seeds} seeds...")
    
    task_names = ['5x5', '7x7', '10x10']
    task_sizes = [5, 7, 10]
    
    results = {
        'ppo_ewc': []  # list of matrices (3x3)
    }
    
    for seed in range(n_seeds):
        print(f"\n--- SEED {seed} ---")
        
        config = {
            'd_obs': 9, 'd_h': 512, 'd_act': 4,
            'lr': 3e-4, 'seed': seed, 'num_envs': 128, 'num_steps': 128,
            'ppo_epochs': 4, 'mini_batch_size': 4096, 'gamma': 0.99, 'gae_lambda': 0.95,
            'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'total_updates': updates_per_task,
            'ewc_lambda': ewc_lambda
        }
        
        model = PPOBaseline(config['d_obs'], config['d_h'], config['d_act']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        
        envs = {
            name: TorchVectorPOMDP(num_envs=config['num_envs'], size=sz, device=device, seed=seed)
            for name, sz in zip(task_names, task_sizes)
        }
        
        matrix = []
        ewc_data = None # Will store Fisher and Params for consolidated tasks
        
        for phase, train_task in enumerate(task_names):
            print(f"Phase {phase+1}: Training PPO+EWC on {train_task}...")
            
            # Train with EWC
            train_ppo_baseline(envs[train_task], model, optimizer, config, device, verbose=False, ewc_data=ewc_data)
            
            # Evaluate on all tasks
            phase_sr = [eval_on_task(model, f'gridworld_{t}', device, episodes=64) for t in task_names]
            matrix.append(phase_sr)
            print(f"  SRs: {phase_sr}")
            
            # After training, update EWC data (Fisher + Params)
            # For simplicity, we compute Fisher for the *current* task and add it to a running average or replace
            # Here we follow standard EWC: sum of Fishers from previous tasks
            new_fisher, new_params = compute_fisher(model, envs[train_task], device, num_samples=1024)
            
            if ewc_data is None:
                ewc_data = {'fisher': new_fisher, 'params': new_params}
            else:
                # Merge Fisher (average or sum)
                for name in ewc_data['fisher']:
                    ewc_data['fisher'][name] = (ewc_data['fisher'][name] + new_fisher[name]) / 2.0
                    # For params, we usually anchor to the latest consolidated weights
                    ewc_data['params'][name] = new_params[name]
                    
        results['ppo_ewc'].append(matrix)
        
    # Save results
    with open("experiments/ewc_results.json", "w") as f:
        json.dump(results, f)
    
    print("\nEWC Baseline complete. Results saved to experiments/ewc_results.json")

if __name__ == "__main__":
    run_ewc_continual_learning(n_seeds=5, updates_per_task=100, ewc_lambda=1000.0)
