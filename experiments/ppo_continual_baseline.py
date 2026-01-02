import torch
import torch.optim as optim
import numpy as np
import os
import sys
import json

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_ppo_baseline
from experiments.compare_mbm_vs_ppo import PPOBaseline
from eval_current_state import eval_on_task

def run_ppo_continual_learning():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Starting PPO Continual Learning Baseline...")
    
    config = {
        'd_obs': 9, 'd_h': 512, 'd_act': 4,
        'lr': 3e-4, 'num_envs': 128, 'num_steps': 128,
        'ppo_epochs': 4, 'mini_batch_size': 4096, 'gamma': 0.99, 'gae_lambda': 0.95,
        'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
        'total_updates': 100
    }
    
    model = PPOBaseline(config['d_obs'], config['d_h'], config['d_act']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # Tasks
    envs = {
        '5x5': TorchVectorPOMDP(num_envs=config['num_envs'], size=5, device=device),
        '7x7': TorchVectorPOMDP(num_envs=config['num_envs'], size=7, device=device),
        '10x10': TorchVectorPOMDP(num_envs=config['num_envs'], size=10, device=device),
    }
    
    task_names = ['5x5', '7x7', '10x10']
    matrix = []
    
    for train_task in task_names:
        print(f"\n--- Training PPO on Gridworld {train_task} ---")
        train_ppo_baseline(envs[train_task], model, optimizer, config, device, verbose=True)
        
        # Evaluate on all tasks
        phase_results = []
        for eval_task in task_names:
            sr = eval_on_task(model, f'gridworld_{eval_task}', device)
            phase_results.append(sr)
        matrix.append(phase_results)
        print(f"PPO After {train_task} - SRs: {phase_results}")

    results = {
        'task_names': task_names,
        'matrix': matrix
    }
    
    with open("experiments/ppo_continual_results.json", "w") as f:
        json.dump(results, f)
    print("\nPPO Continual Learning Baseline complete. Results saved to experiments/ppo_continual_results.json")

if __name__ == "__main__":
    run_ppo_continual_learning()
