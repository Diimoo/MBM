import torch
import torch.optim as optim
import numpy as np
import os
import sys
import copy

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_mbm
from eval_current_state import eval_on_task

def train_on_task(brain, env, optimizer, num_updates, device, task_name, config):
    print(f"--- Training on {task_name} for {num_updates} updates ---")
    config['total_updates'] = num_updates
    train_mbm(env, brain, optimizer, config, device, verbose=True)
    print(f"Finished training on {task_name}.")

def run_continual_learning_exp():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
        'lr': 3e-4, 'seed': 42, 'num_envs': 128, 'num_steps': 128,
        'ppo_epochs': 4, 'mini_batch_size': 4096, 'gamma': 0.99, 'gae_lambda': 0.95,
        'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
        'total_updates': 50 # Will be overridden in train_on_task
    }
    
    brain_config = config.copy()
    brain_config['sparse_cortex'] = False
    
    brain = DigitalBrain(brain_config).to(device)
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'])
    
    # Task A: Gridworld 5x5
    env_a = TorchVectorPOMDP(num_envs=config['num_envs'], size=5, device=device)
    
    # Task B: Gridworld 7x7
    env_b = TorchVectorPOMDP(num_envs=config['num_envs'], size=7, device=device)
    
    # Task C: Gridworld 10x10
    env_c = TorchVectorPOMDP(num_envs=config['num_envs'], size=10, device=device)
    
    print("--- Phase 1: Train on Task A (Gridworld 5x5) ---")
    sr_a_0 = eval_on_task(brain, 'gridworld_5x5', device)
    print(f"Initial SR A: {sr_a_0:.3f}")
    
    # Actual training on A
    train_on_task(brain, env_a, optimizer, 100, device, 'Gridworld 5x5', config)
    
    sr_a_1 = eval_on_task(brain, 'gridworld_5x5', device)
    sr_b_1 = eval_on_task(brain, 'gridworld_7x7', device)
    sr_c_1 = eval_on_task(brain, 'gridworld_10x10', device)
    print(f"After Phase 1 - SR A: {sr_a_1:.3f}, SR B: {sr_b_1:.3f}, SR C: {sr_c_1:.3f}")
    
    print("\n--- Phase 2: Train on Task B (Gridworld 7x7) ---")
    # Actual training on B
    train_on_task(brain, env_b, optimizer, 100, device, 'Gridworld 7x7', config)
    
    sr_a_2 = eval_on_task(brain, 'gridworld_5x5', device)
    sr_b_2 = eval_on_task(brain, 'gridworld_7x7', device)
    sr_c_2 = eval_on_task(brain, 'gridworld_10x10', device)
    print(f"After Phase 2 - SR A: {sr_a_2:.3f}, SR B: {sr_b_2:.3f}, SR C: {sr_c_2:.3f}")

    print("\n--- Phase 3: Train on Task C (Gridworld 10x10) ---")
    # Actual training on C
    train_on_task(brain, env_c, optimizer, 100, device, 'Gridworld 10x10', config)
    
    sr_a_3 = eval_on_task(brain, 'gridworld_5x5', device)
    sr_b_3 = eval_on_task(brain, 'gridworld_7x7', device)
    sr_c_3 = eval_on_task(brain, 'gridworld_10x10', device)
    print(f"After Phase 3 - SR A: {sr_a_3:.3f}, SR B: {sr_b_3:.3f}, SR C: {sr_c_3:.3f}")
    
    print("\n--- Continual Learning Summary Matrix ---")
    matrix = np.array([
        [sr_a_1, sr_b_1, sr_c_1],
        [sr_a_2, sr_b_2, sr_c_2],
        [sr_a_3, sr_b_3, sr_c_3]
    ])
    print(matrix)
    
    results = {
        'task_names': ['5x5', '7x7', '10x10'],
        'matrix': matrix.tolist()
    }
    import json
    with open("experiments/continual_learning_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    run_continual_learning_exp()
