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
from digital_brain.envs.t_maze import TorchVectorTMaze
from digital_brain.envs.radial_arm_maze import TorchVectorRadialArmMaze
from digital_brain.envs.cartpole import TorchVectorCartPole
from experiments.train_utils import train_mbm, evaluate_vectorized

def run_multi_task_benchmark(device="cuda"):
    print("Starting Multi-Task Benchmark Suite (Tier 1 & Tier 2)...")
    
    tasks = {
        'gridworld_5x5': {'env_class': TorchVectorPOMDP, 'env_args': {'size': 5}, 'd_obs': 9, 'd_act': 4},
        'gridworld_7x7': {'env_class': TorchVectorPOMDP, 'env_args': {'size': 7}, 'd_obs': 9, 'd_act': 4},
        'cartpole': {'env_class': TorchVectorCartPole, 'env_args': {}, 'd_obs': 4, 'd_act': 2},
        't_maze_5': {'env_class': TorchVectorTMaze, 'env_args': {'corridor_length': 5}, 'd_obs': 8, 'd_act': 3},
        'radial_arm_8': {'env_class': TorchVectorRadialArmMaze, 'env_args': {'num_arms': 8, 'arm_length': 3}, 'd_obs': 17, 'd_act': 9},
    }
    
    results = {}
    
    num_envs = 128
    eval_envs = 64
    n_updates = 100
    
    for task_name, info in tasks.items():
        print(f"\n--- Task: {task_name} ---")
        
        train_env = info['env_class'](num_envs=num_envs, device=device, **info['env_args'])
        eval_env = info['env_class'](num_envs=eval_envs, device=device, **info['env_args'])
        
        brain_config = {
            'd_obs': info['d_obs'], 'd_z': 512, 'd_sel': 64, 'd_act': info['d_act'],
            'lr': 3e-4, 'seed': 42,
            'sparse_cortex': False,
            'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'ppo_epochs': 4, 'mini_batch_size': 4096,
            'num_envs': num_envs, 'num_steps': 128,
            'total_updates': n_updates,
            'use_hippocampus': True,
            'use_plasticity': True,
            'use_memory_policy': True,
            'use_cerebellum': True
        }
        
        brain = DigitalBrain(brain_config).to(device)
        optimizer = optim.Adam(brain.parameters(), lr=brain_config['lr'])
        
        history = train_mbm(train_env, brain, optimizer, brain_config, device, verbose=False, eval_env=eval_env)
        final_sr = evaluate_vectorized(brain, eval_env, device)
        
        results[task_name] = {
            'final_sr': final_sr,
            'history': history
        }
        print(f"Task {task_name} Result: SR={final_sr:.3f}")
        
        del brain, train_env, eval_env
        torch.cuda.empty_cache()

    with open("experiments/multi_task_results.json", "w") as f:
        json.dump(results, f)
    print("\nMulti-task benchmark complete. Results saved to experiments/multi_task_results.json")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_multi_task_benchmark(device=device)
