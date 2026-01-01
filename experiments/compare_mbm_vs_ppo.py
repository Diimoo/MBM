import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from digital_brain.envs.t_maze import TorchVectorTMaze
from digital_brain.envs.radial_arm_maze import TorchVectorRadialArmMaze
from digital_brain.envs.cartpole import TorchVectorCartPole
from experiments.train_utils import train_mbm, train_ppo_baseline

# A simple PPO Baseline for comparison
class PPOBaseline(nn.Module):
    def __init__(self, d_obs, d_h, d_act):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(d_obs, d_h),
            nn.ReLU(),
            nn.Linear(d_h, d_h),
            nn.ReLU()
        )
        self.policy_head = nn.Linear(d_h, d_act)
        self.value_head = nn.Linear(d_h, 1)

    def forward(self, x):
        h = self.feature_extractor(x)
        logits = self.policy_head(h)
        value = self.value_head(h)
        return logits, value

    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

def get_env(task_name, num_envs, device):
    if task_name == 'gridworld_5x5':
        return TorchVectorPOMDP(num_envs=num_envs, size=5, device=device), 9, 4
    elif task_name == 'gridworld_7x7':
        return TorchVectorPOMDP(num_envs=num_envs, size=7, device=device), 9, 4
    elif task_name == 't_maze_5':
        return TorchVectorTMaze(num_envs=num_envs, corridor_length=5, device=device), 8, 3
    elif task_name == 'radial_arm_8':
        return TorchVectorRadialArmMaze(num_envs=num_envs, num_arms=8, arm_length=3, device=device), 17, 9
    elif task_name == 'cartpole':
        return TorchVectorCartPole(num_envs=num_envs, device=device), 4, 2
    else:
        raise ValueError(f"Unknown task: {task_name}")

def run_experiment(config_name, task_name, device, total_steps=2_000_000):
    print(f"\n--- Starting {config_name} on {task_name} ---")
    
    train_env, d_obs, d_act = get_env(task_name, 512, device)
    eval_env, _, _ = get_env(task_name, 64, device)
    
    config = {
        'num_envs': 512, 'num_steps': 128, 'ppo_epochs': 4,
        'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
        'value_coef': 0.5, 'entropy_coef': 0.01, 'lr': 3e-4,
        'mini_batch_size': 16384,
        'total_updates': total_steps // (512 * 128),
        'target_kl': 0.015,
        'vf_clip': 0.2,
        'd_obs': d_obs
    }

    if config_name.startswith('mbm'):
        brain_config = {
            'd_obs': d_obs, 'd_z': 512, 'd_sel': 64, 'd_act': d_act,
            'lr': 3e-4, 'seed': 42,
            'sparse_cortex': False,
            'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'ppo_epochs': 4, 'mini_batch_size': 16384,
            'num_envs': 512, 'num_steps': 128,
            'total_updates': config['total_updates'],
            'use_hippocampus': 'no_memory' not in config_name,
            'use_plasticity': 'no_plasticity' not in config_name,
            'use_memory_policy': 'no_memory' not in config_name,
            'use_cerebellum': True
        }
        brain = DigitalBrain(brain_config).to(device)
        optimizer = optim.Adam(brain.parameters(), lr=brain_config['lr'])
        train_mbm(train_env, brain, optimizer, brain_config, device, eval_env=eval_env)
    else:
        # Standard PPO Baseline
        model = PPOBaseline(d_obs, 512, d_act).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        train_ppo_baseline(train_env, model, optimizer, config, device, eval_env=eval_env)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Target configurations from PLAN.md
    configs = ['mbm_full', 'ppo_baseline']
    # Select subset of tasks for initial benchmark
    tasks = ['gridworld_5x5', 't_maze_5', 'cartpole', 'radial_arm_8']
    
    for task in tasks:
        for config in configs:
            run_experiment(config, task, device, total_steps=1_000_000)
