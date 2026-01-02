import torch
import torch.optim as optim
import numpy as np
import os
import sys
import json
from scipy import stats

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from experiments.train_utils import train_mbm, train_ppo_baseline, evaluate_vectorized
from experiments.compare_mbm_vs_ppo import PPOBaseline, get_env

def validate_pomdp_advantage(n_seeds=5, device="cuda"):
    """Replicate MBM vs PPO comparison with statistical rigor."""
    print(f"Starting statistical validation over {n_seeds} seeds...")
    
    results = {
        'mbm_losses': [],
        'ppo_losses': [],
        'mbm_srs': [],
        'ppo_srs': [],
    }
    
    task_name = 'gridworld_5x5'
    num_envs = 512
    eval_envs = 64
    n_updates = 100
    
    for seed in range(n_seeds):
        print(f"\n--- Seed {seed} ---")
        
        # 1) MBM Training
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
            'use_hippocampus': True,
            'use_plasticity': True,
            'use_memory_policy': True,
            'use_cerebellum': True
        }
        
        brain = DigitalBrain(brain_config).to(device)
        optimizer = optim.Adam(brain.parameters(), lr=brain_config['lr'])
        
        print("Training MBM...")
        train_mbm(train_env, brain, optimizer, brain_config, device, verbose=False)
        mbm_sr = evaluate_vectorized(brain, eval_env, device)
        
        # 2) PPO Training
        ppo_config = {
            'num_envs': num_envs, 'num_steps': 128, 'ppo_epochs': 4,
            'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.01, 'lr': 3e-4,
            'mini_batch_size': 16384,
            'total_updates': n_updates,
            'vf_clip': 0.2,
            'd_obs': d_obs
        }
        
        ppo_model = PPOBaseline(d_obs, 512, d_act).to(device)
        ppo_optimizer = optim.Adam(ppo_model.parameters(), lr=ppo_config['lr'])
        
        print("Training PPO...")
        train_ppo_baseline(train_env, ppo_model, ppo_optimizer, ppo_config, device, verbose=False)
        ppo_sr = evaluate_vectorized(ppo_model, eval_env, device)
        
        results['mbm_srs'].append(mbm_sr)
        results['ppo_srs'].append(ppo_sr)
        print(f"Seed {seed} Result: MBM SR={mbm_sr:.3f}, PPO SR={ppo_sr:.3f}")

    # Statistical analysis
    mbm_mean = np.mean(results['mbm_srs'])
    mbm_std = np.std(results['mbm_srs'])
    ppo_mean = np.mean(results['ppo_srs'])
    ppo_std = np.std(results['ppo_srs'])
    
    t_stat, p_value = stats.ttest_ind(results['mbm_srs'], results['ppo_srs'])
    
    print("\n--- Final Results ---")
    print(f"MBM SR: {mbm_mean:.3f} +/- {mbm_std:.3f}")
    print(f"PPO SR: {ppo_mean:.3f} +/- {ppo_std:.3f}")
    print(f"t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Statistically significant difference!")
    else:
        print("❌ No statistically significant difference.")

    with open("experiments/validation_results.json", "w") as f:
        json.dump(results, f)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validate_pomdp_advantage(n_seeds=5, device=device)
