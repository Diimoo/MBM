import torch
import torch.optim as optim
import numpy as np
import os
import sys
import time
import json

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from digital_brain.datatypes import Obs

def benchmark_forward(brain, n_iters=100, device='cuda'):
    x = torch.randn(brain.config['num_envs'], brain.config['d_obs'], device=device)
    obs = Obs(x=x)
    prev_rew = torch.zeros(brain.config['num_envs'], 1, device=device)
    prev_done = torch.zeros(brain.config['num_envs'], 1, dtype=torch.bool, device=device)
    
    # Warmup
    for _ in range(10):
        brain.step(obs, prev_rew, prev_done, learn=False)
        
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        brain.step(obs, prev_rew, prev_done, learn=False)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000 / n_iters

def test_large_scale(device='cuda'):
    configs = [
        {'d_z': 4096, 'sparsity': 0.05},    # Current
        {'d_z': 8192, 'sparsity': 0.03},    # 2x scale
        {'d_z': 16384, 'sparsity': 0.02},   # 4x scale
        {'d_z': 32768, 'sparsity': 0.01},   # 8x scale
    ]
    
    results = []
    
    for cfg in configs:
        print(f"\nTesting d_z={cfg['d_z']}, sparsity={cfg['sparsity']}")
        
        brain_cfg = {
            'd_obs': 9, 'd_z': cfg['d_z'], 'd_sel': 64, 'd_act': 4,
            'lr': 3e-4, 'seed': 42,
            'sparse_cortex': True,
            'sparsity': cfg['sparsity'],
            'num_envs': 128, 'num_steps': 128,
            'gamma': 0.99, 'gae_lambda': 0.95, 'eps_clip': 0.2,
            'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
            'ppo_epochs': 4, 'mini_batch_size': 4096,
            'total_updates': 10
        }
        
        try:
            brain = DigitalBrain(brain_cfg).to(device)
            mem_mb = sum(p.numel() * 4 for p in brain.parameters()) / 1e6
            print(f"  Model size: {mem_mb:.1f} MB")
            
            time_per_step = benchmark_forward(brain, n_iters=50, device=device)
            print(f"  Time/step: {time_per_step:.2f} ms")
            
            # Short training run for stability check
            # env = TorchVectorPOMDP(num_envs=128, size=5, device=device)
            # optimizer = optim.Adam(brain.parameters(), lr=3e-4)
            # from experiments.train_utils import train_mbm
            # train_mbm(env, brain, optimizer, brain_cfg, device, verbose=False)
            
            results.append({
                'd_z': cfg['d_z'],
                'sparsity': cfg['sparsity'],
                'mem_mb': mem_mb,
                'time_ms': time_per_step,
                'stable': True
            })
            
            del brain
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            print(f"  ❌ Failed at scale {cfg['d_z']}: {e}")
            results.append({
                'd_z': cfg['d_z'],
                'sparsity': cfg['sparsity'],
                'error': str(e),
                'stable': False
            })
            torch.cuda.empty_cache()

    with open("experiments/scaling_results.json", "w") as f:
        json.dump(results, f)
    print("\nScaling benchmark complete. Results saved to experiments/scaling_results.json")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_large_scale(device=device)
