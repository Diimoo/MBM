import torch
import sys
import os

# Ensure we can import from the root directory
sys.path.append(os.getcwd())

from digital_brain.envs.torch_minigrid import TorchMiniGrid
import time

def test_torch_minigrid():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Testing TorchMiniGrid on {device}")
    
    num_envs = 10
    env = TorchMiniGrid(num_envs=num_envs, size=8, device=device)
    
    # Check Reset
    obs = env.reset()
    print(f"Reset Obs shape: {obs.shape}") # Should be (10, 7, 7, 3)
    
    assert obs.shape == (num_envs, 7, 7, 3)
    
    # Check Step
    actions = torch.randint(0, 3, (num_envs,), device=device)
    obs, rewards, dones, info = env.step(actions)
    
    print(f"Step Obs shape: {obs.shape}")
    print(f"Rewards shape: {rewards.shape}")
    print(f"Dones shape: {dones.shape}")
    
    assert obs.shape == (num_envs, 7, 7, 3)
    assert rewards.shape == (num_envs,)
    assert dones.shape == (num_envs,)
    
    # Check flattening compatibility
    obs_flat = obs.view(num_envs, -1)
    print(f"Flattened shape: {obs_flat.shape}") # Should be (10, 147)
    
    print("TorchMiniGrid basic test passed!")

if __name__ == "__main__":
    test_torch_minigrid()
