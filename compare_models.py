import torch
import numpy as np
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def eval_model(path, name, device, size=5, episodes=200):
    config = {'d_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4}
    brain = DigitalBrain(config).to(device)
    try:
        brain.load_state_dict(torch.load(path, map_location=device), strict=False)
    except:
        print(f"Could not load {path}")
        return
    
    env = POMDPGridworld(size=size, seed=42)
    success = 0
    returns = []
    
    for _ in range(episodes):
        obs_np = env.reset()
        brain.reset(1, device=device)
        prev_reward = torch.tensor([[0.0]], device=device)
        prev_done = torch.tensor([[False]], device=device)
        done = False
        ep_ret = 0.0
        steps = 0
        max_steps = size * size * 2
        while not done and steps < max_steps:
            obs = Obs(x=torch.from_numpy(obs_np).unsqueeze(0).to(device))
            # New signature: action, log_prob, value, state, log, entropy
            out = brain.step(obs, prev_reward, prev_done)
            action = out[0]
            obs_np, reward, done, _ = env.step(int(action.item()))
            ep_ret += reward
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            steps += 1
        if ep_ret > 5.0:
            success += 1
        returns.append(ep_ret)
    
    sr = success / episodes
    print(f"{name:<25} | SR: {sr:.3f} | Mean Return: {np.mean(returns):.2f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing models on {device}...")
    print(f"{'Model':<25} | {'SR':<6} | {'Mean Return'}")
    print("-" * 45)
    
    models = [
        ("brain_phase3_best.pth", "Phase 3 Best"),
        ("brain_ppo_best.pth", "PPO Best"),
        ("brain_stabilized_best.pth", "Stabilized Best"),
        ("brain_curriculum_size5_best.pth", "Curr Size 5"),
        ("brain_curriculum_size6_best.pth", "Curr Size 6 (Eval on 5)"),
    ]
    
    for path, name in models:
        if os.path.exists(path):
            eval_model(path, name, device)

if __name__ == "__main__":
    main()
