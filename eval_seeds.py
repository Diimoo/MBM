import torch, numpy as np
import sys
import os

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def run_eval():
    config = {'d_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    brain = DigitalBrain(config).to(device)
    try:
        brain.load_state_dict(torch.load("brain_curriculum_size6_best.pth", map_location="cpu"), strict=False)
        print("Loaded brain_curriculum_size6_best.pth")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return
        
    brain.to(device)

    def eval_seed(seed, episodes=200, max_steps=150):
        env = POMDPGridworld(size=5, seed=seed)
        succ = 0
        rets = []
        for _ in range(episodes):
            obs_np = env.reset()
            brain.reset(1, device=device)
            prev_reward = torch.tensor([[0.0]], device=device)
            prev_done = torch.tensor([[False]], device=device)
            done = False
            ep_ret = 0.0
            steps = 0
            while (not done) and steps < max_steps:
                obs = Obs(x=torch.from_numpy(obs_np).unsqueeze(0).to(device))
                # Use act() for evaluation
                action, _, _, _, _, _ = brain.act(obs, prev_reward, prev_done)
                obs_np, r, done, _ = env.step(int(action.item()))
                ep_ret += float(r)
                prev_reward = torch.tensor([[r]], dtype=torch.float32, device=device)
                prev_done = torch.tensor([[done]], device=device)
                steps += 1
            if ep_ret > 5.0: succ += 1
            rets.append(ep_ret)
        return succ/episodes, float(np.mean(rets)), float(np.std(rets))

    seeds=[0,1,2,3,4,5,6,7,8,9]
    rows=[]
    print(f"{'Seed':<5} | {'SR':<6} | {'Return'}")
    print("-" * 30)
    for s in seeds:
        sr, mu, sd = eval_seed(100+s)
        rows.append((s, sr, mu, sd))
        print(f"{s:<5} | {sr:<6.2f} | {mu:<.2f} +/- {sd:<.2f}")

    SRs=[r[1] for r in rows]
    print("-" * 30)
    print("SR mean/std:", float(np.mean(SRs)), float(np.std(SRs)))

if __name__ == "__main__":
    run_eval()
