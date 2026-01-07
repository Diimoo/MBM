import torch
import numpy as np
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld
import os

def verify_stability():
    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'lr': 1e-3, 'total_steps': 1000, 'num_envs': 1, 'num_steps': 10,
        'gamma': 0.99, 'entropy_coef': 0.05, 'value_coef': 0.5,
        'seed': 42, 'eval_every': 10, 'eval_episodes': 10,
        'selection_penalty': 0.001
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain = DigitalBrain(config).to(device)
    
    # Check if we have a checkpoint, else use random
    ckpt = "brain_vectorized_best.pth"
    if os.path.exists(ckpt):
        try:
            brain.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            print(f"Loaded {ckpt}")
        except:
            print("Running with random weights")
    else:
        print("Running with random weights")

    def eval_run(seed):
        # Fix seeds for determinism
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        env = POMDPGridworld(size=5, seed=seed)
        obs_np = env.reset()
        brain.reset(1, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        prev_done = torch.zeros(1, 1, dtype=torch.bool, device=device)
        
        actions = []
        values = []
        
        for _ in range(20):
            obs = Obs(x=torch.from_numpy(obs_np).unsqueeze(0).to(device))
            # Use act() to ensure learn=False
            action, _, value, _, _, _ = brain.act(obs, prev_reward, prev_done)
            
            obs_np, reward, done, _ = env.step(int(action.item()))
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            
            actions.append(int(action.item()))
            values.append(value.item())
            if done: break
        return actions, values

    print("Running Eval 1...")
    a1, v1 = eval_run(123)
    
    print("Running Eval 2...")
    a2, v2 = eval_run(123)
    
    if a1 == a2 and np.allclose(v1, v2):
        print("\nSUCCESS: Evaluations are deterministic (no leakage from learning).")
    else:
        print("\nFAILURE: Evaluations differ!")
        print(f"Actions 1: {a1}")
        print(f"Actions 2: {a2}")
        if len(v1) == len(v2):
            print(f"Value diff: {np.abs(np.array(v1) - np.array(v2)).max()}")

if __name__ == "__main__":
    verify_stability()
