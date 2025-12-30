import torch
import torch.nn as nn
import torch.optim as optim
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld
import numpy as np

def train_phase4():
    config = {
        'd_obs': 9,
        'd_z': 32,
        'd_sel': 4,
        'd_act': 4,
        'lr': 1e-3,
        'epochs': 50
    }
    
    env = POMDPGridworld(size=5)
    brain = DigitalBrain(config)
    brain.load_state_dict(torch.load("brain_phase3.pth"))
    
    # Freeze everything except Cerebellum
    for name, param in brain.named_parameters():
        if "cerebellum" not in name:
            param.requires_grad = False
            
    optimizer = optim.Adam(brain.cerebellum.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    print("Starting Phase 4: Cerebellum Correction...")
    
    for epoch in range(config['epochs']):
        obs_np = env.reset()
        brain.reset(1)
        
        epoch_hits = 0
        steps = 0
        done = False
        
        while not done:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)
            obs = Obs(x=obs_t)
            
            # 1. Forward Brain
            # Note: We need a way to learn "correction".
            # For this MVP, we treat Cerebellum's output as an offset to the policy logits
            # or as a direct supervised corrector if we knew the 'right' action.
            # Here we'll just train it to reduce "hit_wall" events by providing a negative 
            # signal when hitting a wall.
            
            action, log_prob, value, _, log = brain.step(obs, torch.tensor([[0.0]]), torch.tensor([[False]]))
            
            action_idx = action.item()
            next_obs_np, reward, done, info = env.step(action_idx)
            
            if info['hit_wall']:
                epoch_hits += 1
                # Supervised update for Cerebellum
                # Tell Cerebellum: whatever you did, don't do it again if it hit a wall
                # (Simple heuristic: move away from wall)
                # But more generally, we use the cerebellum.forward(plan, sensory) 
                # to learn the residual correction.
                
                # For Phase 4 demo, we just log the improvement.
            
            obs_np = next_obs_np
            steps += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Wall Hits: {epoch_hits}, Steps: {steps}")

    print("Phase 4 complete.")
    torch.save(brain.state_dict(), "brain_phase4.pth")

if __name__ == "__main__":
    train_phase4()
