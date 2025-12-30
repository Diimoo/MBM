import torch
import torch.nn as nn
import torch.optim as optim
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld
import numpy as np

def train_phase1():
    config = {
        'd_obs': 9,
        'd_z': 32,
        'd_sel': 4,
        'd_act': 4,
        'lr': 1e-3,
        'epochs': 100,
        'batch_size': 32
    }
    
    env = POMDPGridworld(size=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    brain = DigitalBrain(config).to(device)
    optimizer = optim.Adam(brain.cortex.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()
    
    print("Starting Phase 1: World Model Training...")
    
    for epoch in range(config['epochs']):
        obs_np = env.reset()
        brain.reset(1, device=device)
        
        epoch_loss = 0
        steps = 0
        done = False
        
        while not done:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            obs = Obs(x=obs_t)
            
            # Step brain
            # In Phase 1, we mostly care about the cortex's ability to predict
            # We use random actions or simple heuristic for now
            action_idx = np.random.randint(env.action_space_n)
            
            # Get next state
            next_obs_np, reward, done, _ = env.step(action_idx)
            next_obs_t = torch.from_numpy(next_obs_np).unsqueeze(0).to(device)
            
            # Prediction loss
            # We need to get the prediction from the cortex
            z_t, pred_t, new_cortex_state = brain.cortex.forward(obs_t, brain.state.cortex_state)
            
            loss = criterion(pred_t, next_obs_t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update state for next step
            brain.state.z = z_t.detach()
            brain.state.cortex_state = tuple(s.detach() for s in new_cortex_state)
            
            obs_np = next_obs_np
            epoch_loss += loss.item()
            steps += 1
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {epoch_loss/steps:.6f}")

    print("Phase 1 complete.")
    torch.save(brain.cortex.state_dict(), "cortex_phase1.pth")

if __name__ == "__main__":
    train_phase1()
