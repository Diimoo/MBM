import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def train_curriculum():
    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'lr': 1e-4, 
        'epochs_per_stage': 1000, 
        'gamma': 0.99,
        'entropy_coef': 0.05,
        'value_coef': 0.5,
        'batch_size': 8,
        'seed': 42,
        'eval_every': 100,
        'eval_episodes': 50
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Start with best model from stabilization
    brain = DigitalBrain(config).to(device)
    if os.path.exists("brain_stabilized_best.pth"):
        brain.load_state_dict(torch.load("brain_stabilized_best.pth", map_location=device), strict=False)
        print("Loaded brain_stabilized_best.pth")
    elif os.path.exists("brain_phase3_best.pth"):
        brain.load_state_dict(torch.load("brain_phase3_best.pth", map_location=device), strict=False)
        print("Loaded brain_phase3_best.pth")

    for p in brain.cortex.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, brain.parameters()), lr=config['lr'])
    
    curriculum_stages = [5, 6, 7]
    best_overall_sr = 0.0

    for size in curriculum_stages:
        print(f"\n--- Starting Curriculum Stage: Grid Size {size} ---")
        env = POMDPGridworld(size=size, seed=config['seed'])
        eval_env = POMDPGridworld(size=size, seed=config['seed'] + 1)
        
        stage_best_sr = 0.0
        
        for epoch in range(config['epochs_per_stage']):
            batch_actor_loss = 0
            batch_critic_loss = 0
            batch_entropy_loss = 0
            
            for _ in range(config['batch_size']):
                obs_np = env.reset()
                brain.reset(1, device=device)
                
                prev_reward = torch.tensor([[0.0]], device=device)
                prev_done = torch.tensor([[False]], device=device)
                
                log_probs = []
                values = []
                rewards = []
                dones = []
                entropies = []
                
                done = False
                steps = 0
                max_steps = size * size * 2
                
                while not done and steps < max_steps:
                    obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                    obs = Obs(x=obs_t)
                    
                    action, log_prob, value, _, _, entropy = brain.step(obs, prev_reward, prev_done)
                    
                    log_probs.append(log_prob)
                    values.append(value)
                    entropies.append(entropy)
                    
                    obs_np, reward, done, _ = env.step(int(action.item()))
                    
                    rewards.append(reward)
                    dones.append(done)
                    
                    prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
                    prev_done = torch.tensor([[done]], device=device)
                    steps += 1

                # Compute returns
                returns = []
                discounted_reward = 0
                for reward, is_done in zip(reversed(rewards), reversed(dones)):
                    if is_done:
                        discounted_reward = 0
                    discounted_reward = reward + (config['gamma'] * discounted_reward)
                    returns.insert(0, discounted_reward)
                    
                returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
                log_probs_t = torch.stack(log_probs).squeeze()
                values_t = torch.stack(values).squeeze()
                entropies_t = torch.stack(entropies).squeeze()
                
                if log_probs_t.dim() == 0:
                    log_probs_t = log_probs_t.unsqueeze(0)
                    values_t = values_t.unsqueeze(0)
                    entropies_t = entropies_t.unsqueeze(0)

                advantage = (returns.squeeze(1) - values_t).detach()
                
                batch_actor_loss += -(log_probs_t * advantage).mean()
                batch_critic_loss += nn.HuberLoss()(values_t, returns.squeeze(1))
                batch_entropy_loss += -entropies_t.mean()

            # Update
            loss = (batch_actor_loss + config['value_coef'] * batch_critic_loss + config['entropy_coef'] * batch_entropy_loss) / config['batch_size']
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            optimizer.step()

            if (epoch + 1) % config['eval_every'] == 0:
                sr = eval_simple(brain, eval_env, device, config['eval_episodes'], size * size * 2)
                print(f"Size {size} | Epoch {epoch+1} | SR: {sr:.2f} | Loss: {loss.item():.4f}")
                
                if sr > stage_best_sr:
                    stage_best_sr = sr
                    torch.save(brain.state_dict(), f"brain_curriculum_size{size}_best.pth")
                
                if sr >= 0.7:
                    print(f"Goal reached for size {size} (SR={sr:.2f}). Moving to next stage.")
                    break

@torch.no_grad()
def eval_simple(brain, env, device, episodes, max_steps):
    success = 0
    for _ in range(episodes):
        obs_np = env.reset()
        brain.reset(1, device=device)
        prev_reward = torch.tensor([[0.0]], device=device)
        prev_done = torch.tensor([[False]], device=device)
        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs = Obs(x=torch.from_numpy(obs_np).unsqueeze(0).to(device))
            action, _, _, _, _, _ = brain.step(obs, prev_reward, prev_done)
            obs_np, reward, done, _ = env.step(int(action.item()))
            ep_ret += reward
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            steps += 1
        if ep_ret > 5.0:
            success += 1
    return success / episodes

if __name__ == "__main__":
    train_curriculum()
