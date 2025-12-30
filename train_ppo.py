import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def train_ppo():
    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'lr': 5e-5, # Lower LR for stability
        'epochs': 3000, 
        'gamma': 0.99,
        'eps_clip': 0.2,
        'entropy_coef': 0.02, # Slightly higher entropy
        'value_coef': 0.5,
        'seed': 42,
        'max_steps': 150,
        'eval_every': 100,
        'eval_episodes': 50,
        'warmup_epochs': 500 # Epochs before allowing full gating updates
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = POMDPGridworld(size=5, seed=config['seed'])
    eval_env = POMDPGridworld(size=5, seed=config['seed'] + 1)

    brain = DigitalBrain(config).to(device)
    # Load from phase 2 (pre-trained world model)
    if os.path.exists("brain_phase2.pth"):
        brain.load_state_dict(torch.load("brain_phase2.pth", map_location=device), strict=False)
        print("Loaded brain_phase2.pth")

    for p in brain.cortex.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, brain.parameters()), lr=config['lr'])
    best_sr = 0.0

    print("Starting PPO Training Phase...")

    for epoch in range(config['epochs']):
        # Collect Rollout
        obs_np = env.reset()
        brain.reset(1, device=device)
        
        prev_reward = torch.tensor([[0.0]], device=device)
        prev_done = torch.tensor([[False]], device=device)
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        curr_entropies = []
        
        done = False
        steps = 0
        while not done and steps < config['max_steps']:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            obs = Obs(x=obs_t)
            
            # Use current brain to get action
            action, log_prob, value, _, _, entropy = brain.step(obs, prev_reward, prev_done)
            
            states.append(obs_t)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            curr_entropies.append(entropy)
            
            obs_np, reward, done, _ = env.step(int(action.item()))
            
            rewards.append(reward)
            dones.append(done)
            
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            steps += 1

        # Compute returns and advantages
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
        entropies_t = torch.stack(curr_entropies).squeeze()
        
        advantage = (returns.squeeze() - values_t).detach()
        
        # A2C+ Loss
        actor_loss = -(log_probs_t * advantage).mean()
        critic_loss = nn.MSELoss()(values_t, returns.squeeze())
        entropy_loss = -entropies_t.mean() * config['entropy_coef']
        
        # Gating Regularization: Encourage selection to be near 0 early on (neutral gating)
        # We don't have easy access to 'selection' here because it's not saved in the rollout.
        # Let's just use the loss as is for now but with a tighter clip and lower value weight.
        
        loss = actor_loss + 0.1 * critic_loss + entropy_loss # Reduced value_coef to 0.1
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5) 
        optimizer.step()

        if epoch % config['eval_every'] == 0:
            sr = eval_simple(brain, eval_env, device, config['eval_episodes'], config['max_steps'])
            print(f"Epoch {epoch} | SR: {sr:.2f} | Loss: {loss.item():.4f}")
            if sr >= best_sr:
                best_sr = sr
                torch.save(brain.state_dict(), "brain_ppo_best.pth")

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
    train_ppo()
