import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.vector_env import VectorPOMDP

def train_vectorized():
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4, 
        'lr': 2e-3, 
        'total_steps': 100000000,
        'num_envs': 12288, # High parallelism for throughput
        'num_steps': 48, # Reduced steps per update to save VRAM during advantage calculation
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.05,
        'value_coef': 0.5,
        'seed': 42,
        'eval_every': 20,
        'eval_episodes': 50,
        'selection_penalty': 0.001
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Parallel Envs: {config['num_envs']}")

    envs = VectorPOMDP(num_envs=config['num_envs'], size=5, seed=config['seed'])
    eval_env = VectorPOMDP(num_envs=1, size=5, seed=config['seed'] + 1000)

    brain = DigitalBrain(config).to(device)
    
    # Try to load existing vectorized model if possible
    if os.path.exists("brain_vectorized_best.pth"):
        try:
            state_dict = torch.load("brain_vectorized_best.pth", map_location=device)
            brain.load_state_dict(state_dict, strict=False)
            print("Loaded existing vectorized best (partial if size changed)")
        except Exception as e:
            print(f"Starting fresh: {e}")

    # Full brain learning enabled
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'])
    best_sr = 0.0

    obs_np = envs.reset()
    brain.reset(config['num_envs'], device=device)
    
    prev_reward = torch.zeros(config['num_envs'], 1, device=device)
    prev_done = torch.zeros(config['num_envs'], 1, dtype=torch.bool, device=device)

    num_updates = config['total_steps'] // (config['num_envs'] * config['num_steps'])
    
    print(f"Starting High-Throughput Vectorized Training ({config['num_envs']} envs, {config['num_steps']} steps/upd)...")

    for update in range(num_updates):
        start_time = time.time()
        
        # Memory-efficient experience storage
        values = torch.zeros(config['num_steps'], config['num_envs'], device=device)
        log_probs = torch.zeros(config['num_steps'], config['num_envs'], device=device)
        rewards = torch.zeros(config['num_steps'], config['num_envs'], device=device)
        dones = torch.zeros(config['num_steps'], config['num_envs'], device=device)
        entropies = torch.zeros(config['num_steps'], config['num_envs'], device=device)
        # We don't store selections as they are huge (num_steps, num_envs, d_sel)
        # unless we specifically need to regularize them. Let's do it sparingly.
        avg_selection_loss = 0

        # Collect experience
        for t in range(config['num_steps']):
            obs_t = torch.from_numpy(obs_np).to(device)
            obs = Obs(x=obs_t)
            
            action, log_prob, value, state, log, entropy = brain.step(obs, prev_reward, prev_done)
            
            obs_np, reward, done, _ = envs.step(action.cpu().numpy())
            
            values[t] = value.squeeze(-1)
            log_probs[t] = log_prob
            rewards[t] = torch.from_numpy(reward).to(device)
            dones[t] = torch.from_numpy(done).to(device)
            entropies[t] = entropy
            
            # Local selection regularization to save memory
            avg_selection_loss += torch.mean(torch.pow(brain._prev_selection - 1.0, 2))
            
            prev_reward = torch.from_numpy(reward).float().to(device).unsqueeze(1)
            prev_done = torch.from_numpy(done).to(device).unsqueeze(1)

        avg_selection_loss /= config['num_steps']

        # Compute GAE
        with torch.no_grad():
            _, _, next_value, _, _, _ = brain.step(Obs(x=torch.from_numpy(obs_np).to(device)), prev_reward, prev_done)
            next_value = next_value.squeeze(-1)
            
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(config['num_steps'])):
            if t == config['num_steps'] - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t + 1]
            
            delta = rewards[t] + config['gamma'] * next_values * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + config['gamma'] * config['gae_lambda'] * next_non_terminal * last_gae_lam
            
        returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Losses
        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = nn.HuberLoss()(values, returns.detach())
        entropy_loss = -entropies.mean()

        loss = actor_loss + config['value_coef'] * critic_loss + config['entropy_coef'] * entropy_loss + config['selection_penalty'] * avg_selection_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
        optimizer.step()

        # Detach state to prevent backprop through time indefinitely
        # (Though we already do it in brain.step)
        
        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)

        if (update + 1) % config['eval_every'] == 0:
            sr = eval_vectorized(brain, eval_env, device, config['eval_episodes'], 150)
            print(f"Update {update+1}/{num_updates} | SR: {sr:.2f} | Loss: {loss.item():.4f} | FPS: {fps:.0f}")
            if sr > best_sr:
                best_sr = sr
                torch.save(brain.state_dict(), "brain_vectorized_best.pth")
                print(f"New Best SR: {best_sr:.2f}")

@torch.no_grad()
def eval_vectorized(brain, env, device, episodes, max_steps):
    success = 0
    for _ in range(episodes):
        obs_np = env.reset()
        brain.reset(1, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        prev_done = torch.zeros(1, 1, dtype=torch.bool, device=device)
        done = [False]
        ep_ret = 0.0
        steps = 0
        while not done[0] and steps < max_steps:
            obs = Obs(x=torch.from_numpy(obs_np).to(device))
            action, _, _, _, _, _ = brain.step(obs, prev_reward, prev_done)
            obs_np, reward, done, _ = env.step(action.cpu().numpy())
            ep_ret += reward[0]
            prev_reward = torch.from_numpy(reward).float().to(device).unsqueeze(1)
            prev_done = torch.from_numpy(done).to(device).unsqueeze(1)
            steps += 1
        if ep_ret > 5.0:
            success += 1
    return success / episodes

if __name__ == "__main__":
    train_vectorized()
