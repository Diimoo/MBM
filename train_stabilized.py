import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def train_stabilized():
    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'lr': 3e-4, # As per user suggestion
        'epochs': 3000, 
        'gamma': 0.99,
        'entropy_coef': 0.05,
        'value_coef': 0.5,
        'batch_size': 16,
        'seed': 42,
        'max_steps': 150,
        'eval_every': 100,
        'eval_episodes': 100,
        'selection_penalty': 0.001 # Reduced penalty
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = POMDPGridworld(size=5, seed=config['seed'])
    eval_env = POMDPGridworld(size=5, seed=config['seed'] + 1)

    brain = DigitalBrain(config).to(device)
    # Start from the best curriculum model
    if os.path.exists("brain_curriculum_size6_best.pth"):
        brain.load_state_dict(torch.load("brain_curriculum_size6_best.pth", map_location=device), strict=False)
        print("Loaded brain_curriculum_size6_best.pth as baseline")
    elif os.path.exists("brain_stabilized_best.pth"):
        brain.load_state_dict(torch.load("brain_stabilized_best.pth", map_location=device), strict=False)
        print("Loaded brain_stabilized_best.pth")

    for p in brain.cortex.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, brain.parameters()), lr=config['lr'])
    best_sr = 0.0

    print("Starting Final Optimization Push...")

    for epoch in range(config['epochs']):
        batch_actor_loss = 0
        batch_critic_loss = 0
        batch_entropy_loss = 0
        batch_selection_loss = 0
        
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
            selections = []
            
            done = False
            steps = 0
            while not done and steps < config['max_steps']:
                obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                obs = Obs(x=obs_t)
                
                # We need to capture selection to regularize it
                # Modify brain.step to return selection or access it
                action, log_prob, value, state, log, entropy = brain.step(obs, prev_reward, prev_done)
                
                log_probs.append(log_prob)
                values.append(value)
                entropies.append(entropy)
                # brain._prev_selection is the one used in the NEXT step, but it's the one computed in THIS step
                selections.append(brain._prev_selection)
                
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
            selections_t = torch.stack(selections).squeeze()
            
            if log_probs_t.dim() == 0:
                log_probs_t = log_probs_t.unsqueeze(0)
                values_t = values_t.unsqueeze(0)
                entropies_t = entropies_t.unsqueeze(0)
                selections_t = selections_t.unsqueeze(0)

            advantage = (returns.squeeze(1) - values_t).detach()
            
            batch_actor_loss += -(log_probs_t * advantage).mean()
            batch_critic_loss += nn.HuberLoss()(values_t, returns.squeeze(1))
            batch_entropy_loss += -entropies_t.mean()
            # Selection regularization: encourage selection to not be zero/low (i.e. keep gates open)
            # gate = sigmoid(selection), so we want selection to be positive
            batch_selection_loss += torch.mean(torch.pow(selections_t - 1.0, 2))

        # Update
        loss = (batch_actor_loss + 
                config['value_coef'] * batch_critic_loss + 
                config['entropy_coef'] * batch_entropy_loss +
                config['selection_penalty'] * batch_selection_loss) / config['batch_size']
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
        optimizer.step()

        if (epoch + 1) % config['eval_every'] == 0:
            sr = eval_simple(brain, eval_env, device, config['eval_episodes'], config['max_steps'])
            print(f"Epoch {epoch+1} | SR: {sr:.2f} | Loss: {loss.item():.4f}")
            if sr > best_sr:
                best_sr = sr
                torch.save(brain.state_dict(), "brain_stabilized_best.pth")
                print(f"New Best SR: {best_sr:.2f}")

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
            # Use act() for evaluation to avoid state mutation/learning
            action, _, _, _, _, _ = brain.act(obs, prev_reward, prev_done)
            obs_np, reward, done, _ = env.step(int(action.item()))
            ep_ret += reward
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            steps += 1
        if ep_ret > 5.0:
            success += 1
    return success / episodes

if __name__ == "__main__":
    train_stabilized()
