import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def verify_gpu():
    print("--- GPU Verification Run ---")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        return

    device = torch.device("cuda")
    print(f"Device: {device}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'lr': 1e-3, 'epochs': 5, 'gamma': 0.99, # Short run of 5 epochs
        'entropy_coef': 0.05,
        'seed': seed, 'max_steps': 50,
        'eval_every': 5, 'eval_episodes': 2,
    }

    env = POMDPGridworld(size=5, seed=seed)
    brain = DigitalBrain(config).to(device)
    
    # Initialize optimization (same as train_phase3)
    # Cortex parameters are frozen for SGD
    for p in brain.cortex.parameters():
        p.requires_grad = False
        
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, brain.parameters()), lr=config['lr'])
    
    # Snapshot weights
    bg_pol_weight_init = brain.bg.policy_head.weight.clone()
    cortex_wee_init = brain.cortex.microcircuit.W_ee.clone()

    losses = []
    
    print("Starting training loop...")
    for epoch in range(config['epochs']):
        obs_np = env.reset()
        brain.reset(1, device=device)
        
        prev_reward = torch.tensor([[0.0]], device=device)
        prev_done = torch.tensor([[False]], device=device)
        
        done = False
        steps = 0
        log_probs = []
        values = []
        rewards = []
        
        while not done and steps < config['max_steps']:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            obs = Obs(x=obs_t)
            
            # Check for NaN in input
            if torch.isnan(obs_t).any():
                print(f"NaN detected in observation at step {steps}")
                break

            action, log_prob, value, _, _, _ = brain.step(obs, prev_reward, prev_done)
            
            # Check for NaN in outputs
            if torch.isnan(log_prob).any() or torch.isnan(value).any():
                print(f"NaN detected in brain output (log_prob/value) at step {steps}")
                break
                
            action_idx = int(action.item())
            next_obs_np, reward, done, _ = env.step(action_idx)
            
            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(reward)
            
            obs_np = next_obs_np
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            steps += 1
            
        # Compute Loss
        if len(rewards) > 0:
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + config['gamma'] * R
                returns.insert(0, R)
            
            returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
            log_probs_t = torch.cat(log_probs, dim=0).squeeze(-1)
            values_t = torch.cat(values, dim=0)
            
            advantage = (returns - values_t).detach().squeeze(1)
            actor_loss = -(log_probs_t * advantage).mean()
            critic_loss = nn.MSELoss()(values_t, returns)
            loss = actor_loss + critic_loss
            
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}")
                return

            optimizer.zero_grad()
            loss.backward()
            
            # Check gradients
            has_nan_grad = False
            for name, param in brain.named_parameters():
                if param.requires_grad and param.grad is not None:
                     if torch.isnan(param.grad).any():
                         print(f"NaN gradient in {name}")
                         has_nan_grad = True
            
            if has_nan_grad:
                return

            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()
            
            losses.append(loss.item())
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    # Final Checks
    print("\n--- Checks ---")
    
    # Check 1: Loss Exists
    print(f"Final Loss: {losses[-1]}")
    
    # Check 2: Weights changed
    # BG Policy (SGD)
    bg_diff = (brain.bg.policy_head.weight - bg_pol_weight_init).abs().sum().item()
    print(f"BG Policy Weight Change (L1): {bg_diff}")
    
    # Cortex W_ee (Plasticity)
    # Note: Plasticity updates happen in forward pass via cortex.update_weights
    cortex_diff = (brain.cortex.microcircuit.W_ee - cortex_wee_init).abs().sum().item()
    print(f"Cortex W_ee Weight Change (L1): {cortex_diff}")
    
    if bg_diff == 0:
        print("WARNING: BG Weights did not update!")
    if cortex_diff == 0:
        print("WARNING: Cortex Weights did not update (Plasticity inactive?)")
    
    # Check 3: Checkpoint
    ckpt_path = "verify_gpu_ckpt.pth"
    torch.save(brain.state_dict(), ckpt_path)
    if os.path.exists(ckpt_path):
        print(f"Checkpoint saved successfully: {ckpt_path}")
        os.remove(ckpt_path)
    else:
        print("ERROR: Checkpoint not saved!")

if __name__ == "__main__":
    verify_gpu()
