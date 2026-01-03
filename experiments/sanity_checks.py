#!/usr/bin/env python3
"""
Sanity checks to verify:
1. Environment is correct and learnable
2. Random policy performance (baseline)
3. Simple MLP can learn the task (proves task is learnable)
4. Observation/action space verification
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from digital_brain.envs.torch_vector_env import TorchVectorPOMDP


class SimpleMLP(nn.Module):
    """Tiny MLP to verify task is learnable."""
    def __init__(self, d_obs=9, d_hidden=64, d_act=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_obs, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.policy = nn.Linear(d_hidden, d_act)
        self.value = nn.Linear(d_hidden, 1)
    
    def forward(self, x):
        h = self.net(x)
        return self.policy(h), self.value(h)
    
    def act(self, x):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value.squeeze(-1)


def check_environment(size=5, seed=0, num_envs=4):
    """Verify environment behavior."""
    print(f"\n{'='*50}")
    print(f"ENVIRONMENT CHECK: {size}x{size} grid, seed={seed}")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TorchVectorPOMDP(num_envs=num_envs, size=size, device=device, seed=seed)
    
    obs = env.reset()
    print(f"\nObservation:")
    print(f"  Shape: {obs.shape} (expected: ({num_envs}, 9))")
    print(f"  Dtype: {obs.dtype}")
    print(f"  Range: [{obs.min().item():.3f}, {obs.max().item():.3f}]")
    print(f"  Sample obs[0]: {obs[0].cpu().numpy()}")
    
    # Check action space
    print(f"\nAction space: 4 discrete actions (0=up, 1=right, 2=down, 3=left)")
    
    # Take some random actions
    print(f"\nRandom rollout (10 steps):")
    total_reward = 0
    for step in range(10):
        action = torch.randint(0, 4, (num_envs,), device=device)
        obs, reward, done, info = env.step(action)
        total_reward += reward.sum().item()
        if step < 3:
            print(f"  Step {step}: action={action[0].item()}, reward={reward[0].item():.2f}, done={done[0].item()}")
    
    print(f"  Total reward (10 steps, {num_envs} envs): {total_reward:.2f}")
    
    return True


def eval_random_policy(size=5, seed=0, episodes=1000):
    """Evaluate random policy to establish baseline."""
    print(f"\n{'='*50}")
    print(f"RANDOM POLICY BASELINE: {size}x{size} grid")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = min(64, episodes)
    env = TorchVectorPOMDP(num_envs=num_envs, size=size, device=device, seed=seed)
    
    success = 0
    completed = 0
    max_steps = 150
    
    while completed < episodes:
        obs = env.reset()
        ep_returns = torch.zeros(num_envs, device=device)
        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            action = torch.randint(0, 4, (num_envs,), device=device)
            obs, reward, done, _ = env.step(action)
            ep_returns += reward * (~ep_done).float()
            ep_done = ep_done | done
            
            if ep_done.all():
                break
        
        success += (ep_returns > 5.0).sum().item()
        completed += num_envs
    
    sr = success / completed
    print(f"\n  Episodes: {completed}")
    print(f"  Successes: {success}")
    print(f"  Success Rate: {sr:.4f} ({sr*100:.2f}%)")
    print(f"  (Expected: ~1-5% for random policy on small grids)")
    
    return sr


def train_simple_mlp(size=5, seed=0, updates=100, verbose=True):
    """Train simple MLP to verify task is learnable."""
    print(f"\n{'='*50}")
    print(f"SIMPLE MLP TRAINING: {size}x{size} grid")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    num_envs = 64
    num_steps = 64
    
    model = SimpleMLP(d_obs=9, d_hidden=64, d_act=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    env = TorchVectorPOMDP(num_envs=num_envs, size=size, device=device, seed=seed)
    
    # Buffers
    obs_buf = torch.zeros((num_steps, num_envs, 9), device=device)
    act_buf = torch.zeros((num_steps, num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((num_steps, num_envs), device=device)
    val_buf = torch.zeros((num_steps, num_envs), device=device)
    rew_buf = torch.zeros((num_steps, num_envs), device=device)
    done_buf = torch.zeros((num_steps, num_envs), dtype=torch.bool, device=device)
    
    obs_t = env.reset()
    best_sr = 0.0
    
    for update in range(updates):
        # Collect experience
        with torch.no_grad():
            for t in range(num_steps):
                obs_buf[t] = obs_t
                action, logp, value = model.act(obs_t)
                obs_next, reward, done, _ = env.step(action)
                
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = value
                rew_buf[t] = reward
                done_buf[t] = done
                obs_t = obs_next
        
        # Compute returns (simple discounted sum)
        gamma = 0.99
        gae_lambda = 0.95
        
        with torch.no_grad():
            _, next_value = model(obs_t)
            next_value = next_value.squeeze(-1)
        
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = 0
        for t in reversed(range(num_steps)):
            next_non_terminal = (~done_buf[t]).float()
            next_val = next_value if t == num_steps - 1 else val_buf[t+1]
            delta = rew_buf[t] + gamma * next_val * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        ret_buf = adv_buf + val_buf
        
        # PPO update
        for _ in range(4):
            logits, values = model(obs_buf.view(-1, 9))
            values = values.squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            
            new_logp = dist.log_prob(act_buf.view(-1))
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_logp - logp_buf.view(-1))
            mb_adv = adv_buf.view(-1)
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * ((values - ret_buf.view(-1))**2).mean()
            
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
        
        # Evaluate
        if (update + 1) % 10 == 0:
            sr = evaluate_mlp(model, size, device)
            best_sr = max(best_sr, sr)
            if verbose:
                print(f"  Update {update+1}/{updates}: SR = {sr:.3f} (best: {best_sr:.3f})")
    
    print(f"\n  Final SR: {sr:.3f}")
    print(f"  Best SR: {best_sr:.3f}")
    
    if best_sr > 0.5:
        print(f"  ✓ Task is LEARNABLE (>50% SR achieved)")
    elif best_sr > 0.2:
        print(f"  ~ Task is partially learnable (20-50% SR)")
    else:
        print(f"  ✗ Task may have issues or needs more training")
    
    return best_sr


def evaluate_mlp(model, size, device, episodes=64):
    """Evaluate MLP model."""
    env = TorchVectorPOMDP(num_envs=min(32, episodes), size=size, device=device, seed=999)
    
    success = 0
    completed = 0
    max_steps = 150
    
    while completed < episodes:
        obs = env.reset()
        ep_returns = torch.zeros(env.num_envs, device=device)
        ep_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            with torch.no_grad():
                logits, _ = model(obs)
                action = torch.argmax(logits, dim=-1)
            
            obs, reward, done, _ = env.step(action)
            ep_returns += reward * (~ep_done).float()
            ep_done = ep_done | done
            
            if ep_done.all():
                break
        
        success += (ep_returns > 5.0).sum().item()
        completed += env.num_envs
    
    return success / completed


def run_all_sanity_checks():
    """Run all sanity checks."""
    print("\n" + "="*70)
    print("RUNNING ALL SANITY CHECKS")
    print("="*70)
    
    results = {}
    
    # 1. Environment check
    for size in [5, 7, 10]:
        check_environment(size=size, seed=0)
    
    # 2. Random policy baseline
    for size in [5, 7, 10]:
        sr = eval_random_policy(size=size, episodes=500)
        results[f'random_{size}x{size}'] = sr
    
    # 3. Simple MLP training
    print("\n" + "="*70)
    print("SIMPLE MLP LEARNABILITY TEST")
    print("="*70)
    
    for size in [5, 7]:
        sr = train_simple_mlp(size=size, updates=50, verbose=True)
        results[f'mlp_{size}x{size}'] = sr
    
    # Summary
    print("\n" + "="*70)
    print("SANITY CHECK SUMMARY")
    print("="*70)
    
    print("\nRandom Policy Baselines:")
    for size in [5, 7, 10]:
        key = f'random_{size}x{size}'
        if key in results:
            print(f"  {size}x{size}: {results[key]*100:.2f}%")
    
    print("\nSimple MLP (verifies learnability):")
    for size in [5, 7]:
        key = f'mlp_{size}x{size}'
        if key in results:
            status = "✓" if results[key] > 0.5 else "~" if results[key] > 0.2 else "✗"
            print(f"  {size}x{size}: {results[key]*100:.1f}% {status}")
    
    # Interpretation
    print("\nInterpretation:")
    if results.get('mlp_5x5', 0) > 0.5:
        print("  - 5x5 task is learnable → if MBM fails, issue is in MBM")
    else:
        print("  - 5x5 task may be too hard or environment has issues")
    
    if results.get('mlp_7x7', 0) > 0.3:
        print("  - 7x7 task is learnable with simple MLP")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", choices=['env', 'random', 'mlp', 'all'], default='all')
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    if args.check == 'env':
        check_environment(args.size, args.seed)
    elif args.check == 'random':
        eval_random_policy(args.size, args.seed)
    elif args.check == 'mlp':
        train_simple_mlp(args.size, args.seed)
    else:
        run_all_sanity_checks()
