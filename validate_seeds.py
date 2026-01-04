#!/usr/bin/env python3
"""Quick multi-seed validation with direct training."""
import torch
import torch.nn as nn
import numpy as np
import time

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP


def train_seed(seed, steps=5000, num_envs=1024):
    """Train on a seed and return final SR."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
        'layer_sizes': [128, 256, 128],
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': True,
        'use_cerebellum': True,
    }
    
    brain = DigitalBrain(config).to(device)
    brain.reset(batch_size=num_envs)
    
    envs = TorchVectorPOMDP(num_envs=num_envs, size=5, device=device, seed=seed)
    optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)
    
    obs_t = envs.reset()
    reward = torch.zeros(num_envs, 1, device=device)
    done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
    
    successes, episodes = 0, 0
    best_sr = 0.0
    
    # Collect rollouts and do PPO-style updates
    gamma, gae_lambda = 0.99, 0.95
    clip_eps = 0.2
    
    for update in range(steps // 128):
        # Collect rollout
        obs_batch, act_batch, logp_batch, val_batch = [], [], [], []
        rew_batch, done_batch, ent_batch = [], [], []
        
        for step in range(128):
            obs = Obs(x=obs_t)
            with torch.no_grad():
                action, log_prob, value, _, _, entropy = brain.step(obs, reward, done, learn=False)
            
            obs_batch.append(obs_t.clone())
            act_batch.append(action)
            logp_batch.append(log_prob)
            val_batch.append(value.squeeze())
            ent_batch.append(entropy)
            
            obs_t, rew, done_np, _ = envs.step(action)
            
            if done_np.any():
                episodes += done_np.sum().item()
                successes += (rew > 5).sum().item()
            
            rew_batch.append(rew)
            done_batch.append(done_np)
            
            reward = rew.unsqueeze(1)
            done = done_np.unsqueeze(1)
            
            if done.any():
                brain.reset(batch_size=num_envs)
        
        # Compute returns (simple discounted returns)
        with torch.no_grad():
            obs = Obs(x=obs_t)
            _, _, next_val, _, _, _ = brain.step(obs, reward, done, learn=False)
        
        returns = []
        R = next_val.squeeze()
        for t in reversed(range(128)):
            R = rew_batch[t] + gamma * R * (~done_batch[t])
            returns.insert(0, R)
        
        # Flatten and update
        obs_flat = torch.stack(obs_batch).view(-1, 9)
        act_flat = torch.stack(act_batch).view(-1)
        logp_old = torch.stack(logp_batch).view(-1)
        ret_flat = torch.stack(returns).view(-1)
        val_flat = torch.stack(val_batch).view(-1)
        
        adv = ret_flat - val_flat
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        # PPO update (single epoch for speed)
        brain.reset(batch_size=obs_flat.shape[0])
        obs_ppo = Obs(x=obs_flat)
        
        action, log_prob, value, _, _, entropy = brain.step(
            obs_ppo, 
            torch.zeros(obs_flat.shape[0], 1, device=device),
            torch.zeros(obs_flat.shape[0], 1, dtype=torch.bool, device=device),
            learn=True
        )
        
        # Recompute log_prob for old actions
        logits = brain.bg.policy_head(brain.state.z)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob_new = dist.log_prob(act_flat)
        entropy_new = dist.entropy()
        
        ratio = torch.exp(log_prob_new - logp_old)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * ((value.squeeze() - ret_flat) ** 2).mean()
        entropy_loss = -entropy_new.mean() * 0.01
        
        loss = policy_loss + 0.5 * value_loss + entropy_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
        optimizer.step()
        
        brain.reset(batch_size=num_envs)
        
        # Track best SR
        if episodes > 0:
            sr = successes / episodes
            best_sr = max(best_sr, sr)
    
    final_sr = successes / max(episodes, 1)
    return final_sr, best_sr, episodes


def main():
    seeds = [100, 101, 102, 103, 104]
    steps = 10000  # 10k steps per seed
    
    print("="*60)
    print("MULTI-SEED VALIDATION")
    print("="*60)
    print(f"Testing {len(seeds)} seeds, {steps} steps each\n")
    
    results = {}
    
    for seed in seeds:
        print(f"Seed {seed}...", end=" ", flush=True)
        start = time.time()
        final_sr, best_sr, episodes = train_seed(seed, steps=steps)
        elapsed = time.time() - start
        results[seed] = {'final': final_sr, 'best': best_sr, 'episodes': episodes}
        print(f"SR={final_sr:.3f} (best={best_sr:.3f}), {episodes} eps, {elapsed:.1f}s")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    finals = [r['final'] for r in results.values()]
    bests = [r['best'] for r in results.values()]
    
    print(f"\nFinal SR: {np.mean(finals):.3f} ± {np.std(finals):.3f}")
    print(f"Best SR:  {np.mean(bests):.3f} ± {np.std(bests):.3f}")
    print(f"Range:    [{np.min(finals):.3f}, {np.max(finals):.3f}]")
    
    # Criteria
    mean_sr = np.mean(finals)
    std_sr = np.std(finals)
    min_sr = np.min(finals)
    
    print(f"\nCriteria:")
    print(f"  Mean > 0.60: {'✅' if mean_sr > 0.60 else '❌'} ({mean_sr:.3f})")
    print(f"  Std < 0.15:  {'✅' if std_sr < 0.15 else '❌'} ({std_sr:.3f})")
    print(f"  Min > 0.30:  {'✅' if min_sr > 0.30 else '❌'} ({min_sr:.3f})")


if __name__ == "__main__":
    main()
