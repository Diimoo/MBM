#!/usr/bin/env python3
"""
Multi-seed validation to confirm stability fixes enable consistent learning.
Run this overnight to get definitive results.
"""
import torch
import numpy as np
import sys
import os
import time

sys.path.insert(0, os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs, BrainState
from digital_brain.envs.vector_env import VectorPOMDP


def train_and_evaluate(seed, train_steps=20000, eval_episodes=200, verbose=True):
    """Train MBM on a specific seed and evaluate."""
    
    # Config matching train_vectorized.py (POMDP gridworld)
    config = {
        'd_obs': 9,  # 3x3 local observation
        'd_z': 256,
        'd_sel': 32,
        'd_act': 4,  # 4 actions: up/down/left/right
        'layer_sizes': [128, 256, 128],
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': True,
        'use_cerebellum': True,
        'sparse_cortex': False,
        'hip_confidence_threshold': 0.5,
    }
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    brain = DigitalBrain(config)
    brain.reset(batch_size=16)  # 16 parallel envs
    
    # Training hyperparameters
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_eps = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    max_grad_norm = 0.5
    
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
    
    # Create environment
    envs = VectorPOMDP(num_envs=16, size=5, seed=seed)
    obs_raw = envs.reset()
    
    # Convert obs to tensor
    def process_obs(obs_raw):
        return torch.tensor(obs_raw, dtype=torch.float32)
    
    obs = Obs(x=process_obs(obs_raw))
    reward = torch.zeros(16, 1)
    done = torch.zeros(16, 1, dtype=torch.bool)
    
    # Training loop
    steps_per_update = 128
    num_updates = train_steps // (16 * steps_per_update)
    
    best_sr = 0.0
    learning_curve = []
    
    for update in range(num_updates):
        # Collect rollout
        obs_batch, act_batch, rew_batch, done_batch = [], [], [], []
        logp_batch, val_batch, ent_batch = [], [], []
        
        for step in range(steps_per_update):
            with torch.no_grad():
                action, log_prob, value, state, log, entropy = brain.step(
                    obs, reward, done, learn=False
                )
            
            obs_batch.append(obs.x)
            act_batch.append(action)
            logp_batch.append(log_prob)
            val_batch.append(value)
            ent_batch.append(entropy)
            
            # Step environment (VectorPOMDP returns 4 values)
            obs_raw, rew, done_np, info = envs.step(action.numpy())
            
            reward = torch.tensor(rew, dtype=torch.float32).unsqueeze(1)
            done = torch.tensor(done_np, dtype=torch.bool).unsqueeze(1)
            obs = Obs(x=process_obs(obs_raw))
            
            rew_batch.append(reward)
            done_batch.append(done)
            
            # Reset brain state for done envs
            if done.any():
                brain.reset(batch_size=16)
        
        # Compute returns and advantages (GAE)
        with torch.no_grad():
            _, _, next_value, _, _, _ = brain.step(obs, reward, done, learn=False)
        
        returns = []
        advantages = []
        gae = torch.zeros(16, 1)
        
        for t in reversed(range(steps_per_update)):
            if t == steps_per_update - 1:
                next_val = next_value
            else:
                next_val = val_batch[t + 1]
            
            delta = rew_batch[t] + gamma * next_val * (~done_batch[t]) - val_batch[t]
            gae = delta + gamma * gae_lambda * (~done_batch[t]) * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + val_batch[t])
        
        # Flatten batches
        obs_flat = torch.stack(obs_batch).view(-1, config['d_obs'])
        act_flat = torch.stack(act_batch).view(-1)
        logp_old = torch.stack(logp_batch).view(-1)
        val_old = torch.stack(val_batch).view(-1)
        ret_flat = torch.stack(returns).view(-1)
        adv_flat = torch.stack(advantages).view(-1)
        
        # Normalize advantages
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        # PPO update
        brain.reset(batch_size=obs_flat.shape[0])
        
        for _ in range(4):  # PPO epochs
            # Forward pass
            obs_ppo = Obs(x=obs_flat)
            action, log_prob, value, _, _, entropy = brain.step(
                obs_ppo, 
                torch.zeros(obs_flat.shape[0], 1),
                torch.zeros(obs_flat.shape[0], 1, dtype=torch.bool),
                learn=True,
                action_to_eval=act_flat
            )
            
            # PPO loss
            ratio = torch.exp(log_prob - logp_old)
            surr1 = ratio * adv_flat
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_flat
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = 0.5 * ((value.squeeze() - ret_flat) ** 2).mean()
            entropy_loss = -entropy.mean()
            
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), max_grad_norm)
            optimizer.step()
        
        # Evaluate periodically
        if update % 10 == 0:
            sr = quick_eval(brain, config, seed, episodes=64)
            learning_curve.append((update, sr))
            best_sr = max(best_sr, sr)
            
            if verbose:
                print(f"  Seed {seed} | Update {update:4d} | SR: {sr:.3f} | Best: {best_sr:.3f}")
        
        brain.reset(batch_size=16)
    
    # Final evaluation
    final_sr = quick_eval(brain, config, seed, episodes=eval_episodes)
    
    return {
        'seed': seed,
        'final_sr': final_sr,
        'best_sr': best_sr,
        'learning_curve': learning_curve
    }


def quick_eval(brain, config, seed, episodes=64):
    """Quick evaluation without gradient tracking."""
    eval_env = VectorPOMDP(num_envs=16, size=5, seed=seed+1000)
    
    successes = 0
    total = 0
    
    brain.reset(batch_size=16)
    obs_raw = eval_env.reset()
    
    def process_obs(obs_raw):
        return torch.tensor(obs_raw, dtype=torch.float32)
    
    obs = Obs(x=process_obs(obs_raw))
    reward = torch.zeros(16, 1)
    done = torch.zeros(16, 1, dtype=torch.bool)
    
    while total < episodes:
        with torch.no_grad():
            action, _, _, _, _, _ = brain.step(obs, reward, done, learn=False)
        
        obs_raw, rew, done_np, info = eval_env.step(action.numpy())
        
        # Count successes (reward > 0 means goal reached)
        for i, (r, d) in enumerate(zip(rew, done_np)):
            if d:
                total += 1
                if r > 0:
                    successes += 1
        
        reward = torch.tensor(rew, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done_np, dtype=torch.bool).unsqueeze(1)
        obs = Obs(x=process_obs(obs_raw))
        
        if done.any():
            brain.reset(batch_size=16)
    
    return successes / max(total, 1)


def main():
    seeds = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    
    print("="*60)
    print("MULTI-SEED VALIDATION")
    print("="*60)
    print(f"Testing {len(seeds)} seeds with 20k training steps each")
    print()
    
    results = {}
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"SEED {seed}")
        print(f"{'='*60}")
        
        start_time = time.time()
        result = train_and_evaluate(seed, train_steps=20000, eval_episodes=200)
        elapsed = time.time() - start_time
        
        results[seed] = result
        print(f"\nSeed {seed} complete in {elapsed:.1f}s")
        print(f"  Final SR: {result['final_sr']:.3f}")
        print(f"  Best SR:  {result['best_sr']:.3f}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    final_srs = [r['final_sr'] for r in results.values()]
    best_srs = [r['best_sr'] for r in results.values()]
    
    print(f"\nFinal Success Rates:")
    for seed, result in results.items():
        status = "✅" if result['final_sr'] >= 0.70 else "⚠️" if result['final_sr'] >= 0.40 else "❌"
        print(f"  Seed {seed}: {result['final_sr']:.3f} {status}")
    
    print(f"\nAggregate Statistics:")
    print(f"  Mean Final SR: {np.mean(final_srs):.3f} ± {np.std(final_srs):.3f}")
    print(f"  Mean Best SR:  {np.mean(best_srs):.3f} ± {np.std(best_srs):.3f}")
    print(f"  Range: [{np.min(final_srs):.3f}, {np.max(final_srs):.3f}]")
    
    # Success criteria check
    print(f"\n{'='*60}")
    print("SUCCESS CRITERIA CHECK")
    print(f"{'='*60}")
    
    mean_sr = np.mean(final_srs)
    std_sr = np.std(final_srs)
    min_sr = np.min(final_srs)
    
    print(f"  Mean SR > 0.60: {'✅' if mean_sr > 0.60 else '❌'} ({mean_sr:.3f})")
    print(f"  Std SR < 0.15:  {'✅' if std_sr < 0.15 else '❌'} ({std_sr:.3f})")
    print(f"  Min SR > 0.30:  {'✅' if min_sr > 0.30 else '❌'} ({min_sr:.3f})")
    
    if mean_sr > 0.60 and std_sr < 0.15 and min_sr > 0.30:
        print(f"\n🎉 ALL CRITERIA MET - STABILITY WORKED!")
    elif mean_sr > 0.45:
        print(f"\n⚠️ PARTIAL SUCCESS - Variance still present")
    else:
        print(f"\n❌ CRITERIA NOT MET - Need further investigation")


if __name__ == "__main__":
    main()
