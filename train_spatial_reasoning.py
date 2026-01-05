"""
Train Digital Brain on Spatial Reasoning Task

Phase 1, Task 1: Test whether MBM builds internal world models through spatial relationships.

Training: 4 relations (left, right, above, below)
Testing: Novel relation 'near' (never seen during training)

Success criteria:
- Training accuracy >85% (understands trained relations)
- Test accuracy on 'near' >60% (generalizes to new relation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from datetime import datetime

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.spatial_reasoning import VectorizedSpatialEnv, evaluate_spatial_reasoning


def create_spatial_brain_config(vocab_size: int = 15, device: str = 'cuda') -> dict:
    """Create brain configuration for spatial reasoning task."""
    return {
        # Core dimensions matching DigitalBrain expectations
        'd_obs': 196,  # 4 channels x 7x7 grid = 196
        'd_z': 128,    # Latent dimension
        'd_sel': 64,   # Selection dimension
        'd_act': 6,    # Actions: 4 movement + pick + place
        
        # Language encoder
        'use_language': True,
        'vocab_size': vocab_size,
        'd_lang_embed': 64,
        'd_lang_hidden': 128,
        
        # Biological components
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': False,
        'use_cerebellum': True,
        
        # Device
        'device': device,
    }


def collect_experience(brain, env, num_steps: int, device: str):
    """Collect experience for PPO training."""
    obs_buf = []
    inst_buf = []
    act_buf = []
    rew_buf = []
    done_buf = []
    val_buf = []
    logp_buf = []
    
    obs, instructions = env.reset()
    obs = obs.to(device)
    instructions = instructions.to(device)
    
    brain.reset(env.num_envs)
    
    # Initialize reward/done for first step
    prev_reward = torch.zeros(env.num_envs, device=device)
    prev_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
    
    for step in range(num_steps):
        with torch.no_grad():
            # Flatten obs for brain: [batch, 4, 7, 7] -> [batch, 196]
            obs_flat = obs.view(obs.size(0), -1)
            
            # Wrap observation in Obs object
            obs_wrapped = Obs(x=obs_flat)
            
            # Call brain.step - returns tuple: (action, log_prob, value, state, log, entropy)
            action, logp, value, state, log, entropy = brain.step(
                obs_wrapped, prev_reward, prev_done, 
                learn=False, instruction=instructions
            )
            
            # Value comes as [B, 1], squeeze to [B]
            value = value.squeeze(-1)
        
        obs_buf.append(obs_flat.cpu())
        inst_buf.append(instructions.cpu())
        act_buf.append(action.cpu())
        logp_buf.append(logp.cpu())
        val_buf.append(value.cpu())
        
        # Step environment
        obs, rewards, dones, infos = env.step(action)
        obs = obs.to(device)
        
        rew_buf.append(rewards)
        done_buf.append(dones)
        
        # Update for next iteration
        prev_reward = rewards.to(device)
        prev_done = dones.to(device)
        instructions = env.get_instructions().to(device)
    
    # Stack buffers
    experience = {
        'obs': torch.stack(obs_buf),           # [T, B, obs_dim]
        'instructions': torch.stack(inst_buf), # [T, B, seq_len]
        'actions': torch.stack(act_buf),       # [T, B]
        'rewards': torch.stack(rew_buf),       # [T, B]
        'dones': torch.stack(done_buf),        # [T, B]
        'values': torch.stack(val_buf),        # [T, B]
        'log_probs': torch.stack(logp_buf),    # [T, B]
    }
    
    return experience


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation."""
    T, B = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = values[t]  # Bootstrap from last value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (~dones[t]).float() - values[t]
        advantages[t] = last_gae = delta + gamma * lam * (~dones[t]).float() * last_gae
    
    returns = advantages + values
    return advantages, returns


def ppo_update(brain, experience, optimizer, epochs=4, mini_batch_size=512, 
               clip_eps=0.2, vf_coef=0.5, ent_coef=0.01, device='cuda'):
    """PPO update step."""
    obs = experience['obs'].to(device)
    instructions = experience['instructions'].to(device)
    actions = experience['actions'].to(device)
    old_log_probs = experience['log_probs'].to(device)
    advantages = experience['advantages'].to(device)
    returns = experience['returns'].to(device)
    
    T, B = obs.shape[:2]
    total_samples = T * B
    
    # Flatten
    obs_flat = obs.view(total_samples, -1)
    inst_flat = instructions.view(total_samples, -1)
    act_flat = actions.view(total_samples)
    oldlp_flat = old_log_probs.view(total_samples)
    adv_flat = advantages.view(total_samples)
    ret_flat = returns.view(total_samples)
    
    # Normalize advantages
    adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
    
    total_loss = 0
    num_updates = 0
    
    for epoch in range(epochs):
        indices = torch.randperm(total_samples, device=device)
        
        for start in range(0, total_samples, mini_batch_size):
            end = min(start + mini_batch_size, total_samples)
            idx = indices[start:end]
            
            mb_obs = obs_flat[idx]
            mb_inst = inst_flat[idx]
            mb_act = act_flat[idx]
            mb_oldlp = oldlp_flat[idx]
            mb_adv = adv_flat[idx]
            mb_ret = ret_flat[idx]
            
            # Forward pass - use action_to_eval to get log_prob for specific actions
            brain.reset(len(idx))
            obs_wrapped = Obs(x=mb_obs)
            
            # Manually compute forward pass to get logits
            # This mimics brain.step but lets us compute log_prob for mb_act
            # cortex returns (z_t, pred_t, new_state) tuple
            z_t, pred_t, new_state = brain.cortex(obs_wrapped.x, brain.state.cortex_state)
            
            # Language fusion if enabled
            if brain.use_language and mb_inst is not None:
                lang_h = brain.lang_encoder(mb_inst)
                z_combined = torch.cat([z_t, lang_h], dim=-1)
                z_t = brain.lang_projection(z_combined)
            
            # Get policy logits from BG
            logits = brain.bg.policy_head(z_t)
            logits = torch.clamp(logits, min=-20, max=20)
            
            # Get value
            values = brain.bg.value_head(z_t).squeeze(-1)
            
            # Policy loss
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(mb_act)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_log_probs - mb_oldlp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, mb_ret)
            
            # Total loss
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            num_updates += 1
    
    return total_loss / max(num_updates, 1)


def train_spatial_reasoning(
    num_updates: int = 200,
    num_envs: int = 256,
    num_steps: int = 64,
    mini_batch_size: int = 2048,
    learning_rate: float = 3e-4,
    eval_interval: int = 25,
    seed: int = 42,
    device: str = 'cuda'
):
    """Train brain on spatial reasoning task."""
    
    print("=" * 70)
    print("SPATIAL REASONING TRAINING")
    print("=" * 70)
    print(f"Training relations: left, right, above, below")
    print(f"Test relation: near (novel)")
    print(f"Seed: {seed}")
    print("=" * 70)
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        device = 'cpu'
    
    # Create environment
    env = VectorizedSpatialEnv(num_envs=num_envs, size=7)
    
    # Create brain
    config = create_spatial_brain_config(vocab_size=env.vocab_size, device=device)
    brain = DigitalBrain(config).to(device)
    
    # Optimizer
    optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)
    
    # Training metrics
    best_train_acc = 0
    best_test_acc = 0
    history = {
        'train_acc': [],
        'test_acc': [],
        'loss': [],
        'relation_acc': []
    }
    
    print(f"\nStarting training for {num_updates} updates...")
    print("-" * 70)
    
    for update in range(1, num_updates + 1):
        # Collect experience (only on training relations)
        experience = collect_experience(brain, env, num_steps, device)
        
        # Compute advantages
        advantages, returns = compute_gae(
            experience['rewards'],
            experience['values'],
            experience['dones']
        )
        experience['advantages'] = advantages
        experience['returns'] = returns
        
        # PPO update
        loss = ppo_update(brain, experience, optimizer, 
                         mini_batch_size=mini_batch_size, device=device)
        
        history['loss'].append(loss)
        
        # Evaluation
        if update % eval_interval == 0:
            # Evaluate on training relations
            train_results = evaluate_spatial_reasoning(
                brain, env, num_episodes=100, 
                use_test_relations=False, device=device
            )
            train_acc = train_results['accuracy']
            
            # Evaluate on test relation (novel 'near')
            test_results = evaluate_spatial_reasoning(
                brain, env, num_episodes=100,
                use_test_relations=True, device=device
            )
            test_acc = test_results['accuracy']
            
            history['train_acc'].append(train_acc)
            history['test_acc'].append(test_acc)
            history['relation_acc'].append(train_results.get('relation_accuracy', {}))
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
            
            print(f"Update {update:4d}: Train={train_acc:5.1f}% (best: {best_train_acc:.1f}%) | "
                  f"Test(near)={test_acc:5.1f}% (best: {best_test_acc:.1f}%) | "
                  f"Loss={loss:.4f}")
            
            # Per-relation breakdown
            if train_results.get('relation_accuracy'):
                rel_str = " | ".join([f"{r}:{a:.0f}%" 
                                     for r, a in train_results['relation_accuracy'].items()])
                print(f"         Relations: {rel_str}")
    
    print("-" * 70)
    print(f"\nTraining complete!")
    print(f"Best training accuracy: {best_train_acc:.1f}%")
    print(f"Best test accuracy (near): {best_test_acc:.1f}%")
    print(f"Random baseline: 20% (1/5 positions correct)")
    
    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA:")
    print(f"  Training >85%: {'✓ PASS' if best_train_acc > 85 else '✗ FAIL'} ({best_train_acc:.1f}%)")
    print(f"  Test >60%:     {'✓ PASS' if best_test_acc > 60 else '✗ FAIL'} ({best_test_acc:.1f}%)")
    print("=" * 70)
    
    return {
        'best_train_acc': best_train_acc,
        'best_test_acc': best_test_acc,
        'history': history,
        'brain': brain
    }


def run_ablation_spatial(seeds: list = [42, 43, 44], device: str = 'cuda'):
    """Run ablation study on spatial reasoning."""
    
    print("\n" + "#" * 70)
    print("# SPATIAL REASONING ABLATION STUDY")
    print("#" * 70)
    
    configs = {
        'Full MBM': {'use_hippocampus': True, 'use_plasticity': True},
        'No Hippocampus': {'use_hippocampus': False, 'use_plasticity': True},
        'No Plasticity': {'use_hippocampus': True, 'use_plasticity': False},
        'Baseline': {'use_hippocampus': False, 'use_plasticity': False},
    }
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"SEED {seed}")
        print('#' * 70)
        
        for config_name, config_overrides in configs.items():
            print(f"\n{'=' * 60}")
            print(f"Config: {config_name}")
            for k, v in config_overrides.items():
                print(f"  {k}: {v}")
            print('=' * 60)
            
            # Set seeds
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            # Create environment
            env = VectorizedSpatialEnv(num_envs=256, size=7)
            
            # Create brain with config overrides
            brain_config = create_spatial_brain_config(vocab_size=env.vocab_size, device=device)
            brain_config.update(config_overrides)
            brain = DigitalBrain(brain_config).to(device)
            
            optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)
            
            best_train = 0
            best_test = 0
            
            # Train
            for update in range(1, 151):
                experience = collect_experience(brain, env, 64, device)
                advantages, returns = compute_gae(experience['rewards'], 
                                                  experience['values'], 
                                                  experience['dones'])
                experience['advantages'] = advantages
                experience['returns'] = returns
                
                ppo_update(brain, experience, optimizer, mini_batch_size=2048, device=device)
                
                if update % 25 == 0:
                    train_res = evaluate_spatial_reasoning(brain, env, 100, 
                                                          use_test_relations=False, device=device)
                    test_res = evaluate_spatial_reasoning(brain, env, 100,
                                                         use_test_relations=True, device=device)
                    
                    if train_res['accuracy'] > best_train:
                        best_train = train_res['accuracy']
                    if test_res['accuracy'] > best_test:
                        best_test = test_res['accuracy']
                    
                    print(f"  Update {update:3d}: Train={train_res['accuracy']:.1f}% | "
                          f"Test(near)={test_res['accuracy']:.1f}%")
            
            key = f"{config_name}_seed{seed}"
            all_results[key] = {
                'config': config_name,
                'seed': seed,
                'train_acc': best_train,
                'test_acc': best_test
            }
            
            print(f"\n  Final: Train={best_train:.1f}% | Test(near)={best_test:.1f}%")
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    
    for config_name in configs.keys():
        train_accs = [all_results[f"{config_name}_seed{s}"]['train_acc'] for s in seeds]
        test_accs = [all_results[f"{config_name}_seed{s}"]['test_acc'] for s in seeds]
        
        print(f"\n{config_name}:")
        print(f"  Train: {np.mean(train_accs):.1f}% ± {np.std(train_accs):.1f}%")
        print(f"  Test (near): {np.mean(test_accs):.1f}% ± {np.std(test_accs):.1f}%")
    
    # Save results
    results_path = 'experiments/spatial_reasoning_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return all_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'ablation'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.mode == 'train':
        results = train_spatial_reasoning(
            num_updates=args.updates,
            seed=args.seed,
            device=args.device
        )
    else:
        results = run_ablation_spatial(
            seeds=[42, 43, 44],
            device=args.device
        )
