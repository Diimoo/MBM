"""
Train Digital Brain on Inference from Description Task

Phase 1, Task 2: Build world model from language, then act on it.
Agent receives language descriptions of object positions, then must navigate
to a target object using only the language-built world model.

Success criteria:
- Agent successfully navigates using language-built world model
- Shows language is necessary (not just vision)
"""

import torch
import torch.nn.functional as F
import numpy as np
import random

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.inference_env import VectorizedInferenceEnv, evaluate_inference


def create_config(vocab_size: int = 25, device: str = 'cuda') -> dict:
    return {
        'd_obs': 30,     # 5x5 grid + 5 metadata
        'd_z': 64,
        'd_sel': 32,
        'd_act': 4,
        'use_language': True,
        'vocab_size': vocab_size,
        'd_lang_embed': 32,
        'd_lang_hidden': 64,
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': False,
        'use_cerebellum': True,
        'device': device,
    }


def train_inference(
    num_updates: int = 400,
    num_envs: int = 128,
    num_steps: int = 40,
    learning_rate: float = 3e-4,
    eval_interval: int = 50,
    seed: int = 42,
    device: str = 'cuda'
):
    print("=" * 70)
    print("INFERENCE FROM DESCRIPTION TRAINING")
    print("=" * 70)
    print("Agent builds world model from language descriptions,")
    print("then navigates to target using language-built model.")
    print(f"Seed: {seed}")
    print("=" * 70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        device = 'cpu'
    
    env = VectorizedInferenceEnv(num_envs=num_envs, size=5)
    config = create_config(vocab_size=env.vocab_size, device=device)
    brain = DigitalBrain(config).to(device)
    
    optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)
    
    best_acc = 0
    
    print(f"\nTraining for {num_updates} updates...")
    print("-" * 70)
    
    for update in range(1, num_updates + 1):
        obs_buf, inst_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], [], []
        
        obs, instructions = env.reset()
        obs = obs.to(device)
        instructions = instructions.to(device)
        brain.reset(num_envs)
        
        prev_reward = torch.zeros(num_envs, device=device)
        prev_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for step in range(num_steps):
            with torch.no_grad():
                obs_wrapped = Obs(x=obs)
                action, logp, value, state, log, entropy = brain.step(
                    obs_wrapped, prev_reward, prev_done, learn=False, instruction=instructions
                )
                value = value.squeeze(-1)
            
            obs_buf.append(obs.cpu())
            inst_buf.append(instructions.cpu())
            act_buf.append(action.cpu())
            logp_buf.append(logp.cpu())
            val_buf.append(value.cpu())
            
            obs, rewards, dones, infos = env.step(action)
            obs = obs.to(device)
            instructions = env.get_instructions().to(device)
            
            rew_buf.append(rewards)
            done_buf.append(dones)
            
            prev_reward = rewards.to(device)
            prev_done = dones.to(device)
        
        # Stack and compute GAE
        obs_t = torch.stack(obs_buf)
        inst_t = torch.stack(inst_buf)
        act_t = torch.stack(act_buf)
        rew_t = torch.stack(rew_buf)
        done_t = torch.stack(done_buf)
        val_t = torch.stack(val_buf)
        logp_t = torch.stack(logp_buf)
        
        T, B = rew_t.shape
        advantages = torch.zeros_like(rew_t)
        last_gae = 0
        gamma, lam = 0.99, 0.95
        
        for t in reversed(range(T)):
            next_val = val_t[t] if t == T - 1 else val_t[t + 1]
            delta = rew_t[t] + gamma * next_val * (~done_t[t]).float() - val_t[t]
            advantages[t] = last_gae = delta + gamma * lam * (~done_t[t]).float() * last_gae
        
        returns = advantages + val_t
        
        # Flatten
        total = T * B
        obs_flat = obs_t.view(total, -1).to(device)
        inst_flat = inst_t.view(total, -1).to(device)
        act_flat = act_t.view(total).to(device)
        oldlp_flat = logp_t.view(total).to(device)
        adv_flat = advantages.view(total).to(device)
        ret_flat = returns.view(total).to(device)
        
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        
        # PPO update
        mini_batch = 512
        epochs = 4
        
        for epoch in range(epochs):
            indices = torch.randperm(total, device=device)
            
            for start in range(0, total, mini_batch):
                end = min(start + mini_batch, total)
                idx = indices[start:end]
                
                mb_obs = obs_flat[idx]
                mb_inst = inst_flat[idx]
                mb_act = act_flat[idx]
                mb_oldlp = oldlp_flat[idx]
                mb_adv = adv_flat[idx]
                mb_ret = ret_flat[idx]
                
                brain.reset(len(idx))
                z_t, _, _ = brain.cortex(mb_obs, brain.state.cortex_state)
                
                if brain.use_language:
                    lang_h = brain.lang_encoder(mb_inst)
                    z_combined = torch.cat([z_t, lang_h], dim=-1)
                    z_t = brain.lang_projection(z_combined)
                
                logits = torch.clamp(brain.bg.policy_head(z_t), -20, 20)
                values = brain.bg.value_head(z_t).squeeze(-1)
                
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(mb_act)
                ent = dist.entropy().mean()
                
                ratio = torch.exp(new_lp - mb_oldlp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 0.8, 1.2) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, mb_ret)
                
                loss = policy_loss + 0.5 * value_loss - 0.01 * ent
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                optimizer.step()
        
        # Evaluation
        if update % eval_interval == 0:
            res = evaluate_inference(brain, env, 200, device=device)
            
            if res['accuracy'] > best_acc:
                best_acc = res['accuracy']
            
            print(f"Update {update:4d}: Accuracy={res['accuracy']:5.1f}% (best:{best_acc:.1f}%)")
    
    print("-" * 70)
    print(f"\nBest Accuracy: {best_acc:.1f}%")
    print(f"Random baseline: ~20% (1/5 objects)")
    
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA:")
    print(f"  Accuracy >60%: {'✓ PASS' if best_acc > 60 else '✗ FAIL'} ({best_acc:.1f}%)")
    print("=" * 70)
    
    return {'best_acc': best_acc, 'brain': brain}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=400)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    train_inference(num_updates=args.updates, seed=args.seed, device=args.device)
