"""
Train and Test Semantic Understanding

Phase 3: Test negation reasoning, synonym understanding, and relational concepts.

Tests:
1. Negation: "go to circle that is NOT blue" - train on positive, test on negation
2. Synonyms: "crimson" -> red (few-shot transfer)
3. Relations: "go to circle NEAR square"
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import json

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.semantic_env import VectorizedSemanticEnv, evaluate_semantic


def create_config(vocab_size: int = 25, device: str = 'cuda') -> dict:
    return {
        'd_obs': 28,  # 5x5 + 3 metadata
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


def train_semantic(
    mode: str = 'negation',
    num_updates: int = 300,
    num_envs: int = 128,
    num_steps: int = 32,
    learning_rate: float = 3e-4,
    eval_interval: int = 25,
    seed: int = 42,
    device: str = 'cuda'
):
    """Train on semantic understanding task."""
    
    print("=" * 70)
    print(f"SEMANTIC UNDERSTANDING: {mode.upper()}")
    print("=" * 70)
    print(f"Seed: {seed}")
    print("=" * 70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        device = 'cpu'
    
    env = VectorizedSemanticEnv(num_envs=num_envs, size=5, mode=mode)
    config = create_config(vocab_size=env.vocab_size, device=device)
    brain = DigitalBrain(config).to(device)
    
    optimizer = torch.optim.Adam(brain.parameters(), lr=learning_rate)
    
    best_acc = 0
    best_syn_acc = 0  # For synonym mode
    
    print(f"\nTraining for {num_updates} updates...")
    print("-" * 70)
    
    for update in range(1, num_updates + 1):
        # Collect experience
        obs_buf, inst_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], [], []
        
        # For synonym mode, train on standard colors (not synonyms)
        use_syn = False
        obs, instructions = env.reset(use_synonym=use_syn)
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
            # Standard evaluation
            res = evaluate_semantic(brain, env, 200, use_synonym=False, device=device)
            
            if res['accuracy'] > best_acc:
                best_acc = res['accuracy']
            
            output = f"Update {update:4d}: Acc={res['accuracy']:5.1f}% (best:{best_acc:.1f}%)"
            
            # For synonym mode, also test with synonyms (zero-shot transfer)
            if mode == 'synonym':
                syn_res = evaluate_semantic(brain, env, 200, use_synonym=True, device=device)
                if syn_res['accuracy'] > best_syn_acc:
                    best_syn_acc = syn_res['accuracy']
                output += f" | Synonym={syn_res['accuracy']:.1f}% (best:{best_syn_acc:.1f}%)"
            
            print(output)
    
    print("-" * 70)
    print(f"\nBest Standard Accuracy: {best_acc:.1f}%")
    if mode == 'synonym':
        print(f"Best Synonym Transfer: {best_syn_acc:.1f}%")
    print(f"Random baseline: ~33% (3 objects)")
    
    # Success criteria based on mode
    if mode == 'negation':
        success = best_acc > 60
        print(f"\nNegation >60%: {'✓ PASS' if success else '✗ FAIL'} ({best_acc:.1f}%)")
    elif mode == 'synonym':
        success = best_syn_acc > 40  # Lower bar for zero-shot
        print(f"\nSynonym transfer >40%: {'✓ PASS' if success else '✗ FAIL'} ({best_syn_acc:.1f}%)")
    else:
        success = best_acc > 60
        print(f"\nRelation >60%: {'✓ PASS' if success else '✗ FAIL'} ({best_acc:.1f}%)")
    
    return {
        'mode': mode,
        'best_acc': best_acc,
        'best_syn_acc': best_syn_acc if mode == 'synonym' else None,
        'success': success,
        'brain': brain
    }


def run_all_semantic_tests(seeds: list = [42, 43, 44], device: str = 'cuda'):
    """Run all semantic understanding tests."""
    
    print("\n" + "#" * 70)
    print("# COMPREHENSIVE SEMANTIC UNDERSTANDING TESTS")
    print("#" * 70)
    
    all_results = {}
    
    for mode in ['negation', 'relation', 'synonym']:
        print(f"\n{'#' * 70}")
        print(f"# MODE: {mode.upper()}")
        print('#' * 70)
        
        mode_results = []
        
        for seed in seeds:
            result = train_semantic(
                mode=mode,
                num_updates=200,
                seed=seed,
                device=device
            )
            mode_results.append({
                'seed': seed,
                'acc': result['best_acc'],
                'syn_acc': result.get('best_syn_acc'),
                'success': result['success']
            })
        
        # Aggregate
        accs = [r['acc'] for r in mode_results]
        all_results[mode] = {
            'mean_acc': np.mean(accs),
            'std_acc': np.std(accs),
            'results': mode_results
        }
        
        if mode == 'synonym':
            syn_accs = [r['syn_acc'] for r in mode_results if r['syn_acc'] is not None]
            all_results[mode]['mean_syn_acc'] = np.mean(syn_accs)
            all_results[mode]['std_syn_acc'] = np.std(syn_accs)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for mode, res in all_results.items():
        print(f"\n{mode.upper()}:")
        print(f"  Standard: {res['mean_acc']:.1f}% ± {res['std_acc']:.1f}%")
        if 'mean_syn_acc' in res:
            print(f"  Synonym:  {res['mean_syn_acc']:.1f}% ± {res['std_syn_acc']:.1f}%")
    
    # Save results
    with open('experiments/semantic_results.json', 'w') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'results'} 
                  for k, v in all_results.items()}, f, indent=2)
    print("\nResults saved to experiments/semantic_results.json")
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['negation', 'synonym', 'relation', 'all'])
    parser.add_argument('--updates', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.mode == 'all':
        run_all_semantic_tests(seeds=[42, 43, 44], device=args.device)
    else:
        train_semantic(mode=args.mode, num_updates=args.updates, 
                      seed=args.seed, device=args.device)
