"""
Continual Language Learning Training Script

Train MBM on English first, then French, measuring forgetting.
Demonstrates that hippocampus prevents catastrophic forgetting.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs, ModSignals
from digital_brain.envs.language_gridworld import TorchVectorLanguageGridworld


@dataclass
class ContinualResults:
    """Results from continual learning experiment."""
    config_name: str
    english_before_french: float
    english_after_french: float
    french_accuracy: float
    forgetting_percent: float
    backward_transfer: float
    seed: int


def create_brain_config(args, ablation_config: Optional[Dict] = None) -> dict:
    """Create brain configuration with optional ablations."""
    config = {
        'd_obs': 9,
        'd_z': args.d_z,
        'd_sel': 64,
        'd_act': 4,
        'use_language': True,
        'vocab_size': 50,
        'd_lang_embed': args.d_lang_embed,
        'd_lang_hidden': args.d_lang_hidden,
        # Default: all biological components enabled
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': True,
        'use_cerebellum': True,
        # Training params
        'lr': args.lr,
        'num_envs': args.num_envs,
        'num_steps': args.num_steps,
        'ppo_epochs': 4,
        'mini_batch_size': args.mini_batch_size,
        'eps_clip': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'vf_clip': 0.2,
        'target_kl': 0.015,
        'seed': args.seed,
    }
    
    # Apply ablation overrides
    if ablation_config:
        config.update(ablation_config)
    
    return config


def evaluate_language(brain: DigitalBrain,
                     env: TorchVectorLanguageGridworld,
                     device: torch.device,
                     language: str = 'english',
                     episodes: int = 100,
                     max_steps: int = 50) -> float:
    """Evaluate brain on all 9 combinations in specified language."""
    brain.eval()
    all_combos = [(c, s) for c in range(3) for s in range(3)]
    total_success = 0
    total_episodes = 0
    
    with torch.no_grad():
        for color_idx, shape_idx in all_combos:
            combo_episodes = 0
            target_episodes = episodes // 9
            
            while combo_episodes < target_episodes:
                obs, instructions = env.reset(
                    target_combo=(color_idx, shape_idx),
                    language=language
                )
                brain.reset(env.num_envs, device=device)
                
                prev_reward = torch.zeros(env.num_envs, 1, device=device)
                prev_done = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=device)
                
                ep_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
                ep_success = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
                
                for step in range(max_steps):
                    obs_obj = Obs(x=obs, ctx=env.get_context())
                    action, _, _, _, _, _ = brain.act(
                        obs_obj, prev_reward, prev_done, instruction=instructions
                    )
                    
                    obs, reward, done, info = env.step(action)
                    
                    new_done = done & ~ep_done
                    ep_success = ep_success | (info['success'] & new_done)
                    ep_done = ep_done | done
                    
                    prev_reward = reward.float().unsqueeze(1)
                    prev_done = done.unsqueeze(1)
                    
                    if ep_done.all():
                        break
                
                total_success += ep_success.sum().item()
                combo_episodes += env.num_envs
                total_episodes += env.num_envs
    
    brain.train()
    return total_success / total_episodes


def train_language(brain: DigitalBrain,
                  env: TorchVectorLanguageGridworld,
                  optimizer: optim.Optimizer,
                  config: dict,
                  device: torch.device,
                  language: str,
                  num_updates: int,
                  eval_env: TorchVectorLanguageGridworld,
                  eval_every: int = 10,
                  verbose: bool = True) -> float:
    """Train on specific language, return final accuracy."""
    T = config['num_steps']
    E = config['num_envs']
    all_combos = [(c, s) for c in range(3) for s in range(3)]
    
    best_acc = 0.0
    
    for update in range(num_updates):
        # Buffers
        obs_buf = torch.zeros((T, E, config['d_obs']), dtype=torch.float32)
        inst_buf = torch.zeros((T, E, 6), dtype=torch.long)
        act_buf = torch.zeros((T, E), dtype=torch.long)
        logp_buf = torch.zeros((T, E), dtype=torch.float32)
        val_buf = torch.zeros((T, E), dtype=torch.float32)
        rew_buf = torch.zeros((T, E), dtype=torch.float32)
        done_buf = torch.zeros((T, E), dtype=torch.bool)
        prev_rew_buf = torch.zeros((T, E), dtype=torch.float32)
        prev_done_buf = torch.zeros((T, E), dtype=torch.bool)
        
        # Reset with random target
        combo_idx = np.random.randint(0, len(all_combos))
        target_combo = all_combos[combo_idx]
        obs, instructions = env.reset(target_combo=target_combo, language=language)
        brain.reset(E, device=device)
        
        prev_reward = torch.zeros(E, 1, device=device)
        prev_done = torch.zeros(E, 1, dtype=torch.bool, device=device)
        
        # Collect experience
        with torch.inference_mode():
            for t in range(T):
                obs_buf[t] = obs.cpu()
                inst_buf[t] = instructions.cpu()
                prev_rew_buf[t] = prev_reward.squeeze(-1).cpu()
                prev_done_buf[t] = prev_done.squeeze(-1).cpu()
                
                obs_obj = Obs(x=obs, ctx=env.get_context())
                action, log_prob, value, _, _, _ = brain.step(
                    obs_obj, prev_reward, prev_done, learn=True, instruction=instructions
                )
                
                obs, reward, done, info = env.step(action)
                
                if done.any():
                    combo_idx = np.random.randint(0, len(all_combos))
                    target_combo = all_combos[combo_idx]
                    obs, instructions = env.reset(
                        env_mask=done, target_combo=target_combo, language=language
                    )
                
                act_buf[t] = action.cpu()
                logp_buf[t] = log_prob.float().cpu()
                val_buf[t] = value.squeeze(-1).float().cpu()
                rew_buf[t] = reward.cpu()
                done_buf[t] = done.cpu()
                
                prev_reward = reward.float().unsqueeze(1)
                prev_done = done.unsqueeze(1)
        
        # Compute GAE
        with torch.inference_mode():
            gated_last = brain.thalamus.gate(obs, brain._prev_selection, brain._prev_mods)
            z_last, _, _ = brain.cortex.forward(gated_last, brain.state.cortex_state, update_trace=False)
            
            if brain.use_language:
                lang_h = brain.lang_encoder(instructions)
                z_combined = torch.cat([z_last, lang_h], dim=-1)
                z_last = brain.lang_projection(z_combined)
            
            next_value = brain.bg.value_head(z_last).squeeze(-1).float().cpu()
        
        adv_buf = torch.zeros((T, E), dtype=torch.float32)
        last_gae_lam = torch.zeros(E, dtype=torch.float32)
        for t in reversed(range(T)):
            next_non_terminal = (~done_buf[t]).float()
            next_values = next_value if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + config['gamma'] * next_values * next_non_terminal - val_buf[t]
            last_gae_lam = delta + config['gamma'] * config['gae_lambda'] * next_non_terminal * last_gae_lam
            adv_buf[t] = last_gae_lam
        
        ret_buf = adv_buf + val_buf
        
        # Save collector state
        collector_state = (
            brain.state,
            brain._prev_selection.clone(),
            ModSignals(DA=brain._prev_mods.DA.clone(), NE=brain._prev_mods.NE.clone(),
                       ACh=brain._prev_mods.ACh.clone(), HT5=brain._prev_mods.HT5.clone()),
            brain._prev_pred.clone()
        )
        
        # PPO Update
        envs_per_batch = config['mini_batch_size'] // T
        
        for epoch in range(config['ppo_epochs']):
            env_perm = torch.randperm(E)
            
            for batch_start in range(0, E, envs_per_batch):
                batch_end = min(batch_start + envs_per_batch, E)
                env_idx = env_perm[batch_start:batch_end]
                M = len(env_idx)
                
                obs_seq = obs_buf[:, env_idx].to(device)
                inst_seq = inst_buf[:, env_idx].to(device)
                act_seq = act_buf[:, env_idx].to(device)
                logp_old_seq = logp_buf[:, env_idx].to(device)
                adv_seq = adv_buf[:, env_idx].to(device)
                ret_seq = ret_buf[:, env_idx].to(device)
                val_old_seq = val_buf[:, env_idx].to(device)
                done_seq = done_buf[:, env_idx].to(device)
                
                adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)
                
                brain.reset(M, device=device)
                
                policy_losses = []
                value_losses = []
                entropies = []
                
                for t in range(T):
                    if t > 0:
                        done_mask = done_seq[t-1].view(M, 1)
                        if done_mask.any():
                            with torch.no_grad():
                                brain.state.z = brain.state.z * (~done_mask).float()
                                e_act, i_act, trace_ee = brain.state.cortex_state
                                e_act = e_act * (~done_mask).float()
                                i_act = i_act * (~done_mask).float()
                                brain.state.cortex_state = (e_act, i_act, trace_ee)
                                brain.state.bg_state['prev_value'] = brain.state.bg_state['prev_value'] * (~done_mask).float()
                                brain._prev_selection = brain._prev_selection * (~done_mask).float()
                                brain._prev_pred = brain._prev_pred * (~done_mask).float()
                    
                    obs_t = obs_seq[t]
                    inst_t = inst_seq[t]
                    
                    gated_x = brain.thalamus.gate(obs_t, brain._prev_selection, brain._prev_mods)
                    z_t, pred_t, new_cortex_state = brain.cortex.forward(
                        gated_x, brain.state.cortex_state, update_trace=False
                    )
                    
                    if brain.use_language:
                        lang_h = brain.lang_encoder(inst_t)
                        z_combined = torch.cat([z_t, lang_h], dim=-1)
                        z_t = brain.lang_projection(z_combined)
                    
                    retrieved = brain.hippocampus.retrieve(z_t) if config.get('use_hippocampus', True) else None
                    correction, _ = brain.cerebellum.forward(z_t, obs_t) if config.get('use_cerebellum', True) else (None, None)
                    
                    selection, da, _, new_logp, value, entropy = brain.bg.step(
                        z_t,
                        torch.zeros(M, 1, device=device),
                        None,
                        torch.zeros(M, 1, dtype=torch.bool, device=device),
                        brain.state.bg_state['prev_value'],
                        memory_context=retrieved if config.get('use_memory_policy', True) else None,
                        cerebellum_correction=correction,
                        action_to_eval=act_seq[t]
                    )
                    
                    new_val = value.squeeze(-1).float()
                    
                    with torch.no_grad():
                        brain.state.z = z_t.detach()
                        brain.state.cortex_state = tuple(s.detach() for s in new_cortex_state)
                        brain.state.bg_state['prev_value'] = new_val.unsqueeze(-1).detach()
                        brain._prev_pred = pred_t.detach()
                    
                    log_ratio = new_logp - logp_old_seq[t]
                    ratio = torch.exp(log_ratio)
                    mb_adv = adv_seq[t]
                    
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * mb_adv
                    policy_loss_t = -torch.min(surr1, surr2).mean()
                    
                    old_val = val_old_seq[t]
                    v_clipped = old_val + (new_val - old_val).clamp(-config['vf_clip'], config['vf_clip'])
                    v_loss1 = (new_val - ret_seq[t])**2
                    v_loss2 = (v_clipped - ret_seq[t])**2
                    val_loss_t = 0.5 * torch.max(v_loss1, v_loss2).mean()
                    
                    policy_losses.append(policy_loss_t)
                    value_losses.append(val_loss_t)
                    entropies.append(entropy.mean())
                
                policy_loss = torch.stack(policy_losses).mean()
                val_loss = torch.stack(value_losses).mean()
                entropy_loss = torch.stack(entropies).mean()
                
                loss = policy_loss + config['value_coef'] * val_loss - config['entropy_coef'] * entropy_loss
                
                if not torch.isnan(loss):
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                    optimizer.step()
        
        # Restore collector state
        brain.state = collector_state[0]
        brain._prev_selection = collector_state[1]
        brain._prev_mods = collector_state[2]
        brain._prev_pred = collector_state[3]
        
        # Evaluate
        if (update + 1) % eval_every == 0:
            acc = evaluate_language(brain, eval_env, device, language=language)
            if acc > best_acc:
                best_acc = acc
            if verbose:
                print(f"  [{language.upper():7s}] Update {update+1:3d}: {acc*100:.1f}% (best: {best_acc*100:.1f}%)")
    
    return best_acc


def run_continual_experiment(args, ablation_config: Dict, config_name: str) -> ContinualResults:
    """Run single continual learning experiment (English → French)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    config = create_brain_config(args, ablation_config)
    
    # Create environments
    env = TorchVectorLanguageGridworld(
        num_envs=config['num_envs'],
        size=args.grid_size,
        device=device,
        seed=args.seed
    )
    
    eval_env = TorchVectorLanguageGridworld(
        num_envs=64,
        size=args.grid_size,
        device=device,
        seed=args.seed + 1000
    )
    
    # Create brain
    brain = DigitalBrain(config).to(device)
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'], eps=1e-5)
    
    print(f"\n{'='*60}")
    print(f"Config: {config_name}")
    print(f"  use_hippocampus: {config.get('use_hippocampus', True)}")
    print(f"  use_plasticity: {config.get('use_plasticity', True)}")
    print(f"{'='*60}")
    
    # Stage 1: Train English
    print("\nStage 1: Training English...")
    train_language(brain, env, optimizer, config, device, 
                  language='english', num_updates=args.stage_updates,
                  eval_env=eval_env, eval_every=args.eval_every)
    
    english_before = evaluate_language(brain, eval_env, device, language='english')
    print(f"  English accuracy BEFORE French: {english_before*100:.1f}%")
    
    # Stage 2: Train French (continual learning)
    print("\nStage 2: Training French (continual)...")
    train_language(brain, env, optimizer, config, device,
                  language='french', num_updates=args.stage_updates,
                  eval_env=eval_env, eval_every=args.eval_every)
    
    english_after = evaluate_language(brain, eval_env, device, language='english')
    french_acc = evaluate_language(brain, eval_env, device, language='french')
    
    # Compute metrics
    if english_before > 0:
        forgetting = (english_before - english_after) / english_before * 100
    else:
        forgetting = 0.0
    
    backward_transfer = english_after - 0.33  # Compare to random baseline
    
    print(f"\n  English AFTER French: {english_after*100:.1f}%")
    print(f"  French accuracy: {french_acc*100:.1f}%")
    print(f"  Forgetting: {forgetting:.1f}%")
    
    return ContinualResults(
        config_name=config_name,
        english_before_french=english_before,
        english_after_french=english_after,
        french_accuracy=french_acc,
        forgetting_percent=forgetting,
        backward_transfer=backward_transfer,
        seed=args.seed
    )


def run_ablation_study(args):
    """Run ablation study comparing different configurations."""
    print("=" * 70)
    print("ABLATION STUDY: Hippocampus Effect on Continual Language Learning")
    print("=" * 70)
    
    ablation_configs = [
        {'name': 'Full MBM', 'config': {'use_hippocampus': True, 'use_plasticity': True}},
        {'name': 'No Hippocampus', 'config': {'use_hippocampus': False, 'use_plasticity': True}},
        {'name': 'No Plasticity', 'config': {'use_hippocampus': True, 'use_plasticity': False}},
        {'name': 'Baseline', 'config': {'use_hippocampus': False, 'use_plasticity': False}},
    ]
    
    all_results = []
    
    for seed in range(args.seed, args.seed + args.num_seeds):
        args.seed = seed
        print(f"\n{'#'*70}")
        print(f"SEED {seed}")
        print(f"{'#'*70}")
        
        for ablation in ablation_configs:
            result = run_continual_experiment(args, ablation['config'], ablation['name'])
            all_results.append(result)
    
    # Aggregate results
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    
    # Group by config name
    config_results = {}
    for result in all_results:
        if result.config_name not in config_results:
            config_results[result.config_name] = []
        config_results[result.config_name].append(result)
    
    print("\n| Config | English Before | English After | Forgetting | French Acc |")
    print("|--------|----------------|---------------|------------|------------|")
    
    summary = {}
    for config_name, results in config_results.items():
        en_before = np.mean([r.english_before_french for r in results])
        en_after = np.mean([r.english_after_french for r in results])
        forgetting = np.mean([r.forgetting_percent for r in results])
        french = np.mean([r.french_accuracy for r in results])
        
        en_before_std = np.std([r.english_before_french for r in results])
        forgetting_std = np.std([r.forgetting_percent for r in results])
        
        print(f"| {config_name:14s} | {en_before*100:5.1f}% ± {en_before_std*100:4.1f}% | "
              f"{en_after*100:5.1f}% | {forgetting:5.1f}% ± {forgetting_std:4.1f}% | {french*100:5.1f}% |")
        
        summary[config_name] = {
            'english_before': en_before,
            'english_after': en_after,
            'forgetting': forgetting,
            'french': french,
            'forgetting_std': forgetting_std,
        }
    
    # Statistical comparison
    if 'Full MBM' in config_results and 'No Hippocampus' in config_results:
        full_forgetting = [r.forgetting_percent for r in config_results['Full MBM']]
        no_hip_forgetting = [r.forgetting_percent for r in config_results['No Hippocampus']]
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(full_forgetting, no_hip_forgetting)
        
        print(f"\nStatistical Significance (Full MBM vs No Hippocampus):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant (p < 0.05): {p_value < 0.05}")
    
    # Save results
    results_path = "experiments/continual_language_results.json"
    os.makedirs("experiments", exist_ok=True)
    
    results_dict = {
        'summary': summary,
        'raw_results': [asdict(r) for r in all_results],
        'args': vars(args)
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to {results_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Continual Language Learning")
    parser.add_argument('--mode', type=str, default='ablation',
                       choices=['single', 'ablation'],
                       help='Run single experiment or full ablation')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_seeds', type=int, default=3)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--d_z', type=int, default=256)
    parser.add_argument('--d_lang_embed', type=int, default=64)
    parser.add_argument('--d_lang_hidden', type=int, default=128)
    parser.add_argument('--num_envs', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--stage_updates', type=int, default=100,
                       help='Updates per language stage')
    parser.add_argument('--eval_every', type=int, default=20)
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.mode == 'single':
        result = run_continual_experiment(
            args, 
            {'use_hippocampus': True, 'use_plasticity': True},
            'Full MBM'
        )
        print(f"\nFinal Result: {asdict(result)}")
    else:
        run_ablation_study(args)


if __name__ == "__main__":
    main()
