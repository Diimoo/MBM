"""
Language Grounding Training Script

Train MBM to understand language instructions in a gridworld environment.
Demonstrates compositional language learning and continual learning capabilities.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import argparse
from typing import Dict, List, Tuple, Optional

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs, ModSignals
from digital_brain.envs.language_gridworld import (
    TorchVectorLanguageGridworld, 
    get_train_test_split,
    get_all_instructions
)

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def create_brain_config(args) -> dict:
    """Create brain configuration for language learning."""
    return {
        'd_obs': 9,  # 3x3 local view
        'd_z': args.d_z,
        'd_sel': 64,
        'd_act': 4,  # Up, Down, Left, Right
        # Language settings
        'use_language': True,
        'vocab_size': 50,
        'd_lang_embed': args.d_lang_embed,
        'd_lang_hidden': args.d_lang_hidden,
        # Biological components
        'use_hippocampus': args.use_hippocampus,
        'use_plasticity': args.use_plasticity,
        'use_memory_policy': args.use_memory_policy,
        'use_cerebellum': args.use_cerebellum,
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


def evaluate_language(brain: DigitalBrain, 
                     env: TorchVectorLanguageGridworld,
                     device: torch.device,
                     combos: List[Tuple[int, int]],
                     episodes_per_combo: int = 20,
                     max_steps: int = 50,
                     language: str = 'english') -> Dict[str, float]:
    """
    Evaluate brain on specific color-shape combinations.
    
    Returns:
        Dict with per-combo accuracy and overall accuracy
    """
    brain.eval()
    results = {}
    total_success = 0
    total_episodes = 0
    
    with torch.no_grad():
        for color_idx, shape_idx in combos:
            combo_success = 0
            combo_episodes = 0
            
            # Run multiple episodes for this combo
            while combo_episodes < episodes_per_combo:
                # Reset with specific target
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
                    
                    # Track success (only for newly done episodes)
                    new_done = done & ~ep_done
                    ep_success = ep_success | (info['success'] & new_done)
                    ep_done = ep_done | done
                    
                    prev_reward = reward.float().unsqueeze(1)
                    prev_done = done.unsqueeze(1)
                    
                    if ep_done.all():
                        break
                
                combo_success += ep_success.sum().item()
                combo_episodes += env.num_envs
            
            combo_acc = combo_success / combo_episodes
            colors = ['red', 'blue', 'green']
            shapes = ['circle', 'square', 'triangle']
            combo_name = f"{colors[color_idx]}_{shapes[shape_idx]}"
            results[combo_name] = combo_acc
            
            total_success += combo_success
            total_episodes += combo_episodes
    
    results['overall'] = total_success / total_episodes
    brain.train()
    return results


def train_step(brain: DigitalBrain,
               env: TorchVectorLanguageGridworld,
               optimizer: optim.Optimizer,
               config: dict,
               device: torch.device,
               train_combos: Optional[List[Tuple[int, int]]] = None,
               language: str = 'english') -> Dict[str, float]:
    """
    Single training update with PPO.
    
    Args:
        train_combos: If provided, cycle through these specific combos.
                     If None, use random targets.
    """
    T = config['num_steps']
    E = config['num_envs']
    
    # Buffers
    obs_buf = torch.zeros((T, E, config['d_obs']), dtype=torch.float32)
    inst_buf = torch.zeros((T, E, 6), dtype=torch.long)  # Instructions
    act_buf = torch.zeros((T, E), dtype=torch.long)
    logp_buf = torch.zeros((T, E), dtype=torch.float32)
    val_buf = torch.zeros((T, E), dtype=torch.float32)
    rew_buf = torch.zeros((T, E), dtype=torch.float32)
    done_buf = torch.zeros((T, E), dtype=torch.bool)
    prev_rew_buf = torch.zeros((T, E), dtype=torch.float32)
    prev_done_buf = torch.zeros((T, E), dtype=torch.bool)
    
    # Reset env with optional curriculum
    if train_combos:
        combo_idx = np.random.randint(0, len(train_combos))
        target_combo = train_combos[combo_idx]
    else:
        target_combo = None
    
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
            
            # Auto-reset done envs with new target
            if done.any():
                if train_combos:
                    combo_idx = np.random.randint(0, len(train_combos))
                    target_combo = train_combos[combo_idx]
                else:
                    target_combo = None
                obs, instructions = env.reset(
                    env_mask=done, 
                    target_combo=target_combo,
                    language=language
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
        obs_obj = Obs(x=obs, ctx=env.get_context())
        gated_last = brain.thalamus.gate(obs, brain._prev_selection, brain._prev_mods)
        z_last, _, _ = brain.cortex.forward(gated_last, brain.state.cortex_state, update_trace=False)
        
        # Add language encoding for value estimation
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
    total_loss = 0.0
    total_entropy = 0.0
    total_batches = 0
    
    envs_per_batch = config['mini_batch_size'] // T
    
    for epoch in range(config['ppo_epochs']):
        env_perm = torch.randperm(E)
        
        for batch_start in range(0, E, envs_per_batch):
            batch_end = min(batch_start + envs_per_batch, E)
            env_idx = env_perm[batch_start:batch_end]
            M = len(env_idx)
            
            # Load batch to GPU
            obs_seq = obs_buf[:, env_idx].to(device)
            inst_seq = inst_buf[:, env_idx].to(device)
            act_seq = act_buf[:, env_idx].to(device)
            logp_old_seq = logp_buf[:, env_idx].to(device)
            adv_seq = adv_buf[:, env_idx].to(device)
            ret_seq = ret_buf[:, env_idx].to(device)
            val_old_seq = val_buf[:, env_idx].to(device)
            done_seq = done_buf[:, env_idx].to(device)
            prev_rew_seq = prev_rew_buf[:, env_idx].to(device)
            prev_done_seq = prev_done_buf[:, env_idx].to(device)
            
            # Normalize advantages
            adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)
            
            brain.reset(M, device=device)
            
            policy_losses = []
            value_losses = []
            entropies = []
            
            for t in range(T):
                # Reset hidden state for done envs
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
                
                # Language fusion
                if brain.use_language:
                    lang_h = brain.lang_encoder(inst_t)
                    z_combined = torch.cat([z_t, lang_h], dim=-1)
                    z_t = brain.lang_projection(z_combined)
                
                # Retrieval and correction
                retrieved = brain.hippocampus.retrieve(z_t)
                correction, _ = brain.cerebellum.forward(z_t, obs_t)
                
                selection, da, _, new_logp, value, entropy = brain.bg.step(
                    z_t,
                    torch.zeros(M, 1, device=device),
                    None,
                    torch.zeros(M, 1, dtype=torch.bool, device=device),
                    brain.state.bg_state['prev_value'],
                    memory_context=retrieved,
                    cerebellum_correction=correction,
                    action_to_eval=act_seq[t]
                )
                
                new_val = value.squeeze(-1).float()
                
                # Update brain state
                with torch.no_grad():
                    brain.state.z = z_t.detach()
                    brain.state.cortex_state = tuple(s.detach() for s in new_cortex_state)
                    brain.state.bg_state['prev_value'] = new_val.unsqueeze(-1).detach()
                    brain._prev_pred = pred_t.detach()
                
                # PPO losses
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
                
                total_loss += loss.item()
                total_entropy += entropy_loss.item()
                total_batches += 1
    
    # Restore collector state
    brain.state = collector_state[0]
    brain._prev_selection = collector_state[1]
    brain._prev_mods = collector_state[2]
    brain._prev_pred = collector_state[3]
    
    # Compute explained variance
    with torch.no_grad():
        ret_flat = ret_buf.flatten()
        val_flat = val_buf.flatten()
        ev = 1 - (ret_flat - val_flat).var() / (ret_flat.var() + 1e-8)
    
    return {
        'loss': total_loss / max(total_batches, 1),
        'entropy': total_entropy / max(total_batches, 1),
        'explained_variance': ev.item(),
        'mean_reward': rew_buf.mean().item(),
    }


def train_full(args):
    """Train on all 9 instructions."""
    print("=" * 60)
    print("Phase 1: Full Training (All 9 Instructions)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    config = create_brain_config(args)
    
    # All 9 combos (3 colors x 3 shapes)
    all_combos = [(c, s) for c in range(3) for s in range(3)]
    
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
    
    # Load existing if available
    checkpoint_path = f"checkpoints/brain_language_seed{args.seed}.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    if os.path.exists(checkpoint_path) and not args.fresh:
        try:
            state_dict = torch.load(checkpoint_path, map_location=device)
            brain.load_state_dict(state_dict, strict=False)
            print(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Starting fresh: {e}")
    
    # Optimizer (train all parameters)
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'], eps=1e-5)
    
    best_acc = 0.0
    
    print(f"\nTraining for {args.total_updates} updates...")
    print(f"Envs: {config['num_envs']}, Steps/update: {config['num_steps']}")
    
    for update in range(args.total_updates):
        start_time = time.time()
        
        metrics = train_step(
            brain, env, optimizer, config, device,
            train_combos=all_combos,
            language='english'
        )
        
        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)
        
        if (update + 1) % args.eval_every == 0:
            # Evaluate on all combos
            results = evaluate_language(
                brain, eval_env, device, all_combos,
                episodes_per_combo=args.eval_episodes,
                language='english'
            )
            
            acc = results['overall']
            print(f"[Update {update+1:4d}] Loss: {metrics['loss']:.3f} | "
                  f"EV: {metrics['explained_variance']:.2f} | "
                  f"Acc: {acc*100:.1f}% | FPS: {fps:.0f}")
            
            # Per-combo breakdown every 50 updates
            if (update + 1) % 50 == 0:
                print("  Per-combo accuracy:")
                for combo_name, combo_acc in sorted(results.items()):
                    if combo_name != 'overall':
                        print(f"    {combo_name}: {combo_acc*100:.1f}%")
            
            if acc > best_acc:
                best_acc = acc
                torch.save(brain.state_dict(), checkpoint_path)
                print(f"  -> New best: {best_acc*100:.1f}%")
            
            # Early stopping if goal reached
            if acc >= 0.90:
                print(f"\n✅ SUCCESS: Achieved {acc*100:.1f}% accuracy (>= 90%)")
                break
    
    print(f"\nFinal best accuracy: {best_acc*100:.1f}%")
    return brain, best_acc


def train_compositional(args):
    """Train on 6 combos, test on 3 held-out."""
    print("=" * 60)
    print("Phase 2: Compositional Generalization Test")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = create_brain_config(args)
    
    train_combos, test_combos = get_train_test_split()
    print(f"Train combos: {train_combos}")
    print(f"Test combos (held-out): {test_combos}")
    
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
    
    brain = DigitalBrain(config).to(device)
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'], eps=1e-5)
    
    best_train_acc = 0.0
    best_test_acc = 0.0
    
    for update in range(args.total_updates):
        start_time = time.time()
        
        metrics = train_step(
            brain, env, optimizer, config, device,
            train_combos=train_combos,
            language='english'
        )
        
        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)
        
        if (update + 1) % args.eval_every == 0:
            train_results = evaluate_language(
                brain, eval_env, device, train_combos,
                episodes_per_combo=args.eval_episodes,
                language='english'
            )
            
            test_results = evaluate_language(
                brain, eval_env, device, test_combos,
                episodes_per_combo=args.eval_episodes,
                language='english'
            )
            
            train_acc = train_results['overall']
            test_acc = test_results['overall']
            
            print(f"[Update {update+1:4d}] Train: {train_acc*100:.1f}% | "
                  f"Test (novel): {test_acc*100:.1f}% | "
                  f"Random baseline: 33% | FPS: {fps:.0f}")
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                checkpoint_path = f"checkpoints/brain_compositional_seed{args.seed}.pth"
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(brain.state_dict(), checkpoint_path)
            
            # Check success criteria
            if train_acc >= 0.90 and test_acc >= 0.50:
                print(f"\n✅ SUCCESS: Train {train_acc*100:.1f}% >= 90%, Test {test_acc*100:.1f}% >= 50%")
                print("   Compositional understanding demonstrated!")
                break
    
    print(f"\nFinal Results:")
    print(f"  Train accuracy: {best_train_acc*100:.1f}%")
    print(f"  Test accuracy (novel combinations): {best_test_acc*100:.1f}%")
    print(f"  Random baseline: 33%")
    print(f"  Above random: {best_test_acc > 0.33}")
    
    return brain, best_train_acc, best_test_acc


def main():
    parser = argparse.ArgumentParser(description="Language Grounding Training")
    parser.add_argument('--mode', type=str, default='full', 
                       choices=['full', 'compositional'],
                       help='Training mode: full (all 9) or compositional (6 train, 3 test)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--d_z', type=int, default=256)
    parser.add_argument('--d_lang_embed', type=int, default=64)
    parser.add_argument('--d_lang_hidden', type=int, default=128)
    parser.add_argument('--num_envs', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=64)
    parser.add_argument('--mini_batch_size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--total_updates', type=int, default=500)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--eval_episodes', type=int, default=20)
    parser.add_argument('--use_hippocampus', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_plasticity', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_memory_policy', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_cerebellum', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--fresh', action='store_true', help='Start from scratch')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    if args.mode == 'full':
        train_full(args)
    elif args.mode == 'compositional':
        train_compositional(args)


if __name__ == "__main__":
    main()
