import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import sys
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from digital_brain.datatypes import Obs, BrainState, ModSignals
from digital_brain.brain import DigitalBrain

# Enable TF32 for faster matmuls on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def log_stability_metrics(brain, update, avg_kl, avg_entropy, ev, fps, stage_str, avg_da_std, sr=None):
    with torch.no_grad():
        # Cortex metrics
        if hasattr(brain.cortex.microcircuit, 'W_ee'):
            w_ee = brain.cortex.microcircuit.W_ee
        else:
            w_ee = brain.cortex.microcircuit.W_ee_values
            
        w_ee_max = w_ee.abs().max().item()
        w_ee_mean = w_ee.abs().mean().item()
        
        # State metrics (using current brain state)
        e_act, i_act, trace_ee = brain.state.cortex_state
        trace_max = trace_ee.abs().max().item()
        
        # Policy head weight norms
        policy_w_max = brain.bg.policy_head.weight.abs().max().item()
        
        has_nan = torch.isnan(w_ee).any().item()
        
    sr_str = f" | SR {sr:.3f}" if sr is not None else ""
    print(f"[{stage_str}] Upd {update:4d}{sr_str} | EV {ev:.2f} | KL {avg_kl:.4f} | DA_std {avg_da_std:.2f} | W_max {w_ee_max:.2f} | Tr_max {trace_max:.2f} | PolW {policy_w_max:.2f} | FPS {fps:.0f}")
    
    if w_ee_max > 100.0 or has_nan:
        print(f"CRITICAL: Stability issue detected! W_max={w_ee_max:.2f}, NaN={has_nan}")
        return True # Signal instability
    return False

import argparse

def train_vectorized():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sparse_cortex', type=bool, default=False)
    parser.add_argument('--d_z', type=int, default=512)
    parser.add_argument('--total_steps', type=int, default=2000000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=10)
    parser.add_argument('--use_hippocampus', type=bool, default=True)
    parser.add_argument('--use_plasticity', type=bool, default=True)
    parser.add_argument('--use_memory_policy', type=bool, default=True)
    parser.add_argument('--use_cerebellum', type=bool, default=True)
    parser.add_argument('--locality_radius', type=int, default=None)
    args, unknown = parser.parse_known_args()

    config = {
        'd_obs': 9, 'd_z': args.d_z, 'd_sel': 64, 'd_act': 4, 
        'lr': 3.5e-4,
        'total_steps': args.total_steps,
        'num_envs': 4096,
        'num_steps': 128,
        'ppo_epochs': 4,           # Reduced from 8 (large batch doesn't need many epochs)
        'mini_batch_size': 65536,  # Increased from 32768 (= 512 envs × 128 steps)
        'eps_clip': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.005,
        'value_coef': 0.5,
        'vf_clip': 0.2,
        'target_kl': 0.015,  
        'seed': args.seed,
        'eval_every': args.eval_every,
        'eval_episodes': 64,       # Reduced from 200
        'num_eval_envs': 64,  
        'selection_penalty': 0.001,
        'sparse_cortex': args.sparse_cortex,    # Toggle for sparse W_ee (O(N) scaling)
        'sparsity': 0.01,
        'use_hippocampus': args.use_hippocampus,
        'use_plasticity': args.use_plasticity,
        'use_memory_policy': args.use_memory_policy,
        'use_cerebellum': args.use_cerebellum,
        'locality_radius': args.locality_radius,
        'grid_size': args.grid_size
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Parallel Envs: {config['num_envs']}")

    envs = TorchVectorPOMDP(num_envs=config['num_envs'], size=config['grid_size'], device=device, seed=config['seed'])
    eval_env = TorchVectorPOMDP(num_envs=config['num_eval_envs'], size=config['grid_size'], device=device, seed=config['seed'] + 1000)

    brain = DigitalBrain(config).to(device)
    
    if os.path.exists("brain_vectorized_best.pth"):
        try:
            state_dict = torch.load("brain_vectorized_best.pth", map_location=device)
            brain.load_state_dict(state_dict, strict=False)
            print("Loaded existing vectorized best")
        except Exception as e:
            print(f"Starting fresh: {e}")

    # Stage-B: Train BG + Thalamus (Cortex still frozen for stability)
    # Stage-A (only BG) hit ceiling - need more capacity
    stage_a_freeze = False  # Changed to Stage-B
    ev_threshold = 0.2
    
    if stage_a_freeze:
        # Stage-A: Only BG policy/value
        for p in brain.parameters():
            p.requires_grad = False
        for p in brain.bg.policy_head.parameters():
            p.requires_grad = True
        for p in brain.bg.value_head.parameters():
            p.requires_grad = True
        trainable_params = list(brain.bg.policy_head.parameters()) + list(brain.bg.value_head.parameters())
        print(f"Stage-A: Frozen all except BG policy/value ({sum(p.numel() for p in trainable_params)} params)")
    else:
        # Stage-B: BG + Thalamus + Cerebellum + Cortex Heads (Cortex recurrent frozen)
        for p in brain.parameters():
            p.requires_grad = False
        for p in brain.bg.parameters():
            p.requires_grad = True
        for p in brain.thalamus.parameters():
            p.requires_grad = True
        for p in brain.cerebellum.parameters():
            p.requires_grad = True
        for p in brain.cortex.pred_head.parameters():
            p.requires_grad = True
        brain.cortex.microcircuit.W_in.requires_grad = True # Unfreeze input weights for WM
            
        trainable_params = (list(brain.bg.parameters()) + 
                            list(brain.thalamus.parameters()) + 
                            list(brain.cerebellum.parameters()) + 
                            list(brain.cortex.pred_head.parameters()) +
                            [brain.cortex.microcircuit.W_in])
        print(f"Stage-B: Training BG + Thalamus + Cerebellum + PredHead ({sum(p.numel() for p in trainable_params)} params)")

    optimizer = optim.Adam(trainable_params, lr=config['lr'], eps=1e-5)
    best_sr = 0.0
    best_ev = 0.0

    obs_np = envs.reset()
    brain.reset(config['num_envs'], device=device)
    
    prev_reward = torch.zeros(config['num_envs'], 1, device=device)
    prev_done = torch.zeros(config['num_envs'], 1, dtype=torch.bool, device=device)

    num_updates = config['total_steps'] // (config['num_envs'] * config['num_steps'])
    
    print(f"Starting Vectorized PPO Training ({config['num_envs']} envs, {config['num_steps']} steps/upd)...")

    for update in range(num_updates):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start_time = time.time()
        
        # Minimal PPO buffers (CPU to save GPU memory, will transfer per minibatch)
        # No more per-step state buffers - saves ~3 GB
        T, E = config['num_steps'], config['num_envs']
        obs_buf = torch.zeros((T, E, config['d_obs']), dtype=torch.float32)
        act_buf = torch.zeros((T, E), dtype=torch.long)
        logp_buf = torch.zeros((T, E), dtype=torch.float32)
        val_buf = torch.zeros((T, E), dtype=torch.float32)
        rew_buf = torch.zeros((T, E), dtype=torch.float32)
        done_buf = torch.zeros((T, E), dtype=torch.bool)
        prev_rew_buf = torch.zeros((T, E), dtype=torch.float32)  # For brain input
        prev_done_buf = torch.zeros((T, E), dtype=torch.bool)

        # 1) Collect Experience (with inference_mode + autocast for speed)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type=='cuda')):
            for t in range(T):
                obs_t = obs_np # Already a tensor on device from TorchVectorPOMDP
                
                # Store inputs for this step
                obs_buf[t] = obs_t.cpu()
                prev_rew_buf[t] = prev_reward.squeeze(-1).cpu()
                prev_done_buf[t] = prev_done.squeeze(-1).cpu()
                
                # Rollout with online learning enabled (plasticity + hippocampal encoding)
                action, log_prob, value, _, _, _ = brain.step(Obs(x=obs_t), prev_reward, prev_done, learn=True)
                
                obs_np, reward, done, _ = envs.step(action)
                
                act_buf[t] = action.cpu()
                logp_buf[t] = log_prob.float().cpu()
                val_buf[t] = value.squeeze(-1).float().cpu()
                rew_buf[t] = reward.cpu()
                done_buf[t] = done.cpu()
                
                prev_reward = reward.float().unsqueeze(1)
                prev_done = done.unsqueeze(1)

        # 2) Compute GAE (on CPU, then move advantages/returns)
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type=='cuda')):
            obs_last = obs_np # Already a tensor on device
            gated_last = brain.thalamus.gate(obs_last, brain._prev_selection, brain._prev_mods)
            z_last, _, _ = brain.cortex.forward(gated_last, brain.state.cortex_state, update_trace=False)
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

        # 3) Recurrent PPO Update - minibatch over envs, unroll full sequences
        # Save collector state to restore after PPO
        collector_state = (
            brain.state, 
            brain._prev_selection.clone(), 
            ModSignals(DA=brain._prev_mods.DA.clone(), NE=brain._prev_mods.NE.clone(),
                       ACh=brain._prev_mods.ACh.clone(), HT5=brain._prev_mods.HT5.clone()),
            brain._prev_pred.clone()
        )

        # 3.1) World Model update (Optional but recommended for cortex prediction)
        # Predict next obs from current state
        # (This is a simplified version, ideally part of the main PPO update)

        # Diagnostics accumulators
        total_kl, total_clipfrac, total_entropy, total_batches = 0.0, 0.0, 0.0, 0
        total_logit_scale, total_prob_max = 0.0, 0.0
        total_da_std = 0.0
        early_stop_epoch = config['ppo_epochs']
        
        # Recurrent PPO: minibatch_size = envs_per_batch * T
        envs_per_batch = config['mini_batch_size'] // T  # 65536 / 128 = 512 envs
        
        for epoch in range(config['ppo_epochs']):
            epoch_kl = 0.0
            epoch_batches = 0
            env_perm = torch.randperm(E)  # Shuffle env indices
            
            for batch_start in range(0, E, envs_per_batch):
                batch_end = min(batch_start + envs_per_batch, E)
                env_idx = env_perm[batch_start:batch_end]
                M = len(env_idx)  # Actual batch size (may be smaller at end)
                
                # Load sequences for selected envs -> GPU [T, M, ...]
                obs_seq = obs_buf[:, env_idx].to(device)
                act_seq = act_buf[:, env_idx].to(device)
                logp_old_seq = logp_buf[:, env_idx].to(device)
                adv_seq = adv_buf[:, env_idx].to(device)
                ret_seq = ret_buf[:, env_idx].to(device)
                val_old_seq = val_buf[:, env_idx].to(device)
                done_seq = done_buf[:, env_idx].to(device)
                prev_rew_seq = prev_rew_buf[:, env_idx].to(device)
                prev_done_seq = prev_done_buf[:, env_idx].to(device)
                
                # Normalize advantages ONCE per minibatch (across all T*M samples)
                adv_seq = (adv_seq - adv_seq.mean()) / (adv_seq.std() + 1e-8)
                
                # Initialize fresh hidden state for this minibatch
                brain.reset(M, device=device)
                
                # Accumulate losses over sequence
                policy_losses = []
                value_losses = []
                entropies = []
                kls = []
                clipfracs = []
                logit_scales = []
                prob_maxes = []
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(device.type=='cuda')):
                    for t in range(T):
                        # Reset hidden state for envs that were done at t-1
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
                                    mask_1d = (~done_mask.squeeze(-1)).float()
                                    brain._prev_mods.DA = brain._prev_mods.DA * mask_1d
                                    brain._prev_mods.NE = brain._prev_mods.NE * mask_1d
                                    brain._prev_mods.ACh = brain._prev_mods.ACh * mask_1d + (1 - mask_1d) * 0.5
                                    brain._prev_mods.HT5 = brain._prev_mods.HT5 * mask_1d + (1 - mask_1d) * 0.5
                        
                        # Forward pass (no trace update during PPO)
                        obs_t = obs_seq[t]
                        prev_rew_t = prev_rew_seq[t].unsqueeze(-1)
                        prev_done_t = prev_done_seq[t].unsqueeze(-1)
                        
                        gated_x = brain.thalamus.gate(obs_t, brain._prev_selection, brain._prev_mods)
                        z_t, pred_t, new_cortex_state = brain.cortex.forward(gated_x, brain.state.cortex_state, update_trace=False)
                        
                        # Homeostatic Regularization (Priority 2 Stability)
                        # Penalize neurons that are saturated (0 or near clamp limit 50)
                        # Target mean activity ~ 1.0
                        firing_reg = torch.mean(z_t**2) * 0.001
                        policy_losses.append(firing_reg)
                        
                        # Priority 5 & 7: Retrieval and Correction for PPO update consistency
                        retrieved = brain.hippocampus.retrieve(z_t)
                        correction, _ = brain.cerebellum.forward(z_t, obs_t)
                        
                        # Priority 4: Predictive World Model Loss
                        if t < T - 1:
                            wm_loss_t = torch.mean((pred_t - obs_seq[t+1]) ** 2)
                            policy_losses.append(wm_loss_t * 0.1) 
                        
                        selection, da, _, new_logp, value, entropy = brain.bg.step(
                            z_t, 
                            torch.zeros(M, 1, device=device), # DA signal computed internally from value
                            None, 
                            torch.zeros(M, 1, dtype=torch.bool, device=device), # Done masking handled manually above
                            brain.state.bg_state['prev_value'],
                            memory_context=retrieved,
                            cerebellum_correction=correction,
                            action_to_eval=act_seq[t]
                        )
                        
                        # Collect DA stats for stability monitoring
                        total_da_std += da.std().item()
                        
                        new_val = value.squeeze(-1).float()
                        
                        # Update brain state for next step (detached)
                        with torch.no_grad():
                            brain.state.z = z_t.detach()
                            brain.state.cortex_state = tuple(s.detach() for s in new_cortex_state)
                            brain.state.bg_state['prev_value'] = new_val.unsqueeze(-1).detach()
                            # Update selection/mods (simplified - use defaults since not learning)
                            brain._prev_pred = pred_t.detach()
                        
                        # PPO losses for this timestep
                        log_ratio = new_logp - logp_old_seq[t]
                        ratio = torch.exp(log_ratio)
                        
                        # Use pre-normalized advantages (normalized once per minibatch above)
                        mb_adv = adv_seq[t]
                        
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * mb_adv
                        policy_loss_t = -torch.min(surr1, surr2).mean()
                        
                        # Value clipping
                        old_val = val_old_seq[t]
                        v_clipped = old_val + (new_val - old_val).clamp(-config['vf_clip'], config['vf_clip'])
                        v_loss1 = (new_val - ret_seq[t])**2
                        v_loss2 = (v_clipped - ret_seq[t])**2
                        val_loss_t = 0.5 * torch.max(v_loss1, v_loss2).mean()
                        
                        policy_losses.append(policy_loss_t)
                        value_losses.append(val_loss_t)
                        entropies.append(entropy.mean())
                        
                        # Diagnostics (no grad)
                        with torch.no_grad():
                            kls.append((-log_ratio).mean().item())
                            clipfracs.append(((ratio - 1.0).abs() > config['eps_clip']).float().mean().item())
                            # logit_scales.append(logits.abs().mean().item()) # Removed as logits are internal to BG.step
                            prob_maxes.append(torch.exp(new_logp).max(dim=-1).values.mean().item())
                
                # Aggregate losses over sequence
                policy_loss = torch.stack(policy_losses).mean()
                val_loss = torch.stack(value_losses).mean()
                entropy_loss = torch.stack(entropies).mean()
                
                loss = policy_loss + config['value_coef'] * val_loss - config['entropy_coef'] * entropy_loss
                
                if torch.isnan(loss):
                    print(f"NaN loss detected at update {update}, epoch {epoch}! Skipping batch.")
                    optimizer.zero_grad(set_to_none=True)
                    continue

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 0.5)
                optimizer.step()
                
                # Accumulate diagnostics
                batch_kl = np.mean(kls)
                total_kl += batch_kl
                total_clipfrac += np.mean(clipfracs)
                total_entropy += entropy_loss.item()
                total_logit_scale += np.mean(logit_scales)
                total_prob_max += np.mean(prob_maxes)
                total_batches += 1
                epoch_kl += batch_kl
                epoch_batches += 1
            
            # KL Early-Stop check after each epoch
            epoch_kl_avg = epoch_kl / max(epoch_batches, 1)
            if epoch_kl_avg > 1.5 * config['target_kl']:
                early_stop_epoch = epoch + 1
                break

        # Restore collector state
        brain.state = collector_state[0]
        brain._prev_selection = collector_state[1]
        brain._prev_mods = collector_state[2]
        brain._prev_pred = collector_state[3]
        
        # Compute diagnostics averages
        avg_kl = total_kl / max(total_batches, 1)
        avg_clipfrac = total_clipfrac / max(total_batches, 1)
        avg_entropy = total_entropy / max(total_batches, 1)
        avg_logit = total_logit_scale / max(total_batches, 1)
        avg_pmax = total_prob_max / max(total_batches, 1)
        avg_da_std = total_da_std / max(total_batches, 1)
        
        # Explained variance: how well value predicts returns
        with torch.no_grad():
            ret_flat = ret_buf.flatten()
            val_flat = val_buf.flatten()
            ev = 1 - (ret_flat - val_flat).var() / (ret_flat.var() + 1e-8)
            ev = ev.item()

        if device.type == 'cuda':
            torch.cuda.synchronize()
        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)

        if (update + 1) % config['eval_every'] == 0:
            # 1) Log stability metrics using CURRENT training state before eval/reset
            stage_str = "A" if stage_a_freeze else "B"
            is_unstable = log_stability_metrics(brain, update+1, avg_kl, avg_entropy, ev, fps, stage_str, avg_da_std)
            
            if is_unstable:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5
                print(f"  ** Emergency Brake: Reducing LR to {optimizer.param_groups[0]['lr']:.2e} **")

            # 2) Save training state for evaluation
            train_state = (brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred)
            sr = eval_vectorized(brain, eval_env, device, config['eval_episodes'], 150)
            # Restore training state after eval
            brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred = train_state
            
            print(f"  -> Eval SR: {sr:.3f}")
            
            # Track best SR
            if sr > best_sr:
                best_sr = sr
                torch.save(brain.state_dict(), "brain_vectorized_best.pth")
                print(f"  -> New Best SR: {best_sr:.3f}")

@torch.no_grad()
def eval_vectorized(brain, env, device, episodes, max_steps):
    """Vectorized evaluation: runs num_eval_envs in parallel."""
    num_envs = env.num_envs
    success = 0
    completed = 0
    
    while completed < episodes:
        obs_t = env.reset()
        brain.reset(num_envs, device=device)
        prev_reward = torch.zeros(num_envs, 1, device=device)
        prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        
        ep_returns = torch.zeros(num_envs, device=device)
        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            obs = Obs(x=obs_t)
            action, _, _, _, _, _ = brain.act(obs, prev_reward, prev_done)
            obs_t, reward, done, _ = env.step(action)
            
            # Only accumulate for not-yet-done episodes
            ep_returns += reward * (~ep_done).float()
            ep_done = ep_done | done
            
            prev_reward = reward.float().unsqueeze(1)
            prev_done = done.unsqueeze(1)
            
            if ep_done.all():
                break
        
        # Count successes
        success += (ep_returns > 5.0).sum().item()
        completed += num_envs
    
    return success / completed

if __name__ == "__main__":
    train_vectorized()
