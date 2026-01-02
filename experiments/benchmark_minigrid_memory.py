import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import sys
import argparse

# Ensure we can import from root
sys.path.append(os.getcwd())

from digital_brain.envs.torch_minigrid import TorchMiniGridMemory
from digital_brain.datatypes import Obs, BrainState, ModSignals
from digital_brain.brain import DigitalBrain

# Enable TF32
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def log_stability_metrics(brain, update, avg_kl, avg_entropy, ev, fps, stage_str, avg_da_std, sr=None):
    with torch.no_grad():
        # Cortex metrics
        if hasattr(brain.cortex.microcircuit, 'W_ee_values'):
            w_ee = brain.cortex.microcircuit.W_ee_values
        elif hasattr(brain.cortex.microcircuit, 'W_ee'):
            w_ee = brain.cortex.microcircuit.W_ee
        else: # Hierarchical
             w_ee = brain.cortex.microcircuit.layers[0].W_ee_values if hasattr(brain.cortex.microcircuit.layers[0], 'W_ee_values') else brain.cortex.microcircuit.layers[0].W_ee

        w_ee_max = w_ee.abs().max().item()
        
        # State metrics (handle hierarchical list)
        if isinstance(brain.state.cortex_state, list):
             trace_max = brain.state.cortex_state[0][2].abs().max().item()
        else:
             e_act, i_act, trace_ee = brain.state.cortex_state
             trace_max = trace_ee.abs().max().item()
        
        policy_w_max = brain.bg.policy_head.weight.abs().max().item()
        has_nan = torch.isnan(w_ee).any().item()
        
    sr_str = f" | SR {sr:.3f}" if sr is not None else ""
    print(f"[{stage_str}] Upd {update:4d}{sr_str} | EV {ev:.2f} | KL {avg_kl:.4f} | DA_std {avg_da_std:.2f} | W_max {w_ee_max:.2f} | Tr_max {trace_max:.2f} | PolW {policy_w_max:.2f} | FPS {fps:.0f}")
    
    if w_ee_max > 100.0 or has_nan:
        print(f"CRITICAL: Stability issue detected! W_max={w_ee_max:.2f}, NaN={has_nan}")
        return True 
    return False

def flatten_obs(obs):
    # obs: (N, 7, 7, 3) -> (N, 147)
    return obs.reshape(obs.shape[0], -1)

def clone_t(t):
    if isinstance(t, torch.Tensor): return t.clone()
    if isinstance(t, list): return [clone_t(x) for x in t]
    if isinstance(t, tuple): return tuple(clone_t(x) for x in t)
    return t

def clone_brain_state(brain):
    bs = brain.state
    new_z = bs.z.clone()
    new_cortex = clone_t(bs.cortex_state)
    new_bg = {k: v.clone() for k, v in bs.bg_state.items()}
    new_bs = BrainState(z=new_z, cortex_state=new_cortex, bg_state=new_bg, hip_state=None, cerebellum_state=None)
    new_sel = brain._prev_selection.clone()
    new_pred = brain._prev_pred.clone()
    ms = brain._prev_mods
    new_mods = ModSignals(DA=ms.DA.clone(), NE=ms.NE.clone(), ACh=ms.ACh.clone(), HT5=ms.HT5.clone())
    return (new_bs, new_sel, new_mods, new_pred)

def restore_brain_state(brain, saved):
    brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred = saved

def train_minigrid_memory():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d_z', type=int, default=512)
    parser.add_argument('--total_steps', type=int, default=20000000)
    parser.add_argument('--corridor', type=int, default=7)
    parser.add_argument('--hierarchical', action='store_true')
    parser.add_argument('--cpu', action='store_true', default=False) # Changed default to False (GPU)
    args = parser.parse_args()

    # MiniGrid Config
    d_obs = 7 * 7 * 3 # 147
    d_act = 3 # Left, Right, Forward
    
    config = {
        'd_obs': d_obs, 
        'd_z': args.d_z, 
        'd_sel': 64, 
        'd_act': d_act, 
        'lr': 3e-4, 
        'total_steps': args.total_steps,
        'num_envs': 512, # Increased from 128
        'num_steps': 128,
        'ppo_epochs': 4,
        'mini_batch_size': 16384, # Increased from 4096
        'eps_clip': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.05, # Higher entropy for exploration in memory task
        'value_coef': 0.5,
        'vf_clip': 0.2,
        'target_kl': 0.02,  
        'seed': 42,
        'eval_every': 20,
        'eval_episodes': 32,
        'num_eval_envs': 32,
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': True,
        'use_cerebellum': True,
    }

    if args.hierarchical:
        config['layer_sizes'] = [256, 512, 256]

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    print(f"Using device: {device} | Corridor: {args.corridor}")
    
    # Environments
    envs = TorchMiniGridMemory(num_envs=config['num_envs'], corridor_length=args.corridor, device=device, seed=config['seed'])
    eval_env = TorchMiniGridMemory(num_envs=config['num_eval_envs'], corridor_length=args.corridor, device=device, seed=config['seed'] + 1000)

    brain = DigitalBrain(config).to(device)
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'], eps=1e-5)
    
    # Reset
    obs_raw = envs.reset()
    obs_flat = flatten_obs(obs_raw)
    brain.reset(config['num_envs'], device=device)
    
    prev_reward = torch.zeros(config['num_envs'], 1, device=device)
    prev_done = torch.zeros(config['num_envs'], 1, dtype=torch.bool, device=device)
    
    num_updates = config['total_steps'] // (config['num_envs'] * config['num_steps'])
    print(f"Starting MiniGrid-Memory Training...")

    for update in range(num_updates):
        start_time = time.time()
        T, E = config['num_steps'], config['num_envs']
        
        # Buffers
        obs_buf = torch.zeros((T, E, config['d_obs']), device=device)
        act_buf = torch.zeros((T, E), dtype=torch.long, device=device)
        logp_buf = torch.zeros((T, E), device=device)
        val_buf = torch.zeros((T, E), device=device)
        rew_buf = torch.zeros((T, E), device=device)
        done_buf = torch.zeros((T, E), dtype=torch.bool, device=device)
        prev_rew_buf = torch.zeros((T, E), device=device)
        prev_done_buf = torch.zeros((T, E), dtype=torch.bool, device=device)
        
        # 1. Collect
        with torch.inference_mode():
            for t in range(T):
                obs_buf[t] = obs_flat
                prev_rew_buf[t] = prev_reward.squeeze(-1)
                prev_done_buf[t] = prev_done.squeeze(-1)
                
                action, log_prob, value, _, _, _ = brain.step(Obs(x=obs_flat), prev_reward, prev_done, learn=True)
                
                obs_next_raw, reward, done, _ = envs.step(action)
                obs_next_flat = flatten_obs(obs_next_raw)
                
                act_buf[t] = action
                logp_buf[t] = log_prob
                val_buf[t] = value.squeeze(-1)
                rew_buf[t] = reward
                done_buf[t] = done
                
                prev_reward = reward.float().unsqueeze(1)
                prev_done = done.unsqueeze(1)
                obs_flat = obs_next_flat

        # 2. GAE
        next_value = 0.0 
        adv_buf = torch.zeros((T, E), device=device)
        last_gae = 0
        for t in reversed(range(T)):
            next_non_terminal = (~done_buf[t]).float()
            next_val = next_value if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + config['gamma'] * next_val * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + config['gamma'] * config['gae_lambda'] * next_non_terminal * last_gae
        ret_buf = adv_buf + val_buf
        
        # 3. Recurrent PPO Update
        collector_state = clone_brain_state(brain)
        envs_per_batch = config['mini_batch_size'] // T
        
        kl_list, entropy_list = [], []
        
        for epoch in range(config['ppo_epochs']):
            perm = torch.randperm(E, device=device)
            for start in range(0, E, envs_per_batch):
                end = min(start + envs_per_batch, E)
                env_idx = perm[start:end]
                M = len(env_idx)
                
                mb_obs = obs_buf[:, env_idx]
                mb_act = act_buf[:, env_idx]
                mb_logp = logp_buf[:, env_idx]
                mb_adv = adv_buf[:, env_idx]
                mb_ret = ret_buf[:, env_idx]
                mb_prev_rew = prev_rew_buf[:, env_idx]
                mb_prev_done = prev_done_buf[:, env_idx]
                
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                brain.reset(M, device=device)
                policy_losses, value_losses, mb_entropies = [], [], []
                
                for t in range(T):
                    if t > 0:
                        done_mask = mb_prev_done[t].view(M, 1)
                        if done_mask.any():
                            with torch.no_grad():
                                brain.state.z = brain.state.z * (~done_mask).float()
                                if isinstance(brain.state.cortex_state, list):
                                     brain.state.cortex_state = [
                                         (e*(~done_mask).float(), i*(~done_mask).float(), tr) 
                                         for e, i, tr in brain.state.cortex_state
                                     ]
                                else:
                                     e, i, tr = brain.state.cortex_state
                                     brain.state.cortex_state = (e*(~done_mask).float(), i*(~done_mask).float(), tr)
                                brain.state.bg_state['prev_value'] = brain.state.bg_state['prev_value'] * (~done_mask).float()
                                brain._prev_selection = brain._prev_selection * (~done_mask).float()
                                brain._prev_pred = brain._prev_pred * (~done_mask).float()

                    obs_t = mb_obs[t]
                    gated_x = brain.thalamus.gate(obs_t, brain._prev_selection, brain._prev_mods)
                    z_t, pred_t, new_cortex_state = brain.cortex.forward(gated_x, brain.state.cortex_state, update_trace=False)
                    retrieved = brain.hippocampus.retrieve(z_t)
                    correction, _ = brain.cerebellum.forward(z_t, obs_t)
                    
                    selection, da, _, new_logp, value, entropy = brain.bg.step(
                        z_t, torch.zeros(M, 1, device=device), None, 
                        torch.zeros(M, 1, dtype=torch.bool, device=device), 
                        brain.state.bg_state['prev_value'],
                        memory_context=retrieved, cerebellum_correction=correction,
                        action_to_eval=mb_act[t]
                    )
                    
                    with torch.no_grad():
                        brain.state.z = z_t.detach()
                        if isinstance(new_cortex_state, list):
                            brain.state.cortex_state = [tuple(s.detach() for s in ls) for ls in new_cortex_state]
                        else:
                            brain.state.cortex_state = tuple(s.detach() for s in new_cortex_state)
                        brain.state.bg_state['prev_value'] = value.detach()
                        brain._prev_selection = selection.detach()
                        brain._prev_pred = pred_t.detach()
                    
                    log_ratio = new_logp - mb_logp[t]
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * mb_adv[t]
                    surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * mb_adv[t]
                    
                    policy_losses.append(-torch.min(surr1, surr2).mean())
                    value_losses.append(0.5 * ((value.squeeze(-1) - mb_ret[t])**2).mean())
                    mb_entropies.append(entropy.mean())
                    kl_list.append((-log_ratio).mean().item())

                loss = torch.stack(policy_losses).mean() + config['value_coef'] * torch.stack(value_losses).mean() - config['entropy_coef'] * torch.stack(mb_entropies).mean()
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                optimizer.step()
                entropy_list.append(torch.stack(mb_entropies).mean().item())

        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)
        
        if update % 5 == 0:
            log_stability_metrics(brain, update, np.mean(kl_list), np.mean(entropy_list), 0.0, fps, "TRAIN", 0.0)

        if (update + 1) % config['eval_every'] == 0:
            sr = eval_minigrid(brain, eval_env, device, config['eval_episodes'])
            print(f"  -> Eval SR: {sr:.3f}")

        restore_brain_state(brain, collector_state)

def eval_minigrid(brain, env, device, episodes):
    completed, success = 0, 0
    num_envs = env.num_envs
    while completed < episodes:
        obs_raw = env.reset()
        obs_flat = flatten_obs(obs_raw)
        brain.reset(num_envs, device=device)
        prev_rew = torch.zeros(num_envs, 1, device=device)
        prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        ep_ret = torch.zeros(num_envs, device=device)
        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for _ in range(256): # More steps for memory task
            action, _, _, _, _, _ = brain.act(Obs(x=obs_flat), prev_rew, prev_done)
            obs_next, reward, done, _ = env.step(action)
            obs_flat = flatten_obs(obs_next)
            ep_ret += reward * (~ep_done).float()
            ep_done = ep_done | done
            prev_rew = reward.float().unsqueeze(1)
            prev_done = done.unsqueeze(1)
            if ep_done.all(): break
        
        success += (ep_ret > 0.0).sum().item()
        completed += num_envs
    return success / completed

if __name__ == "__main__":
    train_minigrid_memory()
