import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
import sys
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.vector_env import VectorPOMDP

def train_vectorized():
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4, 
        'lr': 2.5e-4,
        'total_steps': 100000000,
        'num_envs': 4096,
        'num_steps': 128,
        'ppo_epochs': 3,
        'mini_batch_size': 32768,  
        'eps_clip': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.02,  
        'value_coef': 0.5,
        'target_kl': 0.015,  
        'seed': 42,
        'eval_every': 20,
        'eval_episodes': 200,  
        'num_eval_envs': 64,  
        'selection_penalty': 0.001
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Parallel Envs: {config['num_envs']}")

    envs = VectorPOMDP(num_envs=config['num_envs'], size=5, seed=config['seed'])
    eval_env = VectorPOMDP(num_envs=config['num_eval_envs'], size=5, seed=config['seed'] + 1000)

    brain = DigitalBrain(config).to(device)
    
    if os.path.exists("brain_vectorized_best.pth"):
        try:
            state_dict = torch.load("brain_vectorized_best.pth", map_location=device)
            brain.load_state_dict(state_dict, strict=False)
            print("Loaded existing vectorized best")
        except Exception as e:
            print(f"Starting fresh: {e}")

    optimizer = optim.Adam(brain.parameters(), lr=config['lr'], eps=1e-5)
    best_sr = 0.0

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
        
        # Buffer storage
        obs_buf = torch.zeros((config['num_steps'], config['num_envs'], config['d_obs']), device=device)
        act_buf = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        logp_buf = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        val_buf = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        rew_buf = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        done_buf = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        
        # Recurrent state buffers (to re-run PPO)
        # We treat internal brain state as part of the observation for PPO update
        z_buf = torch.zeros((config['num_steps'], config['num_envs'], config['d_z']), device=device)
        e_act_buf = torch.zeros((config['num_steps'], config['num_envs'], config['d_z']), device=device)
        i_act_buf = torch.zeros((config['num_steps'], config['num_envs'], config['d_z']), device=device)
        bg_val_buf = torch.zeros((config['num_steps'], config['num_envs'], 1), device=device)
        sel_buf = torch.zeros((config['num_steps'], config['num_envs'], config['d_sel']), device=device)
        mod_da = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        mod_ne = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        mod_ach = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        mod_5ht = torch.zeros((config['num_steps'], config['num_envs']), device=device)
        pred_buf = torch.zeros((config['num_steps'], config['num_envs'], config['d_obs']), device=device)

        # 1) Collect Experience
        for t in range(config['num_steps']):
            with torch.no_grad():
                # Store current state BEFORE step
                z_buf[t] = brain.state.z
                e_act_buf[t], i_act_buf[t], _ = brain.state.cortex_state
                bg_val_buf[t] = brain.state.bg_state['prev_value']
                sel_buf[t] = brain._prev_selection
                mod_da[t] = brain._prev_mods.DA
                mod_ne[t] = brain._prev_mods.NE
                mod_ach[t] = brain._prev_mods.ACh
                mod_5ht[t] = brain._prev_mods.HT5
                pred_buf[t] = brain._prev_pred
                
                obs_t = torch.from_numpy(obs_np).to(device)
                obs_buf[t] = obs_t
                
                action, log_prob, value, _, _, _ = brain.step(Obs(x=obs_t), prev_reward, prev_done, learn=False)
                
                obs_np, reward, done, _ = envs.step(action.cpu().numpy())
                
                act_buf[t] = action
                logp_buf[t] = log_prob
                val_buf[t] = value.squeeze(-1)
                rew_buf[t] = torch.from_numpy(reward).to(device)
                done_buf[t] = torch.from_numpy(done).to(device).float()
                
                prev_reward = torch.from_numpy(reward).float().to(device).unsqueeze(1)
                prev_done = torch.from_numpy(done).to(device).unsqueeze(1)

        # 2) Compute GAE
        with torch.no_grad():
            obs_last = torch.from_numpy(obs_np).to(device)
            gated_last = brain.thalamus.gate(obs_last, brain._prev_selection, brain._prev_mods)
            z_last, _, _ = brain.cortex.forward(gated_last, brain.state.cortex_state)
            next_value = brain.bg.value_head(z_last).squeeze(-1)
            
        adv_buf = torch.zeros_like(rew_buf)
        last_gae_lam = 0
        for t in reversed(range(config['num_steps'])):
            next_non_terminal = 1.0 - done_buf[t]
            next_values = next_value if t == config['num_steps'] - 1 else val_buf[t + 1]
            delta = rew_buf[t] + config['gamma'] * next_values * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae_lam = delta + config['gamma'] * config['gae_lambda'] * next_non_terminal * last_gae_lam
        
        ret_buf = adv_buf + val_buf

        # Flatten buffers
        obs_f = obs_buf.reshape(-1, config['d_obs'])
        act_f = act_buf.reshape(-1)
        logp_f = logp_buf.reshape(-1)
        adv_f = adv_buf.reshape(-1)
        ret_f = ret_buf.reshape(-1)
        val_f = val_buf.reshape(-1)
        
        # Flatten state buffers
        z_f = z_buf.reshape(-1, config['d_z'])
        e_f = e_act_buf.reshape(-1, config['d_z'])
        i_f = i_act_buf.reshape(-1, config['d_z'])
        bgv_f = bg_val_buf.reshape(-1, 1)
        sel_f = sel_buf.reshape(-1, config['d_sel'])
        da_f = mod_da.reshape(-1)
        ne_f = mod_ne.reshape(-1)
        ach_f = mod_ach.reshape(-1)
        ht_f = mod_5ht.reshape(-1)
        pred_f = pred_buf.reshape(-1, config['d_obs'])

        # 3) PPO Update Epochs with KL Early-Stop and Diagnostics
        collector_state = (
            brain.state, 
            brain._prev_selection, 
            brain._prev_mods, 
            brain._prev_pred
        )
        from digital_brain.datatypes import BrainState, ModSignals

        # Diagnostics accumulators
        total_kl, total_clipfrac, total_entropy, total_batches = 0.0, 0.0, 0.0, 0
        early_stop_epoch = config['ppo_epochs']
        
        for epoch in range(config['ppo_epochs']):
            epoch_kl = 0.0
            epoch_batches = 0
            indices = torch.randperm(obs_f.shape[0], device=device)
            
            for start in range(0, obs_f.shape[0], config['mini_batch_size']):
                end = start + config['mini_batch_size']
                idx = indices[start:end]
                
                # Reconstruct brain state for this minibatch
                brain.state = BrainState(
                    z=z_f[idx],
                    cortex_state=(e_f[idx], i_f[idx], collector_state[0].cortex_state[2]),
                    bg_state={'prev_value': bgv_f[idx]},
                    hip_state=None, cerebellum_state=None
                )
                brain._prev_selection = sel_f[idx]
                brain._prev_mods = ModSignals(DA=da_f[idx], NE=ne_f[idx], ACh=ach_f[idx], HT5=ht_f[idx])
                brain._prev_pred = pred_f[idx]
                
                # Forward pass
                gated_x = brain.thalamus.gate(obs_f[idx], brain._prev_selection, brain._prev_mods)
                z_t, _, _ = brain.cortex.forward(gated_x, brain.state.cortex_state)
                
                logits = brain.bg.policy_head(z_t)
                logits = torch.clamp(logits, min=-20, max=20)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs, validate_args=False)
                
                new_logp = dist.log_prob(act_f[idx])
                entropy = dist.entropy()
                new_val = brain.bg.value_head(z_t).squeeze(-1)
                
                # PPO Diagnostics
                with torch.no_grad():
                    log_ratio = new_logp - logp_f[idx]
                    approx_kl = (-log_ratio).mean().item()  # KL(old||new)
                    clipfrac = ((torch.exp(log_ratio) - 1.0).abs() > config['eps_clip']).float().mean().item()
                
                total_kl += approx_kl
                total_clipfrac += clipfrac
                total_entropy += entropy.mean().item()
                total_batches += 1
                epoch_kl += approx_kl
                epoch_batches += 1
                
                # Ratio and Clipped Objective
                ratio = torch.exp(log_ratio)
                mb_adv = adv_f[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                val_loss = nn.HuberLoss()(new_val, ret_f[idx])
                loss = policy_loss + config['value_coef'] * val_loss - config['entropy_coef'] * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                optimizer.step()
            
            # KL Early-Stop check after each epoch
            epoch_kl_avg = epoch_kl / max(epoch_batches, 1)
            if epoch_kl_avg > 1.5 * config['target_kl']:
                early_stop_epoch = epoch + 1
                break

        # Restore collector state
        brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred = collector_state
        
        # Compute diagnostics averages
        avg_kl = total_kl / max(total_batches, 1)
        avg_clipfrac = total_clipfrac / max(total_batches, 1)
        avg_entropy = total_entropy / max(total_batches, 1)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)

        if (update + 1) % config['eval_every'] == 0:
            # Save training state before eval
            train_state = (brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred)
            sr = eval_vectorized(brain, eval_env, device, config['eval_episodes'], 150)
            # Restore training state after eval
            brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred = train_state
            
            # Log with PPO diagnostics
            print(f"Upd {update+1:4d} | SR {sr:.3f} | KL {avg_kl:.4f} | Clip {avg_clipfrac:.2f} | Ent {avg_entropy:.2f} | Ep {early_stop_epoch} | FPS {fps:.0f}")
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
        obs_np = env.reset()
        brain.reset(num_envs, device=device)
        prev_reward = torch.zeros(num_envs, 1, device=device)
        prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        
        ep_returns = torch.zeros(num_envs, device=device)
        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            obs = Obs(x=torch.from_numpy(obs_np).to(device))
            action, _, _, _, _, _ = brain.act(obs, prev_reward, prev_done)
            obs_np, reward, done, _ = env.step(action.cpu().numpy())
            
            reward_t = torch.from_numpy(reward).to(device)
            done_t = torch.from_numpy(done).to(device)
            
            # Only accumulate for not-yet-done episodes
            ep_returns += reward_t * (~ep_done).float()
            ep_done = ep_done | done_t
            
            prev_reward = reward_t.float().unsqueeze(1)
            prev_done = done_t.unsqueeze(1)
            
            if ep_done.all():
                break
        
        # Count successes
        success += (ep_returns > 5.0).sum().item()
        completed += num_envs
    
    return success / completed

if __name__ == "__main__":
    train_vectorized()
