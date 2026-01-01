import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from digital_brain.datatypes import Obs, ModSignals

def evaluate_vectorized(model, env, device, episodes=64, max_steps=150):
    """Internal evaluation for training loops."""
    num_envs = env.num_envs
    success = 0
    completed = 0
    
    # Save training state if possible (only for MBM)
    orig_state = getattr(model, 'state', None)
    
    while completed < episodes:
        obs_t = env.reset()
        if hasattr(model, 'reset'):
            model.reset(num_envs, device=device)
            
        prev_reward = torch.zeros(num_envs, 1, device=device)
        prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        
        ep_returns = torch.zeros(num_envs, device=device)
        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            with torch.no_grad():
                if hasattr(model, 'act'):
                    # MBM brain or PPO baseline
                    # MBM.act returns (action, log_prob, value, state, log, entropy)
                    # PPO.act returns (action, log_prob, value)
                    out = model.act(Obs(x=obs_t) if hasattr(model, 'step') else obs_t, prev_reward, prev_done) if hasattr(model, 'step') else model.act(obs_t)
                    action = out[0]
                else:
                    logits, _ = model(obs_t)
                    action = torch.argmax(logits, dim=-1)
            
            obs_t, reward, done, _ = env.step(action)
            ep_returns += reward * (~ep_done).float()
            ep_done = ep_done | done
            
            if hasattr(model, 'step'):
                prev_reward = reward.float().unsqueeze(1)
                prev_done = done.unsqueeze(1)
            
            if ep_done.all():
                break
        
        success += (ep_returns > 5.0).sum().item()
        completed += num_envs
        
    if orig_state is not None:
        model.state = orig_state
        
    return success / completed

def train_mbm(env, brain, optimizer, config, device, verbose=True, eval_env=None):
    """Reusable MBM training loop for benchmarking."""
    T, E = config['num_steps'], config['num_envs']
    obs_np = env.reset()
    brain.reset(E, device=device)
    
    prev_reward = torch.zeros(E, 1, device=device)
    prev_done = torch.zeros(E, 1, dtype=torch.bool, device=device)

    # Buffers (CPU to save memory)
    obs_buf = torch.zeros((T, E, config['d_obs']), dtype=torch.float32)
    act_buf = torch.zeros((T, E), dtype=torch.long)
    logp_buf = torch.zeros((T, E), dtype=torch.float32)
    val_buf = torch.zeros((T, E), dtype=torch.float32)
    rew_buf = torch.zeros((T, E), dtype=torch.float32)
    done_buf = torch.zeros((T, E), dtype=torch.bool)
    prev_rew_buf = torch.zeros((T, E), dtype=torch.float32)
    prev_done_buf = torch.zeros((T, E), dtype=torch.bool)

    for update in range(config['total_updates']):
        # 1) Collect Experience
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            for t in range(T):
                obs_t = obs_np
                obs_buf[t] = obs_t.cpu()
                prev_rew_buf[t] = prev_reward.squeeze(-1).cpu()
                # Rollout with online learning enabled (plasticity + hippocampal encoding)
                # brain.step now respects granular flags from config internally
                action, log_prob, value, _, _, _ = brain.step(Obs(x=obs_t), prev_reward, prev_done, learn=True)
                
                obs_np, reward, done, _ = env.step(action)
                
                act_buf[t] = action.cpu()
                logp_buf[t] = log_prob.float().cpu()
                val_buf[t] = value.squeeze(-1).float().cpu()
                rew_buf[t] = reward.cpu()
                done_buf[t] = done.cpu()
                
                prev_reward = reward.float().unsqueeze(1)
                prev_done = done.unsqueeze(1)

        # 2) Compute GAE
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            obs_last = obs_np
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

        # 3) PPO Update
        collector_state = (brain.state, brain._prev_selection.clone(), 
                           ModSignals(DA=brain._prev_mods.DA.clone(), NE=brain._prev_mods.NE.clone(),
                                      ACh=brain._prev_mods.ACh.clone(), HT5=brain._prev_mods.HT5.clone()),
                           brain._prev_pred.clone())
        
        envs_per_batch = config['mini_batch_size'] // T
        for epoch in range(config['ppo_epochs']):
            env_perm = torch.randperm(E)
            for batch_start in range(0, E, envs_per_batch):
                batch_end = min(batch_start + envs_per_batch, E)
                env_idx = env_perm[batch_start:batch_end]
                M = len(env_idx)
                
                obs_seq = obs_buf[:, env_idx].to(device)
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
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
                        gated_x = brain.thalamus.gate(obs_t, brain._prev_selection, brain._prev_mods)
                        z_t, pred_t, new_cortex_state = brain.cortex.forward(gated_x, brain.state.cortex_state, update_trace=False)
                        
                        # Use granular flags from config
                        use_hip = config.get('use_hippocampus', True)
                        use_cereb = config.get('use_cerebellum', True)
                        use_mem_pol = config.get('use_memory_policy', True)

                        # Memory retrieval and cerebellum correction
                        retrieved = brain.hippocampus.retrieve(z_t) if use_hip else None
                        correction = None
                        if use_cereb:
                            correction, _ = brain.cerebellum.forward(z_t, obs_t)
                        
                        # Priority 4: Predictive World Model Loss
                        if t < T - 1:
                            wm_loss_t = torch.mean((pred_t - obs_seq[t+1]) ** 2)
                            policy_losses.append(wm_loss_t * 0.1) 
                        
                        selection, da, _, new_logp, value, entropy = brain.bg.step(
                            z_t, 
                            torch.zeros(M, 1, device=device), 
                            None, 
                            torch.zeros(M, 1, dtype=torch.bool, device=device),
                            brain.state.bg_state['prev_value'], 
                            memory_context=retrieved if use_mem_pol else None, 
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
                        
                        surr1 = ratio * adv_seq[t]
                        surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * adv_seq[t]
                        policy_loss_t = -torch.min(surr1, surr2).mean()
                        
                        v_clipped = val_old_seq[t] + (new_val - val_old_seq[t]).clamp(-config['vf_clip'], config['vf_clip'])
                        v_loss1 = (new_val - ret_seq[t])**2
                        v_loss2 = (v_clipped - ret_seq[t])**2
                        val_loss_t = 0.5 * torch.max(v_loss1, v_loss2).mean()
                        
                        policy_losses.append(policy_loss_t)
                        value_losses.append(val_loss_t)

                policy_loss = torch.stack(policy_losses).mean()
                val_loss = torch.stack(value_losses).mean()
                loss = policy_loss + config['value_coef'] * val_loss
                
                if torch.isnan(loss):
                    if verbose:
                        print(f"NaN loss detected at update {update}! Skipping batch.")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                # Check for NaN gradients
                valid_grads = True
                for p in brain.parameters():
                    if p.grad is not None and torch.isnan(p.grad).any():
                        valid_grads = False
                        break
                if valid_grads:
                    optimizer.step()
                else:
                    if verbose:
                        print(f"NaN gradients detected at update {update}! Skipping step.")
                    optimizer.zero_grad()

        brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred = collector_state
        
        if (update + 1) % 10 == 0:
            sr_str = ""
            if eval_env is not None:
                sr = evaluate_vectorized(brain, eval_env, device)
                sr_str = f" | SR: {sr:.3f}"
            if verbose:
                print(f"Update {update+1}/{config['total_updates']} | Loss: {loss.item():.4f}{sr_str}")

def train_ppo_baseline(env, model, optimizer, config, device, verbose=True, eval_env=None):
    """Simplified standard PPO training loop for baseline comparison."""
    T, E = config['num_steps'], config['num_envs']
    obs_t = env.reset()
    
    # Buffers
    obs_buf = torch.zeros((T, E, config['d_obs']), device=device)
    act_buf = torch.zeros((T, E), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, E), device=device)
    val_buf = torch.zeros((T, E), device=device)
    rew_buf = torch.zeros((T, E), device=device)
    done_buf = torch.zeros((T, E), dtype=torch.bool, device=device)

    for update in range(config['total_updates']):
        # 1) Collect Experience
        with torch.no_grad():
            for t in range(T):
                obs_buf[t] = obs_t
                action, logp, value = model.act(obs_t)
                obs_next, reward, done, _ = env.step(action)
                
                act_buf[t] = action
                logp_buf[t] = logp
                val_buf[t] = value.squeeze(-1)
                rew_buf[t] = reward
                done_buf[t] = done
                obs_t = obs_next

        # 2) Compute GAE
        with torch.no_grad():
            _, next_value = model(obs_t)
            next_value = next_value.squeeze(-1)
            
        adv_buf = torch.zeros_like(rew_buf)
        last_gae = 0
        for t in reversed(range(T)):
            next_non_terminal = (~done_buf[t]).float()
            next_val = next_value if t == T - 1 else val_buf[t+1]
            delta = rew_buf[t] + config['gamma'] * next_val * next_non_terminal - val_buf[t]
            adv_buf[t] = last_gae = delta + config['gamma'] * config['gae_lambda'] * next_non_terminal * last_gae
        
        ret_buf = adv_buf + val_buf
        
        # 3) PPO Update
        for _ in range(config['ppo_epochs']):
            logits, values = model(obs_buf.view(-1, obs_buf.shape[-1]))
            values = values.squeeze(-1)
            dist = torch.distributions.Categorical(logits=logits)
            
            new_logp = dist.log_prob(act_buf.view(-1))
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_logp - logp_buf.view(-1))
            mb_adv = adv_buf.view(-1)
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            
            val_loss = 0.5 * ((values - ret_buf.view(-1))**2).mean()
            
            loss = policy_loss + config['value_coef'] * val_loss - config['entropy_coef'] * entropy
            
            if torch.isnan(loss):
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

        if (update + 1) % 10 == 0:
            sr_str = ""
            if eval_env is not None:
                sr = evaluate_vectorized(model, eval_env, device)
                sr_str = f" | SR: {sr:.3f}"
            if verbose:
                print(f"Update {update+1}/{config['total_updates']} | Loss: {loss.item():.4f}{sr_str}")
