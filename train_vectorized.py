import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.vector_env import VectorPOMDP

def train_vectorized():
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4, 
        'lr': 3e-4, # Lower LR for PPO stability
        'total_steps': 100000000,
        'num_envs': 4096, # Reduced for PPO buffer memory
        'num_steps': 128, # More steps for better GAE
        'ppo_epochs': 4,
        'mini_batch_size': 4096,
        'eps_clip': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'entropy_coef': 0.01,
        'value_coef': 0.5,
        'seed': 42,
        'eval_every': 10, # updates
        'eval_episodes': 20,
        'selection_penalty': 0.001
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Parallel Envs: {config['num_envs']}")

    envs = VectorPOMDP(num_envs=config['num_envs'], size=5, seed=config['seed'])
    eval_env = VectorPOMDP(num_envs=1, size=5, seed=config['seed'] + 1000)

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
                
                action, log_prob, value, _, _, _ = brain.step(Obs(x=obs_t), prev_reward, prev_done, learn=True)
                
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

        # 3) PPO Update Epochs
        # Save collector state to restore after updates
        collector_state = (
            brain.state, 
            brain._prev_selection, 
            brain._prev_mods, 
            brain._prev_pred
        )

        for _ in range(config['ppo_epochs']):
            indices = torch.randperm(obs_f.shape[0], device=device)
            for start in range(0, obs_f.shape[0], config['mini_batch_size']):
                end = start + config['mini_batch_size']
                idx = indices[start:end]
                
                # Reconstruct brain state for this minibatch
                mb_B = idx.shape[0]
                from digital_brain.datatypes import BrainState, ModSignals
                brain.state = BrainState(
                    z=z_f[idx],
                    cortex_state=(e_f[idx], i_f[idx], collector_state[0].cortex_state[2]), # Shared trace
                    bg_state={'prev_value': bgv_f[idx]},
                    hip_state=None, cerebellum_state=None
                )
                brain._prev_selection = sel_f[idx]
                brain._prev_mods = ModSignals(DA=da_f[idx], NE=ne_f[idx], ACh=ach_f[idx], HT5=ht_f[idx])
                brain._prev_pred = pred_f[idx]
                
                # Forward pass for minibatch
                gated_x = brain.thalamus.gate(obs_f[idx], brain._prev_selection, brain._prev_mods)
                z_t, _, _ = brain.cortex.forward(gated_x, brain.state.cortex_state)
                
                logits = brain.bg.policy_head(z_t)
                probs = torch.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_logp = dist.log_prob(act_f[idx])
                entropy = dist.entropy()
                new_val = brain.bg.value_head(z_t).squeeze(-1)
                
                # Ratio and Clipped Objective
                ratio = torch.exp(new_logp - logp_f[idx])
                mb_adv = adv_f[idx]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - config['eps_clip'], 1.0 + config['eps_clip']) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss (Huber)
                val_loss = nn.HuberLoss()(new_val, ret_f[idx])
                
                # Total Loss
                loss = policy_loss + config['value_coef'] * val_loss - config['entropy_coef'] * entropy.mean()
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                optimizer.step()

        # Restore collector state for next rollout
        brain.state, brain._prev_selection, brain._prev_mods, brain._prev_pred = collector_state

        fps = (config['num_envs'] * config['num_steps']) / (time.time() - start_time)

        if (update + 1) % config['eval_every'] == 0:
            sr = eval_vectorized(brain, eval_env, device, config['eval_episodes'], 150)
            print(f"Update {update+1}/{num_updates} | SR: {sr:.2f} | Loss: {loss.item():.4f} | FPS: {fps:.0f}")
            if sr > best_sr:
                best_sr = sr
                torch.save(brain.state_dict(), "brain_vectorized_best.pth")
                print(f"New Best SR: {best_sr:.2f}")

@torch.no_grad()
def eval_vectorized(brain, env, device, episodes, max_steps):
    success = 0
    # Ensure hippocampus is clear before evaluation if we want truly clean starts
    # brain.hippocampus.clear() 
    for _ in range(episodes):
        obs_np = env.reset()
        brain.reset(1, device=device)
        prev_reward = torch.zeros(1, 1, device=device)
        prev_done = torch.zeros(1, 1, dtype=torch.bool, device=device)
        done = [False]
        ep_ret = 0.0
        steps = 0
        while not done[0] and steps < max_steps:
            obs = Obs(x=torch.from_numpy(obs_np).to(device))
            # Use act() or learn=False for evaluation
            action, _, _, _, _, _ = brain.act(obs, prev_reward, prev_done)
            obs_np, reward, done, _ = env.step(action.cpu().numpy())
            ep_ret += reward[0]
            prev_reward = torch.from_numpy(reward).float().to(device).unsqueeze(1)
            prev_done = torch.from_numpy(done).to(device).unsqueeze(1)
            steps += 1
        if ep_ret > 5.0:
            success += 1
    return success / episodes

if __name__ == "__main__":
    train_vectorized()
