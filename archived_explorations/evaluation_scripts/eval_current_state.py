import torch
import numpy as np
import copy
import sys
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP
from digital_brain.envs.t_maze import TorchVectorTMaze
from digital_brain.envs.radial_arm_maze import TorchVectorRadialArmMaze

def eval_on_task(model, task_name, device, episodes=64, max_steps=150):
    if task_name == 'gridworld_5x5':
        env = TorchVectorPOMDP(num_envs=64, size=5, device=device)
    elif task_name == 'gridworld_7x7':
        env = TorchVectorPOMDP(num_envs=64, size=7, device=device)
    elif task_name == 'gridworld_10x10':
        env = TorchVectorPOMDP(num_envs=64, size=10, device=device)
    elif task_name == 't_maze_length_5':
        env = TorchVectorTMaze(num_envs=64, corridor_length=5, device=device)
    elif task_name == 'radial_arm_8':
        env = TorchVectorRadialArmMaze(num_envs=64, num_arms=8, arm_length=3, device=device)
    else:
        raise ValueError(f"Unknown task: {task_name}")

    num_envs = env.num_envs
    success = 0
    completed = 0
    
    # Expected observation size for the brain
    d_obs_expected = model.config['d_obs']
    
    # Save model recurrent state if it exists
    orig_state = getattr(model, 'state', None)
    orig_prev_sel = getattr(model, '_prev_selection', None)
    orig_prev_mods = getattr(model, '_prev_mods', None)
    orig_prev_pred = getattr(model, '_prev_pred', None)

    while completed < episodes:
        obs_t = env.reset()
        if hasattr(model, 'reset'):
            model.reset(num_envs, device=device)
            
        prev_reward = torch.zeros(num_envs, 1, device=device)
        prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        
        ep_returns = torch.zeros(num_envs, device=device)
        ep_done = torch.zeros(num_envs, dtype=torch.bool, device=device)
        
        for step in range(max_steps):
            # Pad or truncate observation if necessary
            if obs_t.shape[1] < d_obs_expected:
                padding = torch.zeros(num_envs, d_obs_expected - obs_t.shape[1], device=device)
                obs_padded = torch.cat([obs_t, padding], dim=1)
            elif obs_t.shape[1] > d_obs_expected:
                obs_padded = obs_t[:, :d_obs_expected]
            else:
                obs_padded = obs_t
                
            with torch.no_grad():
                if hasattr(model, 'act'):
                    # Handle both MBM.act and PPOBaseline.act
                    if hasattr(model, 'step'): # MBM
                        obs = Obs(x=obs_padded)
                        out = model.act(obs, prev_reward, prev_done)
                        action = out[0]
                    else: # PPOBaseline
                        out = model.act(obs_padded)
                        action = out[0]
                else:
                    logits, _ = model(obs_padded)
                    action = torch.argmax(logits, dim=-1)
                    
            obs_t, reward, done, _ = env.step(action)
            
            ep_returns += reward * (~ep_done).float()
            ep_done = ep_done | done
            
            prev_reward = reward.float().unsqueeze(1)
            prev_done = done.unsqueeze(1)
            
            if ep_done.all():
                break
        
        success += (ep_returns > 5.0).sum().item()
        completed += num_envs
    
    # Restore model state
    if orig_state is not None:
        model.state = orig_state
    if orig_prev_sel is not None:
        model._prev_selection = orig_prev_sel
    if orig_prev_mods is not None:
        model._prev_mods = orig_prev_mods
    if orig_prev_pred is not None:
        model._prev_pred = orig_prev_pred
    
    return success / completed

def few_shot_eval(brain, task_name, device, episodes=10, max_steps=150):
    """Evaluate how quickly the brain adapts to a new task."""
    # This is a placeholder for few-shot evaluation logic.
    # In a real scenario, we might allow a few steps of learning and see SR improvement.
    return 0.0 # Placeholder

def comprehensive_eval(brain_path=None, num_seeds=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
        'lr': 3.5e-4, 'seed': 42
    }
    
    brain = DigitalBrain(config).to(device)
    if brain_path and os.path.exists(brain_path):
        brain.load_state_dict(torch.load(brain_path, map_location=device), strict=False)
        print(f"Loaded {brain_path}")
    else:
        print("Evaluating fresh/untrained brain")

    results = {
        'gridworld_5x5': [],
        'gridworld_7x7': [],
        'gridworld_10x10': [],
        't_maze_length_5': [],
        'radial_arm_8': [],
        'transfer_5_to_7': [],
    }
    
    for seed in range(num_seeds):
        torch.manual_seed(42 + seed)
        sr_5 = eval_on_task(brain, 'gridworld_5x5', device)
        sr_7 = eval_on_task(brain, 'gridworld_7x7', device)
        sr_10 = eval_on_task(brain, 'gridworld_10x10', device)
        sr_t = eval_on_task(brain, 't_maze_length_5', device)
        sr_r = eval_on_task(brain, 'radial_arm_8', device)
        
        # Transfer Learning: Already evaluated zero-shot above (sr_7).
        # We could add a 'finetuning' step here if desired.
        
        results['gridworld_5x5'].append(sr_5)
        results['gridworld_7x7'].append(sr_7)
        results['gridworld_10x10'].append(sr_10)
        results['t_maze_length_5'].append(sr_t)
        results['radial_arm_8'].append(sr_r)
        results['transfer_5_to_7'].append(sr_7) # Zero-shot transfer
        
    print(f"\n--- Evaluation Results (seeds={num_seeds}) ---")
    for task, srs in results.items():
        print(f"{task:<20} SR: {np.mean(srs):.3f} ± {np.std(srs):.3f}")
    
    return results

if __name__ == "__main__":
    path = "brain_vectorized_best.pth" if os.path.exists("brain_vectorized_best.pth") else None
    comprehensive_eval(path)
