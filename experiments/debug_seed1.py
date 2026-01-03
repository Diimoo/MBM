#!/usr/bin/env python3
"""
Debug script to identify why specific seeds fail completely (0% SR).
Adds extensive logging for:
- Weight magnitudes
- Gradient norms
- Plasticity trace magnitudes
- Dopamine/neuromodulator signal ranges
- NaN/Inf checks
"""
import torch
import torch.optim as optim
import numpy as np
import os
import sys
import argparse

sys.path.append(os.getcwd())

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs, ModSignals
from digital_brain.envs.torch_vector_env import TorchVectorPOMDP


def check_for_nan_inf(tensor, name):
    """Check if tensor contains NaN or Inf values."""
    if tensor is None:
        return False, f"{name}: None"
    if torch.isnan(tensor).any():
        return True, f"{name}: Contains NaN!"
    if torch.isinf(tensor).any():
        return True, f"{name}: Contains Inf!"
    return False, f"{name}: OK (min={tensor.min().item():.4f}, max={tensor.max().item():.4f})"


def check_model_health(brain, step, verbose=True):
    """Check model weights and states for issues."""
    issues = []
    
    # Check cortex weights
    if hasattr(brain.cortex.microcircuit, 'layers'):
        for i, layer in enumerate(brain.cortex.microcircuit.layers):
            if hasattr(layer, 'W_ee_values'):
                has_issue, msg = check_for_nan_inf(layer.W_ee_values, f"Layer{i}.W_ee")
                if has_issue:
                    issues.append(msg)
                elif verbose:
                    print(f"  {msg}")
            if hasattr(layer, 'W_ei'):
                w_ei = layer.W_ei.weight if hasattr(layer.W_ei, 'weight') else layer.W_ei
                has_issue, msg = check_for_nan_inf(w_ei, f"Layer{i}.W_ei")
                if has_issue:
                    issues.append(msg)
    else:
        mc = brain.cortex.microcircuit
        if hasattr(mc, 'W_ee_values'):
            has_issue, msg = check_for_nan_inf(mc.W_ee_values, "W_ee")
            if has_issue:
                issues.append(msg)
            elif verbose:
                print(f"  {msg}")
    
    # Check BG weights
    for name, param in brain.bg.named_parameters():
        has_issue, msg = check_for_nan_inf(param, f"BG.{name}")
        if has_issue:
            issues.append(msg)
    
    # Check state
    if brain.state is not None:
        has_issue, msg = check_for_nan_inf(brain.state.z, "state.z")
        if has_issue:
            issues.append(msg)
        elif verbose:
            print(f"  {msg}")
    
    # Check prev mods
    if brain._prev_mods is not None:
        for attr in ['DA', 'NE', 'ACh', 'HT5']:
            val = getattr(brain._prev_mods, attr)
            has_issue, msg = check_for_nan_inf(val, f"mods.{attr}")
            if has_issue:
                issues.append(msg)
            elif verbose and attr == 'DA':
                print(f"  {msg}")
    
    return issues


def check_gradients(brain):
    """Check gradient norms and look for issues."""
    grad_info = {}
    total_norm = 0.0
    max_grad = 0.0
    nan_params = []
    
    for name, param in brain.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info[name] = grad_norm
            total_norm += grad_norm ** 2
            max_grad = max(max_grad, grad_norm)
            
            if torch.isnan(param.grad).any():
                nan_params.append(name)
    
    total_norm = total_norm ** 0.5
    return {
        'total_norm': total_norm,
        'max_grad': max_grad,
        'nan_params': nan_params,
        'per_param': grad_info
    }


def run_debug_session(seed, num_envs=32, num_steps=50, grid_size=5, verbose=True):
    """Run a debug session with extensive monitoring."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"DEBUG SESSION: Seed {seed}, Grid {grid_size}x{grid_size}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Config matching validate_hierarchical.py
    config = {
        'd_obs': 9, 'd_z': 512, 'd_sel': 64, 'd_act': 4,
        'layer_sizes': [256, 512, 256],
        'lr': 3e-4, 'seed': seed, 'num_envs': num_envs, 'num_steps': 64,
        'ppo_epochs': 4, 'mini_batch_size': 2048, 'gamma': 0.99, 'gae_lambda': 0.95,
        'eps_clip': 0.2, 'value_coef': 0.5, 'entropy_coef': 0.01, 'vf_clip': 0.2,
        'total_updates': 50,
        'use_hippocampus': True, 'use_plasticity': True, 
        'use_memory_policy': True, 'use_cerebellum': True,
        'sparse_cortex': False
    }
    
    # Create brain
    print("Creating DigitalBrain...")
    brain = DigitalBrain(config).to(device)
    optimizer = optim.Adam(brain.parameters(), lr=config['lr'])
    
    # Initial weight check
    print("\n--- Initial Weight Check ---")
    issues = check_model_health(brain, 0, verbose=True)
    if issues:
        print(f"INITIAL ISSUES: {issues}")
        return False
    
    # Check parameter initialization ranges
    print("\n--- Parameter Initialization Ranges ---")
    for name, param in brain.named_parameters():
        if 'weight' in name or 'W_' in name:
            print(f"  {name}: shape={param.shape}, mean={param.mean().item():.6f}, std={param.std().item():.6f}")
    
    # Create environment
    print(f"\n--- Creating Environment (seed={seed}) ---")
    env = TorchVectorPOMDP(num_envs=num_envs, size=grid_size, device=device, seed=seed)
    
    # Check environment
    obs = env.reset()
    print(f"  Obs shape: {obs.shape}")
    print(f"  Obs dtype: {obs.dtype}")
    print(f"  Obs range: [{obs.min().item():.3f}, {obs.max().item():.3f}]")
    print(f"  Obs sample:\n    {obs[0].cpu().numpy()}")
    
    # Reset brain
    brain.reset(num_envs, device=device)
    
    prev_reward = torch.zeros(num_envs, 1, device=device)
    prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
    
    # Tracking
    rewards_collected = []
    success_count = 0
    total_episodes = 0
    
    print(f"\n--- Running {num_steps} Steps ---")
    for step in range(num_steps):
        # Forward pass
        with torch.no_grad():
            action, log_prob, value, state, log, entropy = brain.step(
                Obs(x=obs), prev_reward, prev_done, learn=True
            )
        
        # Check for issues every 10 steps
        if step % 10 == 0:
            issues = check_model_health(brain, step, verbose=False)
            if issues:
                print(f"\n!!! ISSUES AT STEP {step} !!!")
                for issue in issues:
                    print(f"  {issue}")
                return False
            
            # Print stats
            print(f"  Step {step}: action_dist={action.float().mean():.2f}, "
                  f"value={value.mean().item():.4f}, "
                  f"DA={brain._prev_mods.DA.mean().item():.4f}, "
                  f"entropy={entropy.mean().item():.4f}")
        
        # Environment step
        obs, reward, done, _ = env.step(action)
        
        # Track rewards
        rewards_collected.append(reward.sum().item())
        success_count += (reward > 5).sum().item()
        total_episodes += done.sum().item()
        
        prev_reward = reward.float().unsqueeze(1)
        prev_done = done.unsqueeze(1)
    
    print(f"\n--- Rollout Stats ---")
    print(f"  Total reward: {sum(rewards_collected):.2f}")
    print(f"  Episodes completed: {total_episodes}")
    print(f"  Successes: {success_count}")
    if total_episodes > 0:
        print(f"  Success rate: {success_count/total_episodes:.2%}")
    
    # Now do a training update to check gradients
    print(f"\n--- Testing Training Update ---")
    
    # Simple PPO-style update
    T = 32
    obs_buf = torch.zeros((T, num_envs, 9), device=device)
    act_buf = torch.zeros((T, num_envs), dtype=torch.long, device=device)
    logp_buf = torch.zeros((T, num_envs), device=device)
    val_buf = torch.zeros((T, num_envs), device=device)
    rew_buf = torch.zeros((T, num_envs), device=device)
    
    # Collect a batch
    obs = env.reset()
    brain.reset(num_envs, device=device)
    prev_reward = torch.zeros(num_envs, 1, device=device)
    prev_done = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
    
    with torch.no_grad():
        for t in range(T):
            obs_buf[t] = obs
            action, log_prob, value, _, _, _ = brain.step(
                Obs(x=obs), prev_reward, prev_done, learn=True
            )
            obs, reward, done, _ = env.step(action)
            
            act_buf[t] = action
            logp_buf[t] = log_prob
            val_buf[t] = value.squeeze(-1)
            rew_buf[t] = reward
            
            prev_reward = reward.float().unsqueeze(1)
            prev_done = done.unsqueeze(1)
    
    # Simple advantage estimate
    returns = rew_buf.sum(dim=0)
    advantages = returns - val_buf.mean(dim=0)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Forward pass for gradients
    brain.reset(num_envs, device=device)
    
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    for t in range(T):
        action, log_prob, value, _, _, entropy = brain.step(
            Obs(x=obs_buf[t]), 
            torch.zeros(num_envs, 1, device=device),
            torch.zeros(num_envs, 1, dtype=torch.bool, device=device),
            learn=False
        )
        
        # Get log prob for actual actions
        with torch.enable_grad():
            gated_x = brain.thalamus.gate(obs_buf[t], brain._prev_selection, brain._prev_mods)
            z_t, pred_t, _ = brain.cortex.forward(gated_x, brain.state.cortex_state, update_trace=False)
            
            logits = brain.bg.policy_head(z_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_logp = dist.log_prob(act_buf[t])
            new_val = brain.bg.value_head(z_t).squeeze(-1)
            
            ratio = torch.exp(new_logp - logp_buf[t])
            policy_loss = -(ratio * advantages).mean()
            value_loss = 0.5 * ((new_val - rew_buf.sum(dim=0)) ** 2).mean()
            
            loss = policy_loss + 0.5 * value_loss
            
            # Check loss
            has_issue, msg = check_for_nan_inf(loss.unsqueeze(0), "loss")
            if has_issue:
                print(f"  {msg}")
                return False
            
            total_loss = total_loss + loss
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check gradients
    grad_info = check_gradients(brain)
    print(f"  Total grad norm: {grad_info['total_norm']:.4f}")
    print(f"  Max grad: {grad_info['max_grad']:.4f}")
    
    if grad_info['nan_params']:
        print(f"  NaN gradients in: {grad_info['nan_params']}")
        return False
    
    # Check for gradient explosion/vanishing
    if grad_info['total_norm'] > 100:
        print(f"  WARNING: Gradient explosion (norm={grad_info['total_norm']:.1f})")
    elif grad_info['total_norm'] < 1e-6:
        print(f"  WARNING: Gradient vanishing (norm={grad_info['total_norm']:.8f})")
    
    # Print top gradient contributors
    print(f"\n  Top 5 gradient norms:")
    sorted_grads = sorted(grad_info['per_param'].items(), key=lambda x: x[1], reverse=True)[:5]
    for name, norm in sorted_grads:
        print(f"    {name}: {norm:.4f}")
    
    # Optimizer step
    torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
    optimizer.step()
    
    # Final check
    print(f"\n--- Post-Update Weight Check ---")
    issues = check_model_health(brain, -1, verbose=True)
    if issues:
        print(f"POST-UPDATE ISSUES: {issues}")
        return False
    
    print(f"\n{'='*60}")
    print(f"DEBUG SESSION COMPLETE: No critical issues found")
    print(f"{'='*60}")
    return True


def compare_seeds(seeds=[0, 1, 2], grid_size=5):
    """Compare behavior across multiple seeds to identify divergent ones."""
    print("\n" + "="*70)
    print("SEED COMPARISON ANALYSIS")
    print("="*70)
    
    results = {}
    for seed in seeds:
        print(f"\n>>> Testing seed {seed}...")
        success = run_debug_session(seed, num_envs=16, num_steps=30, 
                                     grid_size=grid_size, verbose=False)
        results[seed] = success
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for seed, success in results.items():
        status = "✓ OK" if success else "✗ FAILED"
        print(f"  Seed {seed}: {status}")
    
    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\nFailed seeds: {failed}")
        print("Run with verbose=True for detailed diagnostics.")
    else:
        print("\nAll seeds passed basic health checks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug MBM seed failures")
    parser.add_argument("--seed", type=int, default=1, help="Seed to debug")
    parser.add_argument("--compare", action="store_true", help="Compare multiple seeds")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Seeds to compare (with --compare)")
    parser.add_argument("--grid", type=int, default=5, help="Grid size")
    args = parser.parse_args()
    
    if args.compare:
        compare_seeds(args.seeds, args.grid)
    else:
        run_debug_session(args.seed, grid_size=args.grid)
