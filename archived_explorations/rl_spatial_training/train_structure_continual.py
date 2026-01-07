"""
Harder Continual Learning: Different Linguistic Structures

The original English/French ablation showed no differentiation because both languages
use similar structure (just different tokens for same concepts).

This test uses TRULY different structures:
- English (SVO): "go to red circle"
- German-style (SOV): "red circle to go"  

This tests if hippocampus can protect structural knowledge during continual learning.
"""

import torch
import torch.nn.functional as F
import numpy as np
import random
import json
from typing import Dict, List, Tuple

from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs


class StructuredLanguageEnv:
    """Environment with different word order structures."""
    
    def __init__(self, size: int = 5, structure: str = 'svo'):
        """
        Args:
            structure: 'svo' (Subject-Verb-Object) or 'sov' (Subject-Object-Verb)
        """
        self.size = size
        self.structure = structure
        
        self.colors = ['red', 'blue', 'green']
        self.shapes = ['circle', 'square', 'triangle']
        
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        self.agent_pos = None
        self.objects = {}
        self.target_pos = None
        self.instruction = None
        self.steps = 0
        self.max_steps = 30
        
    def _build_vocab(self) -> Dict[str, int]:
        return {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'go': 3, 'to': 4,
            'red': 5, 'blue': 6, 'green': 7,
            'circle': 8, 'square': 9, 'triangle': 10,
        }
    
    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab['<START>']]
        for word in text.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens.append(self.vocab['<END>'])
        return tokens
    
    def pad_tokens(self, tokens: List[int], max_len: int = 8) -> List[int]:
        if len(tokens) >= max_len:
            return tokens[:max_len]
        return tokens + [self.vocab['<PAD>']] * (max_len - len(tokens))
    
    def _generate_instruction(self, color: str, shape: str) -> str:
        """Generate instruction based on structure type."""
        if self.structure == 'svo':
            # Subject-Verb-Object: "go to red circle"
            return f"go to {color} {shape}"
        elif self.structure == 'sov':
            # Subject-Object-Verb: "red circle to go"
            return f"{color} {shape} to go"
        else:
            raise ValueError(f"Unknown structure: {self.structure}")
    
    def reset(self) -> Tuple[np.ndarray, List[int]]:
        self.steps = 0
        self.objects = {}
        
        # Place 3 objects (one of each color)
        positions = []
        for color in self.colors:
            shape = random.choice(self.shapes)
            while True:
                pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
                if pos not in positions:
                    positions.append(pos)
                    self.objects[(color, shape)] = pos
                    break
        
        # Agent starts randomly
        while True:
            self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if self.agent_pos not in positions:
                break
        
        # Pick random target
        target_key = random.choice(list(self.objects.keys()))
        color, shape = target_key
        self.target_pos = self.objects[target_key]
        
        self.instruction = self._generate_instruction(color, shape)
        tokens = self.tokenize(self.instruction)
        tokens = self.pad_tokens(tokens)
        
        obs = self._get_obs()
        return obs, tokens
    
    def _get_obs(self) -> np.ndarray:
        obs = np.zeros(self.size * self.size + 2, dtype=np.float32)
        
        # Agent
        agent_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        obs[agent_idx] = 1.0
        
        # Objects (with color encoding)
        for (color, shape), pos in self.objects.items():
            obj_idx = pos[0] * self.size + pos[1]
            color_val = (self.colors.index(color) + 1) / len(self.colors)
            obs[obj_idx] = color_val
        
        obs[-2] = self.steps / self.max_steps
        obs[-1] = 1.0 if self.structure == 'svo' else 0.0
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        self.steps += 1
        reward = -0.02
        done = False
        info = {'success': False}
        
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = (new_x, new_y)
        
        if self.agent_pos == self.target_pos:
            reward = 1.0
            done = True
            info['success'] = True
        elif self.steps >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        return obs, reward, done, info


class VectorizedStructuredEnv:
    """Vectorized environment for parallel training."""
    
    def __init__(self, num_envs: int, size: int = 5, structure: str = 'svo'):
        self.num_envs = num_envs
        self.structure = structure
        self.envs = [StructuredLanguageEnv(size, structure) for _ in range(num_envs)]
        self.vocab_size = self.envs[0].vocab_size
        
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_list, inst_list = [], []
        for env in self.envs:
            obs, tokens = env.reset()
            obs_list.append(obs)
            inst_list.append(tokens)
        return (torch.tensor(np.stack(obs_list), dtype=torch.float32),
                torch.tensor(np.array(inst_list), dtype=torch.long))
    
    def step(self, actions: torch.Tensor):
        obs_list, rewards, dones, infos = [], [], [], []
        actions_np = actions.cpu().numpy()
        
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(actions_np[i]))
            if done:
                obs, _ = env.reset()
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        return (torch.tensor(np.stack(obs_list), dtype=torch.float32),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.bool),
                infos)
    
    def get_instructions(self) -> torch.Tensor:
        inst_list = []
        for env in self.envs:
            tokens = env.tokenize(env.instruction)
            tokens = env.pad_tokens(tokens)
            inst_list.append(tokens)
        return torch.tensor(np.array(inst_list), dtype=torch.long)


def evaluate_structure(brain, structure: str, num_episodes: int = 200, 
                      device: str = 'cuda') -> float:
    """Evaluate on specific structure."""
    env = VectorizedStructuredEnv(num_envs=64, structure=structure)
    brain.eval()
    successes = 0
    total = 0
    
    with torch.no_grad():
        episodes_done = 0
        while episodes_done < num_episodes:
            obs, instructions = env.reset()
            obs = obs.to(device)
            instructions = instructions.to(device)
            
            brain.reset(env.num_envs)
            prev_reward = torch.zeros(env.num_envs, device=device)
            prev_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            
            episode_done = [False] * env.num_envs
            
            for step in range(30):
                obs_wrapped = Obs(x=obs)
                action, _, _, _, _, _ = brain.act(
                    obs_wrapped, prev_reward, prev_done, instruction=instructions
                )
                
                obs, rewards, dones, infos = env.step(action)
                obs = obs.to(device)
                instructions = env.get_instructions().to(device)
                prev_reward = rewards.to(device)
                prev_done = dones.to(device)
                
                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done and not episode_done[i]:
                        episode_done[i] = True
                        episodes_done += 1
                        total += 1
                        if info.get('success', False):
                            successes += 1
                
                if all(episode_done) or episodes_done >= num_episodes:
                    break
    
    brain.train()
    return successes / max(total, 1) * 100


def train_stage(brain, optimizer, env, num_updates: int, device: str):
    """Train for one stage."""
    for update in range(num_updates):
        obs_buf, inst_buf, act_buf, rew_buf, done_buf, val_buf, logp_buf = [], [], [], [], [], [], []
        
        obs, instructions = env.reset()
        obs = obs.to(device)
        instructions = instructions.to(device)
        brain.reset(env.num_envs)
        
        prev_reward = torch.zeros(env.num_envs, device=device)
        prev_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        
        for step in range(32):
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
        
        # GAE and PPO update
        T, B = len(rew_buf), env.num_envs
        obs_t = torch.stack(obs_buf)
        inst_t = torch.stack(inst_buf)
        act_t = torch.stack(act_buf)
        rew_t = torch.stack(rew_buf)
        done_t = torch.stack(done_buf)
        val_t = torch.stack(val_buf)
        logp_t = torch.stack(logp_buf)
        
        advantages = torch.zeros_like(rew_t)
        last_gae = 0
        for t in reversed(range(T)):
            next_val = val_t[t] if t == T - 1 else val_t[t + 1]
            delta = rew_t[t] + 0.99 * next_val * (~done_t[t]).float() - val_t[t]
            advantages[t] = last_gae = delta + 0.99 * 0.95 * (~done_t[t]).float() * last_gae
        returns = advantages + val_t
        
        total = T * B
        obs_flat = obs_t.view(total, -1).to(device)
        inst_flat = inst_t.view(total, -1).to(device)
        act_flat = act_t.view(total).to(device)
        oldlp_flat = logp_t.view(total).to(device)
        adv_flat = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).view(total).to(device)
        ret_flat = returns.view(total).to(device)
        
        for epoch in range(4):
            indices = torch.randperm(total, device=device)
            for start in range(0, total, 512):
                idx = indices[start:start+512]
                mb_obs, mb_inst = obs_flat[idx], inst_flat[idx]
                mb_act, mb_oldlp = act_flat[idx], oldlp_flat[idx]
                mb_adv, mb_ret = adv_flat[idx], ret_flat[idx]
                
                brain.reset(len(idx))
                z_t, _, _ = brain.cortex(mb_obs, brain.state.cortex_state)
                if brain.use_language:
                    lang_h = brain.lang_encoder(mb_inst)
                    z_t = brain.lang_projection(torch.cat([z_t, lang_h], dim=-1))
                
                logits = torch.clamp(brain.bg.policy_head(z_t), -20, 20)
                values = brain.bg.value_head(z_t).squeeze(-1)
                
                dist = torch.distributions.Categorical(logits=logits)
                new_lp = dist.log_prob(mb_act)
                ent = dist.entropy().mean()
                
                ratio = torch.exp(new_lp - mb_oldlp)
                policy_loss = -torch.min(ratio * mb_adv, 
                                        torch.clamp(ratio, 0.8, 1.2) * mb_adv).mean()
                value_loss = F.mse_loss(values, mb_ret)
                loss = policy_loss + 0.5 * value_loss - 0.01 * ent
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
                optimizer.step()


def run_structure_continual(seed: int = 42, device: str = 'cuda'):
    """Run continual learning with different structures."""
    
    print("=" * 70)
    print("CONTINUAL LEARNING: DIFFERENT STRUCTURES")
    print("=" * 70)
    print("Stage 1: Train English (SVO): 'go to red circle'")
    print("Stage 2: Train German-style (SOV): 'red circle to go'")
    print("Measure: Forgetting of SVO after learning SOV")
    print(f"Seed: {seed}")
    print("=" * 70)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        device = 'cpu'
    
    config = {
        'd_obs': 27,  # 5x5 + 2
        'd_z': 64,
        'd_sel': 32,
        'd_act': 4,
        'use_language': True,
        'vocab_size': 15,
        'd_lang_embed': 32,
        'd_lang_hidden': 64,
        'use_hippocampus': True,
        'use_plasticity': True,
        'use_memory_policy': False,
        'use_cerebellum': True,
        'device': device,
    }
    
    brain = DigitalBrain(config).to(device)
    optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)
    
    svo_env = VectorizedStructuredEnv(num_envs=128, structure='svo')
    sov_env = VectorizedStructuredEnv(num_envs=128, structure='sov')
    
    # Stage 1: Train SVO
    print("\n--- Stage 1: Training SVO (English) ---")
    for i in range(100):
        train_stage(brain, optimizer, svo_env, 1, device)
        if (i + 1) % 25 == 0:
            acc = evaluate_structure(brain, 'svo', 200, device)
            print(f"  Update {i+1}: SVO={acc:.1f}%")
    
    svo_before = evaluate_structure(brain, 'svo', 300, device)
    print(f"\n  SVO accuracy BEFORE SOV training: {svo_before:.1f}%")
    
    # Stage 2: Train SOV
    print("\n--- Stage 2: Training SOV (German-style) ---")
    for i in range(100):
        train_stage(brain, optimizer, sov_env, 1, device)
        if (i + 1) % 25 == 0:
            svo_acc = evaluate_structure(brain, 'svo', 200, device)
            sov_acc = evaluate_structure(brain, 'sov', 200, device)
            print(f"  Update {i+1}: SVO={svo_acc:.1f}% | SOV={sov_acc:.1f}%")
    
    # Final evaluation
    svo_after = evaluate_structure(brain, 'svo', 300, device)
    sov_final = evaluate_structure(brain, 'sov', 300, device)
    
    forgetting = (svo_before - svo_after) / max(svo_before, 1) * 100
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"SVO (English) BEFORE SOV: {svo_before:.1f}%")
    print(f"SVO (English) AFTER SOV:  {svo_after:.1f}%")
    print(f"SOV (German) final:       {sov_final:.1f}%")
    print(f"Forgetting:               {forgetting:.1f}%")
    print("=" * 70)
    
    return {
        'svo_before': svo_before,
        'svo_after': svo_after,
        'sov_final': sov_final,
        'forgetting': forgetting
    }


def run_structure_ablation(seeds: list = [42, 43, 44], device: str = 'cuda'):
    """Run ablation study on structure-based continual learning."""
    
    print("\n" + "#" * 70)
    print("# STRUCTURE-BASED CONTINUAL LEARNING ABLATION")
    print("#" * 70)
    
    configs = {
        'Full MBM': {'use_hippocampus': True, 'use_plasticity': True},
        'No Hippocampus': {'use_hippocampus': False, 'use_plasticity': True},
        'No Plasticity': {'use_hippocampus': True, 'use_plasticity': False},
        'Baseline': {'use_hippocampus': False, 'use_plasticity': False},
    }
    
    all_results = {}
    
    for seed in seeds:
        print(f"\n{'#' * 70}")
        print(f"SEED {seed}")
        print('#' * 70)
        
        for config_name, overrides in configs.items():
            print(f"\n{'=' * 60}")
            print(f"Config: {config_name}")
            print('=' * 60)
            
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            
            if device == 'cuda' and torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
            
            config = {
                'd_obs': 27, 'd_z': 64, 'd_sel': 32, 'd_act': 4,
                'use_language': True, 'vocab_size': 15,
                'd_lang_embed': 32, 'd_lang_hidden': 64,
                'use_memory_policy': False, 'use_cerebellum': True,
                'device': device,
            }
            config.update(overrides)
            
            brain = DigitalBrain(config).to(device)
            optimizer = torch.optim.Adam(brain.parameters(), lr=3e-4)
            
            svo_env = VectorizedStructuredEnv(num_envs=128, structure='svo')
            sov_env = VectorizedStructuredEnv(num_envs=128, structure='sov')
            
            # Train SVO
            print("  Training SVO...")
            for i in range(100):
                train_stage(brain, optimizer, svo_env, 1, device)
            svo_before = evaluate_structure(brain, 'svo', 200, device)
            print(f"  SVO before: {svo_before:.1f}%")
            
            # Train SOV
            print("  Training SOV...")
            for i in range(100):
                train_stage(brain, optimizer, sov_env, 1, device)
            
            svo_after = evaluate_structure(brain, 'svo', 200, device)
            sov_final = evaluate_structure(brain, 'sov', 200, device)
            forgetting = (svo_before - svo_after) / max(svo_before, 1) * 100
            
            print(f"  SVO after: {svo_after:.1f}% | SOV: {sov_final:.1f}%")
            print(f"  Forgetting: {forgetting:.1f}%")
            
            key = f"{config_name}_seed{seed}"
            all_results[key] = {
                'config': config_name, 'seed': seed,
                'svo_before': svo_before, 'svo_after': svo_after,
                'sov_final': sov_final, 'forgetting': forgetting
            }
    
    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)
    
    for config_name in configs.keys():
        forget_vals = [all_results[f"{config_name}_seed{s}"]['forgetting'] for s in seeds]
        svo_vals = [all_results[f"{config_name}_seed{s}"]['svo_after'] for s in seeds]
        sov_vals = [all_results[f"{config_name}_seed{s}"]['sov_final'] for s in seeds]
        
        print(f"\n{config_name}:")
        print(f"  Forgetting: {np.mean(forget_vals):.1f}% ± {np.std(forget_vals):.1f}%")
        print(f"  SVO after:  {np.mean(svo_vals):.1f}% ± {np.std(svo_vals):.1f}%")
        print(f"  SOV final:  {np.mean(sov_vals):.1f}% ± {np.std(sov_vals):.1f}%")
    
    # Save
    with open('experiments/structure_continual_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nResults saved to experiments/structure_continual_results.json")
    
    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='single', choices=['single', 'ablation'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    if args.mode == 'single':
        run_structure_continual(seed=args.seed, device=args.device)
    else:
        run_structure_ablation(seeds=[42, 43, 44], device=args.device)
