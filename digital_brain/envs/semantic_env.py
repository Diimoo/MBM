"""
Semantic Understanding Environment

Tests:
1. Negation reasoning: "go to the circle that is NOT blue"
2. Synonym understanding: "crimson" -> red (few-shot learning)
3. Relational concepts: "go to the ball NEAR the square"
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

from ..datatypes import Obs


class SemanticGridworld:
    """Gridworld with semantic language understanding challenges."""
    
    def __init__(self, size: int = 5, mode: str = 'negation'):
        """
        Args:
            size: Grid size
            mode: 'negation', 'synonym', or 'relation'
        """
        self.size = size
        self.mode = mode
        
        self.colors = ['red', 'blue', 'green']
        self.shapes = ['circle', 'square', 'triangle']
        
        # Synonyms for few-shot learning
        self.synonyms = {
            'crimson': 'red',
            'scarlet': 'red', 
            'azure': 'blue',
            'cobalt': 'blue',
            'emerald': 'green',
            'jade': 'green',
        }
        
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # State
        self.agent_pos = None
        self.objects = []  # [(color, shape, pos), ...]
        self.target = None
        self.instruction = None
        self.steps = 0
        self.max_steps = 30
        
    def _build_vocab(self) -> Dict[str, int]:
        vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'go': 3, 'to': 4, 'the': 5, 'that': 6, 'is': 7,
            'not': 8, 'near': 9, 'far': 10, 'from': 11,
            'red': 12, 'blue': 13, 'green': 14,
            'circle': 15, 'square': 16, 'triangle': 17,
            # Synonyms
            'crimson': 18, 'scarlet': 19, 'azure': 20, 
            'cobalt': 21, 'emerald': 22, 'jade': 23,
        }
        return vocab
    
    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab['<START>']]
        for word in text.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens.append(self.vocab['<END>'])
        return tokens
    
    def pad_tokens(self, tokens: List[int], max_len: int = 12) -> List[int]:
        if len(tokens) >= max_len:
            return tokens[:max_len]
        return tokens + [self.vocab['<PAD>']] * (max_len - len(tokens))
    
    def reset(self, use_synonym: bool = False) -> Tuple[np.ndarray, List[int]]:
        """Reset environment with new configuration."""
        self.steps = 0
        self.objects = []
        
        # Place 3 objects with different colors
        positions = []
        for color in self.colors:
            shape = random.choice(self.shapes)
            while True:
                pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
                if pos not in positions:
                    positions.append(pos)
                    self.objects.append((color, shape, pos))
                    break
        
        # Agent starts randomly (not on objects)
        while True:
            self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if self.agent_pos not in positions:
                break
        
        # Generate instruction based on mode
        if self.mode == 'negation':
            self.instruction, self.target = self._generate_negation()
        elif self.mode == 'synonym':
            self.instruction, self.target = self._generate_synonym(use_synonym)
        elif self.mode == 'relation':
            self.instruction, self.target = self._generate_relation()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        tokens = self.tokenize(self.instruction)
        tokens = self.pad_tokens(tokens)
        
        obs = self._get_obs()
        return obs, tokens
    
    def _generate_negation(self) -> Tuple[str, Tuple[int, int]]:
        """Generate negation instruction: "go to the circle that is NOT blue" """
        # Pick a shape that appears in multiple colors
        shape = random.choice(self.shapes)
        
        # Find objects with this shape
        matching = [(c, s, p) for c, s, p in self.objects if s == shape]
        
        if len(matching) < 2:
            # Fallback to any object
            target_obj = random.choice(self.objects)
            color, shape, pos = target_obj
            instruction = f"go to the {color} {shape}"
            return instruction, pos
        
        # Pick a color to negate
        negated_color = random.choice([c for c, s, p in matching])
        
        # Target is any object with this shape but NOT this color
        valid_targets = [(c, s, p) for c, s, p in matching if c != negated_color]
        
        if not valid_targets:
            # Fallback
            target_obj = random.choice(self.objects)
            color, shape, pos = target_obj
            instruction = f"go to the {color} {shape}"
            return instruction, pos
        
        target = random.choice(valid_targets)
        instruction = f"go to the {shape} that is not {negated_color}"
        return instruction, target[2]
    
    def _generate_synonym(self, use_synonym: bool) -> Tuple[str, Tuple[int, int]]:
        """Generate synonym instruction: "go to the crimson circle" """
        target_obj = random.choice(self.objects)
        color, shape, pos = target_obj
        
        if use_synonym:
            # Use synonym for color
            syn_options = [s for s, c in self.synonyms.items() if c == color]
            if syn_options:
                color_word = random.choice(syn_options)
            else:
                color_word = color
        else:
            color_word = color
        
        instruction = f"go to the {color_word} {shape}"
        return instruction, pos
    
    def _generate_relation(self) -> Tuple[str, Tuple[int, int]]:
        """Generate relational instruction: "go to the circle near the square" """
        # Pick reference object
        ref_obj = random.choice(self.objects)
        ref_color, ref_shape, ref_pos = ref_obj
        
        # Find object nearest to reference (excluding reference)
        other_objs = [o for o in self.objects if o != ref_obj]
        
        def dist(o):
            return abs(o[2][0] - ref_pos[0]) + abs(o[2][1] - ref_pos[1])
        
        nearest = min(other_objs, key=dist)
        target_color, target_shape, target_pos = nearest
        
        instruction = f"go to the {target_shape} near the {ref_shape}"
        return instruction, target_pos
    
    def _get_obs(self) -> np.ndarray:
        """Get observation: agent pos + object positions encoded."""
        obs = np.zeros(self.size * self.size + 3, dtype=np.float32)
        
        # Agent position
        agent_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        obs[agent_idx] = 1.0
        
        # Object positions with color encoding
        for color, shape, pos in self.objects:
            obj_idx = pos[0] * self.size + pos[1]
            color_val = (self.colors.index(color) + 1) / len(self.colors)
            obs[obj_idx] = color_val
        
        # Relative position to closest object
        if self.objects:
            dists = [abs(self.agent_pos[0] - p[0]) + abs(self.agent_pos[1] - p[1]) 
                    for _, _, p in self.objects]
            obs[-3] = min(dists) / (2 * self.size)
        
        obs[-2] = self.steps / self.max_steps
        obs[-1] = 1.0 if self.mode == 'negation' else (0.5 if self.mode == 'synonym' else 0.0)
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action (0=up, 1=down, 2=left, 3=right)."""
        self.steps += 1
        reward = -0.02
        done = False
        info = {'success': False}
        
        # Movement
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        self.agent_pos = (new_x, new_y)
        
        # Check if reached target
        if self.agent_pos == self.target:
            reward = 1.0
            done = True
            info['success'] = True
        elif self.steps >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        return obs, reward, done, info


class VectorizedSemanticEnv:
    """Vectorized semantic environment."""
    
    def __init__(self, num_envs: int, size: int = 5, mode: str = 'negation'):
        self.num_envs = num_envs
        self.size = size
        self.mode = mode
        self.envs = [SemanticGridworld(size, mode) for _ in range(num_envs)]
        self.vocab_size = self.envs[0].vocab_size
        
    def reset(self, use_synonym: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_list, inst_list = [], []
        for env in self.envs:
            obs, tokens = env.reset(use_synonym=use_synonym)
            obs_list.append(obs)
            inst_list.append(tokens)
        
        return (torch.tensor(np.stack(obs_list), dtype=torch.float32),
                torch.tensor(np.array(inst_list), dtype=torch.long))
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
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


def evaluate_semantic(brain, env: VectorizedSemanticEnv, num_episodes: int = 200,
                     use_synonym: bool = False, device: str = 'cuda') -> Dict:
    """Evaluate semantic understanding."""
    brain.eval()
    successes = 0
    total = 0
    
    with torch.no_grad():
        episodes_done = 0
        while episodes_done < num_episodes:
            obs, instructions = env.reset(use_synonym=use_synonym)
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
    return {'accuracy': successes / max(total, 1) * 100, 'total': total, 'successes': successes}


if __name__ == '__main__':
    print("=== Testing Semantic Environment ===\n")
    
    for mode in ['negation', 'synonym', 'relation']:
        print(f"\n--- Mode: {mode} ---")
        env = SemanticGridworld(size=5, mode=mode)
        obs, tokens = env.reset()
        
        print(f"Instruction: {env.instruction}")
        print(f"Target: {env.target}")
        print(f"Objects: {env.objects}")
        print(f"Tokens: {tokens}")
