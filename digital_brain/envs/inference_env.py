"""
Inference from Description Environment

Agent builds world model purely from language, then acts on it.
Key test: Agent cannot succeed through vision alone - must use language.

Phase 1: Language descriptions tell agent where objects are
Phase 2: Vision shows unlabeled objects, agent must navigate using language-built model
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

from ..datatypes import Obs


class InferenceEnv:
    """
    Two-phase environment:
    1. Description phase: Agent receives language descriptions (no vision)
    2. Action phase: Agent sees unlabeled objects, must use language model to navigate
    """
    
    def __init__(self, size: int = 5):
        self.size = size
        self.objects = ['key', 'door', 'gem', 'box']
        self.colors = ['red', 'blue', 'green']
        
        # Vocabulary
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # State
        self.agent_pos = None
        self.object_positions = {}  # (color, object) -> (x, y)
        self.target_object = None
        self.target_color = None
        self.descriptions = []
        self.phase = 'description'  # or 'action'
        self.description_step = 0
        self.steps = 0
        self.max_steps = 40
        
    def _build_vocab(self) -> Dict[str, int]:
        vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'there': 3, 'is': 4, 'a': 5, 'at': 6, 'position': 7,
            'go': 8, 'to': 9, 'the': 10,
            'red': 11, 'blue': 12, 'green': 13,
            'key': 14, 'door': 15, 'gem': 16, 'box': 17,
            ',': 18, '0': 19, '1': 20, '2': 21, '3': 22, '4': 23,
        }
        return vocab
    
    def tokenize(self, text: str) -> List[int]:
        tokens = [self.vocab['<START>']]
        for word in text.lower().replace(',', ' ,').split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens.append(self.vocab['<END>'])
        return tokens
    
    def pad_tokens(self, tokens: List[int], max_len: int = 15) -> List[int]:
        if len(tokens) >= max_len:
            return tokens[:max_len]
        return tokens + [self.vocab['<PAD>']] * (max_len - len(tokens))
    
    def reset(self) -> Tuple[np.ndarray, List[int]]:
        """Reset environment with new configuration."""
        self.steps = 0
        self.phase = 'description'
        self.description_step = 0
        self.object_positions = {}
        self.descriptions = []
        
        # Place 2-3 objects randomly
        num_objects = random.randint(2, 3)
        positions_used = []
        
        for _ in range(num_objects):
            color = random.choice(self.colors)
            obj = random.choice(self.objects)
            
            while True:
                pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
                if pos not in positions_used:
                    positions_used.append(pos)
                    self.object_positions[(color, obj)] = pos
                    # "there is a red key at position 2,3"
                    desc = f"there is a {color} {obj} at position {pos[0]},{pos[1]}"
                    self.descriptions.append(desc)
                    break
        
        # Choose target (one of placed objects)
        self.target_color, self.target_object = random.choice(list(self.object_positions.keys()))
        
        # Agent starts randomly (not on objects)
        while True:
            self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if self.agent_pos not in positions_used:
                break
        
        # Start with first description
        obs = self._get_obs()
        tokens = self.tokenize(self.descriptions[0])
        tokens = self.pad_tokens(tokens)
        
        return obs, tokens
    
    def _get_obs(self) -> np.ndarray:
        """
        Observation depends on phase:
        - Description phase: Only agent position (no object info in vision)
        - Action phase: Agent + unlabeled object positions
        """
        obs = np.zeros(self.size * self.size + 5, dtype=np.float32)
        
        # Agent position (always visible)
        agent_idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        obs[agent_idx] = 1.0
        
        if self.phase == 'action':
            # Show object positions but NOT colors/types (unlabeled)
            for (color, obj), pos in self.object_positions.items():
                obj_idx = pos[0] * self.size + pos[1]
                obs[obj_idx] = 0.5  # Different value to distinguish from agent
        
        # Phase indicator
        obs[-5] = 1.0 if self.phase == 'description' else 0.0
        
        # Target info (encoded)
        color_idx = self.colors.index(self.target_color)
        obj_idx = self.objects.index(self.target_object)
        obs[-4] = color_idx / len(self.colors)
        obs[-3] = obj_idx / len(self.objects)
        
        # Progress
        obs[-2] = self.description_step / max(len(self.descriptions), 1)
        obs[-1] = self.steps / self.max_steps
        
        return obs
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Actions in description phase: 0 = process description (advance)
        Actions in action phase: 0-3 = movement (up, down, left, right)
        """
        self.steps += 1
        reward = -0.02
        done = False
        info = {'success': False, 'phase': self.phase}
        
        if self.phase == 'description':
            # Action 0 = acknowledge description, move to next
            self.description_step += 1
            
            if self.description_step >= len(self.descriptions):
                # Switch to action phase
                self.phase = 'action'
                # Give instruction to go to target
                target_desc = f"go to the {self.target_color} {self.target_object}"
                tokens = self.tokenize(target_desc)
                tokens = self.pad_tokens(tokens)
                info['instruction'] = tokens
            else:
                # Next description
                tokens = self.tokenize(self.descriptions[self.description_step])
                tokens = self.pad_tokens(tokens)
                info['instruction'] = tokens
        
        else:  # action phase
            # Movement
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action % 4]
            new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
            new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
            self.agent_pos = (new_x, new_y)
            
            # Check if reached target
            target_pos = self.object_positions.get((self.target_color, self.target_object))
            if self.agent_pos == target_pos:
                reward = 1.0
                done = True
                info['success'] = True
        
        if self.steps >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        return obs, reward, done, info


class VectorizedInferenceEnv:
    """Vectorized version for parallel training."""
    
    def __init__(self, num_envs: int, size: int = 5):
        self.num_envs = num_envs
        self.size = size
        self.envs = [InferenceEnv(size) for _ in range(num_envs)]
        self.vocab_size = self.envs[0].vocab_size
        self.current_instructions = None
        
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_list = []
        inst_list = []
        
        for env in self.envs:
            obs, tokens = env.reset()
            obs_list.append(obs)
            inst_list.append(tokens)
        
        obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        self.current_instructions = torch.tensor(np.array(inst_list), dtype=torch.long)
        
        return obs, self.current_instructions
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        obs_list = []
        rewards = []
        dones = []
        infos = []
        new_instructions = []
        
        actions_np = actions.cpu().numpy()
        
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(actions_np[i]))
            
            if done:
                obs, tokens = env.reset()
                new_instructions.append(tokens)
            elif 'instruction' in info:
                new_instructions.append(info['instruction'])
            else:
                # Keep current instruction
                new_instructions.append(self.current_instructions[i].tolist())
            
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        self.current_instructions = torch.tensor(np.array(new_instructions), dtype=torch.long)
        
        return obs, rewards, dones, infos
    
    def get_instructions(self) -> torch.Tensor:
        return self.current_instructions


def evaluate_inference(brain, env: VectorizedInferenceEnv,
                      num_episodes: int = 100,
                      max_steps: int = 40,
                      device: str = 'cuda') -> Dict[str, float]:
    """Evaluate inference from description capability."""
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
            
            for step in range(max_steps):
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
    
    return {
        'accuracy': successes / max(total, 1) * 100,
        'total': total,
        'successes': successes
    }


if __name__ == '__main__':
    env = InferenceEnv(size=5)
    obs, tokens = env.reset()
    
    print("=== Inference Environment Test ===")
    print(f"Objects: {env.object_positions}")
    print(f"Target: {env.target_color} {env.target_object}")
    print(f"Descriptions:")
    for d in env.descriptions:
        print(f"  {d}")
    print(f"\nPhase: {env.phase}")
    print(f"Observation shape: {obs.shape}")
    
    # Simulate description phase
    print("\n--- Description Phase ---")
    for i in range(len(env.descriptions)):
        obs, reward, done, info = env.step(0)
        print(f"Step {i+1}: phase={env.phase}")
        if 'instruction' in info:
            print(f"  New instruction received")
    
    print(f"\n--- Action Phase ---")
    print(f"Target position: {env.object_positions.get((env.target_color, env.target_object))}")
