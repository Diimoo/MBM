"""
Simplified Spatial Navigation Environment

Agent must navigate to a position satisfying a spatial relation with a reference object.
No pick/place - just navigation to test spatial understanding.

Example: "go left of blue_square" - agent must reach any position left of the blue square.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

from ..datatypes import Obs


class SpatialNavEnv:
    """Simple spatial navigation - go to position satisfying spatial relation."""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.objects = ['red_circle', 'blue_square', 'green_triangle']
        self.colors = {'red': 1, 'blue': 2, 'green': 3}
        self.shapes = {'circle': 1, 'square': 2, 'triangle': 3}
        
        # Relations: train on 4, test on 'near'
        self.train_relations = ['left', 'right', 'above', 'below']
        self.test_relations = ['near']
        self.all_relations = self.train_relations + self.test_relations
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        
        # State
        self.agent_pos = None
        self.reference_pos = None
        self.reference_obj = None
        self.target_relation = None
        self.steps = 0
        self.max_steps = 30
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for spatial instructions."""
        vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'go': 3, 'to': 4, 'the': 5,
            'left': 6, 'right': 7, 'above': 8, 'below': 9, 'near': 10,
            'of': 11,
            'red_circle': 12, 'blue_square': 13, 'green_triangle': 14,
        }
        return vocab
    
    def tokenize(self, instruction: str) -> List[int]:
        """Convert instruction string to token IDs."""
        tokens = [self.vocab['<START>']]
        for word in instruction.lower().split():
            if word in self.vocab:
                tokens.append(self.vocab[word])
        tokens.append(self.vocab['<END>'])
        return tokens
    
    def pad_tokens(self, tokens: List[int], max_len: int = 10) -> List[int]:
        """Pad token sequence to fixed length."""
        if len(tokens) >= max_len:
            return tokens[:max_len]
        return tokens + [self.vocab['<PAD>']] * (max_len - len(tokens))
    
    def reset(self, use_test_relations: bool = False) -> Tuple[np.ndarray, List[int]]:
        """Reset environment with new random configuration."""
        self.steps = 0
        
        # Place reference object randomly (not at edges for spatial relations to work)
        self.reference_pos = (
            random.randint(1, self.size - 2),
            random.randint(1, self.size - 2)
        )
        self.reference_obj = random.choice(self.objects)
        
        # Agent starts at random position (not on reference)
        while True:
            self.agent_pos = (
                random.randint(0, self.size - 1),
                random.randint(0, self.size - 1)
            )
            if self.agent_pos != self.reference_pos:
                break
        
        # Generate instruction
        relations = self.test_relations if use_test_relations else self.train_relations
        self.target_relation = random.choice(relations)
        
        # "go left of blue_square"
        instruction = f"go {self.target_relation} of {self.reference_obj}"
        
        tokens = self.tokenize(instruction)
        tokens = self.pad_tokens(tokens)
        
        obs = self._get_obs()
        return obs, tokens
    
    def _get_obs(self) -> np.ndarray:
        """Get observation as flattened local view + global reference position."""
        # Simple observation: agent pos (2) + reference pos (2) + one-hot relation (5)
        obs = np.zeros(9, dtype=np.float32)
        
        # Normalized positions
        obs[0] = self.agent_pos[0] / self.size
        obs[1] = self.agent_pos[1] / self.size
        obs[2] = self.reference_pos[0] / self.size
        obs[3] = self.reference_pos[1] / self.size
        
        # Relative position (normalized)
        obs[4] = (self.agent_pos[0] - self.reference_pos[0]) / self.size
        obs[5] = (self.agent_pos[1] - self.reference_pos[1]) / self.size
        
        # Distance (normalized)
        dist = abs(self.agent_pos[0] - self.reference_pos[0]) + abs(self.agent_pos[1] - self.reference_pos[1])
        obs[6] = dist / (2 * self.size)
        
        # Object encoding
        obj_idx = self.objects.index(self.reference_obj)
        obs[7] = obj_idx / len(self.objects)
        
        # Relation encoding
        rel_idx = self.all_relations.index(self.target_relation) if self.target_relation in self.all_relations else 0
        obs[8] = rel_idx / len(self.all_relations)
        
        return obs
    
    def check_relation(self, agent_pos: Tuple[int, int], relation: str, ref_pos: Tuple[int, int]) -> bool:
        """Check if agent position satisfies relation with reference."""
        ax, ay = agent_pos
        rx, ry = ref_pos
        
        if relation == 'left':
            return ax < rx
        elif relation == 'right':
            return ax > rx
        elif relation == 'above':
            return ay < ry  # y=0 is top
        elif relation == 'below':
            return ay > ry
        elif relation == 'near':
            dist = abs(ax - rx) + abs(ay - ry)
            return 0 < dist <= 2
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action.
        Actions: 0=up, 1=down, 2=left, 3=right
        """
        self.steps += 1
        reward = -0.02  # Small step penalty
        done = False
        info = {'success': False}
        
        # Movement
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
        new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
        
        # Can't move onto reference object
        if (new_x, new_y) != self.reference_pos:
            self.agent_pos = (new_x, new_y)
        
        # Check if agent satisfies spatial relation
        if self.check_relation(self.agent_pos, self.target_relation, self.reference_pos):
            reward = 1.0
            done = True
            info['success'] = True
        elif self.steps >= self.max_steps:
            done = True
        
        obs = self._get_obs()
        return obs, reward, done, info


class VectorizedSpatialNavEnv:
    """Vectorized version for parallel training."""
    
    def __init__(self, num_envs: int, size: int = 5):
        self.num_envs = num_envs
        self.size = size
        self.envs = [SpatialNavEnv(size) for _ in range(num_envs)]
        self.vocab_size = self.envs[0].vocab_size
        
    def reset(self, use_test_relations: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reset all environments."""
        obs_list = []
        inst_list = []
        
        for env in self.envs:
            obs, tokens = env.reset(use_test_relations=use_test_relations)
            obs_list.append(obs)
            inst_list.append(tokens)
        
        obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        instructions = torch.tensor(np.array(inst_list), dtype=torch.long)
        
        return obs, instructions
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[dict]]:
        """Step all environments."""
        obs_list = []
        rewards = []
        dones = []
        infos = []
        
        actions_np = actions.cpu().numpy()
        
        for i, env in enumerate(self.envs):
            obs, reward, done, info = env.step(int(actions_np[i]))
            
            if done:
                obs, _ = env.reset()
            
            obs_list.append(obs)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
        
        obs = torch.tensor(np.stack(obs_list), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        return obs, rewards, dones, infos
    
    def get_instructions(self) -> torch.Tensor:
        """Get all current instructions."""
        inst_list = []
        for env in self.envs:
            instruction = f"go {env.target_relation} of {env.reference_obj}"
            tokens = env.tokenize(instruction)
            tokens = env.pad_tokens(tokens)
            inst_list.append(tokens)
        return torch.tensor(np.array(inst_list), dtype=torch.long)


def evaluate_spatial_nav(brain, env: VectorizedSpatialNavEnv, 
                        num_episodes: int = 200,
                        max_steps: int = 30,
                        use_test_relations: bool = False,
                        device: str = 'cuda') -> Dict[str, float]:
    """Evaluate spatial navigation capability."""
    brain.eval()
    successes = 0
    total = 0
    relation_stats = {}
    
    with torch.no_grad():
        episodes_done = 0
        while episodes_done < num_episodes:
            obs, instructions = env.reset(use_test_relations=use_test_relations)
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
                prev_reward = rewards.to(device)
                prev_done = dones.to(device)
                
                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done and not episode_done[i]:
                        episode_done[i] = True
                        episodes_done += 1
                        total += 1
                        
                        rel = env.envs[i].target_relation
                        if rel not in relation_stats:
                            relation_stats[rel] = {'success': 0, 'total': 0}
                        relation_stats[rel]['total'] += 1
                        
                        if info.get('success', False):
                            successes += 1
                            relation_stats[rel]['success'] += 1
                
                if all(episode_done) or episodes_done >= num_episodes:
                    break
    
    brain.train()
    
    relation_acc = {}
    for rel, stats in relation_stats.items():
        if stats['total'] > 0:
            relation_acc[rel] = stats['success'] / stats['total'] * 100
    
    return {
        'accuracy': successes / max(total, 1) * 100,
        'total_episodes': total,
        'successes': successes,
        'relation_accuracy': relation_acc
    }


if __name__ == '__main__':
    env = SpatialNavEnv(size=5)
    obs, tokens = env.reset()
    
    print(f"Instruction: go {env.target_relation} of {env.reference_obj}")
    print(f"Agent: {env.agent_pos}, Reference: {env.reference_pos}")
    print(f"Observation: {obs}")
    print(f"Tokens: {tokens}")
    
    # Test valid positions
    for rel in env.all_relations:
        count = 0
        for x in range(env.size):
            for y in range(env.size):
                if env.check_relation((x, y), rel, env.reference_pos):
                    count += 1
        print(f"{rel}: {count} valid positions")
