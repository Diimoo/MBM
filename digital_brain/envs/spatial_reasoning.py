"""
Spatial Reasoning Environment for World Model Comprehension

Tests whether MBM builds internal world models through spatial relationship understanding.
Agent must place objects according to spatial instructions like "put red_ball left of blue_square".

Key test: Train on 4 relations (left, right, above, below), test generalization to 'near'.
"""

import torch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional

from ..datatypes import Obs


class SpatialReasoningEnv:
    """Environment for spatial reasoning with object placement."""
    
    def __init__(self, size: int = 7):
        self.size = size
        self.objects = ['red_ball', 'blue_square', 'green_triangle']
        self.colors = {'red_ball': 1, 'blue_square': 2, 'green_triangle': 3}
        
        # Relations: train on 4, test on 'near'
        self.train_relations = ['left', 'right', 'above', 'below']
        self.test_relations = ['near']
        self.all_relations = self.train_relations + self.test_relations
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # State
        self.object_positions = {}  # object_name -> (x, y)
        self.agent_holding = None
        self.agent_pos = None
        self.target_instruction = None
        self.target_relation = None
        self.target_obj1 = None
        self.target_obj2 = None
        
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary for spatial instructions."""
        vocab = {
            '<PAD>': 0, '<START>': 1, '<END>': 2,
            'put': 3, 'the': 4,
            'red_ball': 5, 'blue_square': 6, 'green_triangle': 7,
            'left': 8, 'right': 9, 'above': 10, 'below': 11, 'near': 12,
            'of': 13,
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
        # Place objects randomly (not overlapping)
        positions = []
        for obj in self.objects:
            while True:
                pos = (random.randint(1, self.size-2), random.randint(1, self.size-2))
                if pos not in positions:
                    positions.append(pos)
                    self.object_positions[obj] = pos
                    break
        
        # Agent starts at random position
        while True:
            self.agent_pos = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            if self.agent_pos not in positions:
                break
        
        self.agent_holding = None
        
        # Generate instruction
        relations = self.test_relations if use_test_relations else self.train_relations
        self.target_relation = random.choice(relations)
        
        # Pick two different objects
        obj_list = list(self.objects)
        random.shuffle(obj_list)
        self.target_obj1 = obj_list[0]  # Object to move
        self.target_obj2 = obj_list[1]  # Reference object
        
        # Generate instruction: "put red_ball left of blue_square"
        self.target_instruction = f"put {self.target_obj1} {self.target_relation} of {self.target_obj2}"
        
        tokens = self.tokenize(self.target_instruction)
        tokens = self.pad_tokens(tokens)
        
        obs = self._get_obs()
        return obs, tokens
    
    def _get_obs(self) -> np.ndarray:
        """Get observation as grid with object encodings."""
        # Channel 0: agent position
        # Channel 1-3: object positions (by color ID)
        obs = np.zeros((4, self.size, self.size), dtype=np.float32)
        
        # Agent
        obs[0, self.agent_pos[0], self.agent_pos[1]] = 1.0
        
        # Objects
        for obj, pos in self.object_positions.items():
            color_id = self.colors[obj]
            obs[color_id, pos[0], pos[1]] = 1.0
        
        return obs
    
    def compute_valid_positions(self, relation: str, ref_pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Compute valid positions for given relation relative to reference."""
        rx, ry = ref_pos
        valid = []
        
        if relation == 'left':
            # All positions with x < ref_x
            for x in range(rx):
                for y in range(self.size):
                    valid.append((x, y))
        elif relation == 'right':
            # All positions with x > ref_x
            for x in range(rx + 1, self.size):
                for y in range(self.size):
                    valid.append((x, y))
        elif relation == 'above':
            # All positions with y < ref_y (assuming y increases downward)
            for x in range(self.size):
                for y in range(ry):
                    valid.append((x, y))
        elif relation == 'below':
            # All positions with y > ref_y
            for x in range(self.size):
                for y in range(ry + 1, self.size):
                    valid.append((x, y))
        elif relation == 'near':
            # Within distance 2 (Manhattan or Euclidean)
            for x in range(self.size):
                for y in range(self.size):
                    dist = abs(x - rx) + abs(y - ry)
                    if 0 < dist <= 2:  # Near but not on same position
                        valid.append((x, y))
        
        return valid
    
    def check_relation(self, pos1: Tuple[int, int], relation: str, pos2: Tuple[int, int]) -> bool:
        """Check if pos1 satisfies relation with respect to pos2."""
        x1, y1 = pos1
        x2, y2 = pos2
        
        if relation == 'left':
            return x1 < x2
        elif relation == 'right':
            return x1 > x2
        elif relation == 'above':
            return y1 < y2
        elif relation == 'below':
            return y1 > y2
        elif relation == 'near':
            dist = abs(x1 - x2) + abs(y1 - y2)
            return 0 < dist <= 2
        return False
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Execute action.
        Actions:
            0-3: Move (up, down, left, right)
            4: Pick up object at current position
            5: Place held object at current position
        """
        reward = -0.01  # Small step penalty
        done = False
        info = {'success': False}
        
        # Movement
        dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action] if action < 4 else (0, 0)
        
        if action < 4:
            new_x = max(0, min(self.size - 1, self.agent_pos[0] + dx))
            new_y = max(0, min(self.size - 1, self.agent_pos[1] + dy))
            self.agent_pos = (new_x, new_y)
        
        elif action == 4:  # Pick up
            for obj, pos in self.object_positions.items():
                if pos == self.agent_pos and self.agent_holding is None:
                    self.agent_holding = obj
                    del self.object_positions[obj]
                    break
        
        elif action == 5:  # Place
            if self.agent_holding is not None:
                # Place object
                self.object_positions[self.agent_holding] = self.agent_pos
                placed_obj = self.agent_holding
                self.agent_holding = None
                
                # Check if correct placement
                if placed_obj == self.target_obj1:
                    ref_pos = self.object_positions.get(self.target_obj2)
                    if ref_pos is not None:
                        if self.check_relation(self.agent_pos, self.target_relation, ref_pos):
                            reward = 10.0
                            done = True
                            info['success'] = True
        
        obs = self._get_obs()
        return obs, reward, done, info
    
    def get_instruction_tensor(self) -> torch.Tensor:
        """Get current instruction as tensor."""
        tokens = self.tokenize(self.target_instruction)
        tokens = self.pad_tokens(tokens)
        return torch.tensor(tokens, dtype=torch.long)


class VectorizedSpatialEnv:
    """Vectorized version for parallel training."""
    
    def __init__(self, num_envs: int, size: int = 7):
        self.num_envs = num_envs
        self.size = size
        self.envs = [SpatialReasoningEnv(size) for _ in range(num_envs)]
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
                obs, tokens = env.reset()
            
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
            tokens = env.tokenize(env.target_instruction)
            tokens = env.pad_tokens(tokens)
            inst_list.append(tokens)
        return torch.tensor(np.array(inst_list), dtype=torch.long)


def evaluate_spatial_reasoning(brain, env: VectorizedSpatialEnv, 
                               num_episodes: int = 100,
                               max_steps: int = 50,
                               use_test_relations: bool = False,
                               device: str = 'cuda') -> Dict[str, float]:
    """
    Evaluate spatial reasoning capability.
    
    Args:
        brain: DigitalBrain instance
        env: VectorizedSpatialEnv
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        use_test_relations: If True, use novel 'near' relation
        device: Device to run on
    
    Returns:
        Dictionary with accuracy and other metrics
    """
    brain.eval()
    successes = 0
    total = 0
    relation_stats = {}
    
    with torch.no_grad():
        for ep in range(num_episodes // env.num_envs + 1):
            obs, instructions = env.reset(use_test_relations=use_test_relations)
            obs = obs.to(device)
            instructions = instructions.to(device)
            
            brain.reset(env.num_envs)
            
            # Initialize reward/done for first step
            prev_reward = torch.zeros(env.num_envs, device=device)
            prev_done = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
            
            episode_done = [False] * env.num_envs
            
            for step in range(max_steps):
                # Flatten and wrap observation
                obs_flat = obs.view(obs.size(0), -1)
                obs_wrapped = Obs(x=obs_flat)
                
                # Get action from brain - returns tuple (action, log_prob, value, state, log, entropy)
                action, log_prob, value, state, log, entropy = brain.act(
                    obs_wrapped, prev_reward, prev_done, instruction=instructions
                )
                
                # Step environment
                obs, rewards, dones, infos = env.step(action)
                obs = obs.to(device)
                prev_reward = rewards.to(device)
                prev_done = dones.to(device)
                
                # Track successes
                for i, (done, info) in enumerate(zip(dones, infos)):
                    if done and not episode_done[i]:
                        episode_done[i] = True
                        total += 1
                        if info.get('success', False):
                            successes += 1
                            # Track per-relation stats
                            rel = env.envs[i].target_relation
                            if rel not in relation_stats:
                                relation_stats[rel] = {'success': 0, 'total': 0}
                            relation_stats[rel]['success'] += 1
                            relation_stats[rel]['total'] += 1
                        else:
                            rel = env.envs[i].target_relation
                            if rel not in relation_stats:
                                relation_stats[rel] = {'success': 0, 'total': 0}
                            relation_stats[rel]['total'] += 1
                
                if all(episode_done):
                    break
            
            if total >= num_episodes:
                break
    
    brain.train()
    
    # Compute per-relation accuracy
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
    # Test environment
    env = SpatialReasoningEnv(size=7)
    obs, tokens = env.reset()
    
    print(f"Instruction: {env.target_instruction}")
    print(f"Tokens: {tokens}")
    print(f"Observation shape: {obs.shape}")
    print(f"Object positions: {env.object_positions}")
    print(f"Target: place {env.target_obj1} {env.target_relation} of {env.target_obj2}")
    
    # Test vectorized
    vec_env = VectorizedSpatialEnv(num_envs=4, size=7)
    obs, instructions = vec_env.reset()
    print(f"\nVectorized obs shape: {obs.shape}")
    print(f"Instructions shape: {instructions.shape}")
    
    # Test valid positions
    ref_pos = (3, 3)
    for rel in env.all_relations:
        valid = env.compute_valid_positions(rel, ref_pos)
        print(f"\n{rel} of {ref_pos}: {len(valid)} valid positions")
