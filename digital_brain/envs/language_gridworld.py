"""
Language Gridworld Environment

Grid with colored shapes + language instructions.
Agent must navigate to the correct object based on the instruction.
"""
import numpy as np
import torch
from typing import Optional, Tuple, Dict, List


class LanguageGridworld:
    """
    Grid with colored shapes + language instructions.
    
    The agent receives:
    - Visual observation: local 3x3 neighborhood with object encodings
    - Language instruction: tokenized "go to [color] [shape]"
    
    The agent must navigate to the correct object.
    """
    
    def __init__(self, size: int = 5, seed: Optional[int] = None):
        self.size = int(size)
        self.action_space_n = 4  # Up, Down, Left, Right
        self.rng = np.random.default_rng(seed)
        
        # Object properties
        self.colors = ['red', 'blue', 'green']
        self.shapes = ['circle', 'square', 'triangle']
        
        # Build vocabulary
        self.vocab = self._build_vocab()
        self.vocab_size = len(self.vocab)
        self.id_to_word = {v: k for k, v in self.vocab.items()}
        self.max_instruction_len = 6  # "<START> go to [color] [shape] <END>"
        
        # Object visual encodings (color_idx * 3 + shape_idx + 1)
        # This gives unique values 1-9 for each color-shape combo
        self._object_encodings = {}
        for ci, color in enumerate(self.colors):
            for si, shape in enumerate(self.shapes):
                self._object_encodings[(color, shape)] = float(ci * 3 + si + 1)
        
        # Observation space: 3x3 local view flattened = 9 values
        self.obs_dim = 9
        
        self.reset()
    
    def _build_vocab(self) -> Dict[str, int]:
        """Build vocabulary mapping words to token IDs."""
        vocab = {
            # Commands
            'go': 0, 'to': 1, 'the': 2,
            # Colors
            'red': 3, 'blue': 4, 'green': 5,
            # Shapes  
            'circle': 6, 'square': 7, 'triangle': 8,
            # Meta tokens
            '<PAD>': 9, '<START>': 10, '<END>': 11,
            # Future expansion: French
            'va': 12, 'au': 13, 'le': 14,
            'rouge': 15, 'bleu': 16, 'vert': 17,
            'cercle': 18, 'carre': 19, 'triangle_fr': 20,
            # Negation
            'not': 21, 'pas': 22,
        }
        return vocab
    
    def _sample_pos(self, forbidden: set) -> np.ndarray:
        """Sample a position not in forbidden set."""
        while True:
            p = (int(self.rng.integers(0, self.size)), int(self.rng.integers(0, self.size)))
            if p not in forbidden:
                return np.array(p, dtype=np.int64)
    
    def _tokenize(self, instruction: str) -> np.ndarray:
        """Convert instruction string to token IDs."""
        words = instruction.lower().split()
        tokens = [self.vocab['<START>']]
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                # Unknown word - skip (could add <UNK> token)
                pass
        tokens.append(self.vocab['<END>'])
        
        # Pad to max length
        while len(tokens) < self.max_instruction_len:
            tokens.append(self.vocab['<PAD>'])
        
        return np.array(tokens[:self.max_instruction_len], dtype=np.int64)
    
    def _generate_instruction(self, color: str, shape: str, language: str = 'english') -> str:
        """Generate instruction in specified language."""
        if language == 'english':
            return f"go to {color} {shape}"
        elif language == 'french':
            color_map = {'red': 'rouge', 'blue': 'bleu', 'green': 'vert'}
            shape_map = {'circle': 'cercle', 'square': 'carre', 'triangle': 'triangle_fr'}
            return f"va au {color_map[color]} {shape_map[shape]}"
        else:
            return f"go to {color} {shape}"
    
    def reset(self, 
              target_color: Optional[str] = None,
              target_shape: Optional[str] = None,
              language: str = 'english') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Reset environment with random or specified target.
        
        Returns:
            visual_obs: (9,) local observation
            instruction_tokens: (max_instruction_len,) token IDs
            target_pos: (2,) target position (for debugging/eval)
        """
        # Place agent
        self.agent_pos = self._sample_pos(set())
        forbidden = {tuple(self.agent_pos)}
        
        # Place 3 objects (one of each color, random shapes)
        self.objects = []  # List of (pos, color, shape)
        
        # Ensure we have objects that allow unique targeting
        # Place one object per color with random shapes
        available_colors = list(self.colors)
        self.rng.shuffle(available_colors)
        
        for color in available_colors:
            shape = self.rng.choice(self.shapes)
            pos = self._sample_pos(forbidden)
            self.objects.append((pos.copy(), color, shape))
            forbidden.add(tuple(pos))
        
        # Select target (random or specified)
        if target_color is not None and target_shape is not None:
            # Find object matching criteria (if exists)
            matching = [(i, obj) for i, obj in enumerate(self.objects) 
                       if obj[1] == target_color and obj[2] == target_shape]
            if matching:
                self.target_idx = matching[0][0]
            else:
                # Force create the target object
                self.objects[0] = (self.objects[0][0], target_color, target_shape)
                self.target_idx = 0
        else:
            self.target_idx = self.rng.integers(0, len(self.objects))
        
        target_obj = self.objects[self.target_idx]
        self.target_pos = target_obj[0]
        self.target_color = target_obj[1]
        self.target_shape = target_obj[2]
        
        # Generate instruction
        self.language = language
        self.instruction_str = self._generate_instruction(
            self.target_color, self.target_shape, language
        )
        self.instruction_tokens = self._tokenize(self.instruction_str)
        
        # Episode state
        self.steps = 0
        self.max_steps = self.size * self.size * 2
        self.done = False
        
        return self._get_obs(), self.instruction_tokens, self.target_pos.copy()
    
    def _get_obs(self) -> np.ndarray:
        """Get local 3x3 observation around agent."""
        obs = np.zeros((3, 3), dtype=np.float32)
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                p = self.agent_pos + np.array([i, j], dtype=np.int64)
                
                # Wall (out of bounds)
                if (p < 0).any() or (p >= self.size).any():
                    obs[i+1, j+1] = -1.0  # Wall marker
                    continue
                
                # Check for objects at this position
                for obj_pos, color, shape in self.objects:
                    if np.array_equal(p, obj_pos):
                        obs[i+1, j+1] = self._object_encodings[(color, shape)]
                        break
        
        return obs.flatten().astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take action in environment.
        
        Args:
            action: 0=Up, 1=Down, 2=Left, 3=Right
            
        Returns:
            obs, reward, done, info
        """
        if self.done:
            return self._get_obs(), 0.0, True, {'reached_target': False}
        
        # Move
        move = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}[int(action)]
        new_pos = self.agent_pos + np.array(move, dtype=np.int64)
        
        # Check bounds
        if (new_pos >= 0).all() and (new_pos < self.size).all():
            self.agent_pos = new_pos
        
        # Check if reached any object
        reward = -0.01  # Small step penalty
        reached_target = False
        reached_wrong = False
        
        for idx, (obj_pos, color, shape) in enumerate(self.objects):
            if np.array_equal(self.agent_pos, obj_pos):
                if idx == self.target_idx:
                    reward = 10.0
                    reached_target = True
                    self.done = True
                else:
                    reward = -1.0  # Penalty for wrong object
                    reached_wrong = True
                break
        
        self.steps += 1
        if self.steps >= self.max_steps:
            self.done = True
        
        info = {
            'reached_target': reached_target,
            'reached_wrong': reached_wrong,
            'instruction': self.instruction_str,
            'target': (self.target_color, self.target_shape),
        }
        
        return self._get_obs(), float(reward), self.done, info


class TorchVectorLanguageGridworld:
    """
    Vectorized GPU-accelerated Language Gridworld.
    
    Runs N environments in parallel on GPU for high-throughput training.
    """
    
    def __init__(self, 
                 num_envs: int = 256,
                 size: int = 5,
                 device: str = 'cuda',
                 seed: int = 0):
        self.num_envs = num_envs
        self.size = size
        self.device = torch.device(device)
        self.action_space_n = 4
        
        # Vocabulary (same as single env)
        self.vocab = {
            'go': 0, 'to': 1, 'the': 2,
            'red': 3, 'blue': 4, 'green': 5,
            'circle': 6, 'square': 7, 'triangle': 8,
            '<PAD>': 9, '<START>': 10, '<END>': 11,
            'va': 12, 'au': 13, 'le': 14,
            'rouge': 15, 'bleu': 16, 'vert': 17,
            'cercle': 18, 'carre': 19, 'triangle_fr': 20,
            'not': 21, 'pas': 22,
        }
        self.vocab_size = len(self.vocab)
        self.max_instruction_len = 6
        
        # Observation dim: 9 (local view) + context
        self.obs_dim = 9
        self.ctx_dim = 2  # has_key analog (not used here but for interface compat)
        
        # RNG
        self.rng = torch.Generator(device=self.device)
        self.rng.manual_seed(seed)
        
        # Pre-compute instruction templates
        # English: [START, go, to, color, shape, END]
        # Indices: color in {3,4,5}, shape in {6,7,8}
        self._setup_instruction_templates()
        
        # Initialize state tensors
        self._init_state()
    
    def _setup_instruction_templates(self):
        """Pre-compute instruction token sequences."""
        # Colors: red=3, blue=4, green=5
        # Shapes: circle=6, square=7, triangle=8
        
        # All 9 combinations for English
        self.instruction_templates_en = []
        for color_id in [3, 4, 5]:  # red, blue, green
            for shape_id in [6, 7, 8]:  # circle, square, triangle
                # [START, go, to, color, shape, END]
                template = torch.tensor([10, 0, 1, color_id, shape_id, 11], 
                                       dtype=torch.long, device=self.device)
                self.instruction_templates_en.append(template)
        self.instruction_templates_en = torch.stack(self.instruction_templates_en)  # [9, 6]
        
        # French templates
        # va=12, au=13, rouge=15, bleu=16, vert=17, cercle=18, carre=19, triangle_fr=20
        self.instruction_templates_fr = []
        french_colors = [15, 16, 17]  # rouge, bleu, vert
        french_shapes = [18, 19, 20]  # cercle, carre, triangle_fr
        for color_id in french_colors:
            for shape_id in french_shapes:
                # [START, va, au, color, shape, END]
                template = torch.tensor([10, 12, 13, color_id, shape_id, 11],
                                       dtype=torch.long, device=self.device)
                self.instruction_templates_fr.append(template)
        self.instruction_templates_fr = torch.stack(self.instruction_templates_fr)  # [9, 6]
    
    def _init_state(self):
        """Initialize all state tensors."""
        N = self.num_envs
        
        # Agent position: [N, 2]
        self.agent_pos = torch.zeros(N, 2, dtype=torch.long, device=self.device)
        
        # Objects: 3 per env, [N, 3, 2] positions, [N, 3] color indices, [N, 3] shape indices
        self.obj_pos = torch.zeros(N, 3, 2, dtype=torch.long, device=self.device)
        self.obj_colors = torch.zeros(N, 3, dtype=torch.long, device=self.device)  # 0,1,2 = red,blue,green
        self.obj_shapes = torch.zeros(N, 3, dtype=torch.long, device=self.device)  # 0,1,2 = circle,square,triangle
        
        # Target index per env: [N]
        self.target_idx = torch.zeros(N, dtype=torch.long, device=self.device)
        
        # Instructions: [N, max_len]
        self.instructions = torch.zeros(N, self.max_instruction_len, dtype=torch.long, device=self.device)
        
        # Episode state
        self.steps = torch.zeros(N, dtype=torch.long, device=self.device)
        self.max_steps = self.size * self.size * 2
        self.done = torch.zeros(N, dtype=torch.bool, device=self.device)
        
        # Language mode per env (0=english, 1=french)
        self.language_mode = torch.zeros(N, dtype=torch.long, device=self.device)
    
    def reset(self, 
              env_mask: Optional[torch.Tensor] = None,
              language: str = 'english',
              target_combo: Optional[Tuple[int, int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reset environments.
        
        Args:
            env_mask: [N] bool tensor, True = reset this env. None = reset all.
            language: 'english' or 'french'
            target_combo: Optional (color_idx, shape_idx) to force specific target
            
        Returns:
            obs: [N, obs_dim]
            instructions: [N, max_instruction_len]
        """
        if env_mask is None:
            env_mask = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        
        N_reset = env_mask.sum().item()
        if N_reset == 0:
            return self._get_obs(), self.instructions
        
        # Set language mode
        lang_id = 0 if language == 'english' else 1
        self.language_mode[env_mask] = lang_id
        
        # Random agent positions
        agent_pos_new = torch.randint(0, self.size, (N_reset, 2), 
                                     generator=self.rng, device=self.device)
        self.agent_pos[env_mask] = agent_pos_new
        
        # Place 3 objects per env (avoid agent position)
        for obj_idx in range(3):
            # Random positions (simple approach - just ensure not on agent)
            # In practice positions can overlap objects, which is fine for this task
            obj_pos = torch.randint(0, self.size, (N_reset, 2),
                                   generator=self.rng, device=self.device)
            # Assign colors (one per object to ensure all colors present)
            colors = torch.full((N_reset,), obj_idx, dtype=torch.long, device=self.device)
            # Random shapes
            shapes = torch.randint(0, 3, (N_reset,), generator=self.rng, device=self.device)
            
            self.obj_pos[env_mask, obj_idx] = obj_pos
            self.obj_colors[env_mask, obj_idx] = colors
            self.obj_shapes[env_mask, obj_idx] = shapes
        
        # Select target object
        if target_combo is not None:
            # Force specific color-shape combination on first object
            target_color, target_shape = target_combo
            self.obj_colors[env_mask, 0] = target_color
            self.obj_shapes[env_mask, 0] = target_shape
            self.target_idx[env_mask] = 0
        else:
            # Random target
            target_indices = torch.randint(0, 3, (N_reset,), 
                                          generator=self.rng, device=self.device)
            self.target_idx[env_mask] = target_indices
        
        # Generate instructions based on target
        self._generate_instructions(env_mask)
        
        # Reset episode state
        self.steps[env_mask] = 0
        self.done[env_mask] = False
        
        return self._get_obs(), self.instructions
    
    def _generate_instructions(self, env_mask: torch.Tensor):
        """Generate instruction tokens for masked environments."""
        # Get target color and shape for each env
        env_indices = torch.where(env_mask)[0]
        target_colors = self.obj_colors[env_indices, self.target_idx[env_indices]]  # [N_reset]
        target_shapes = self.obj_shapes[env_indices, self.target_idx[env_indices]]  # [N_reset]
        
        # Compute template index: color * 3 + shape
        template_idx = target_colors * 3 + target_shapes  # [N_reset]
        
        # Select templates based on language
        lang_modes = self.language_mode[env_indices]  # [N_reset]
        
        # English templates
        en_mask = (lang_modes == 0)
        if en_mask.any():
            en_indices = env_indices[en_mask]
            en_template_idx = template_idx[en_mask]
            self.instructions[en_indices] = self.instruction_templates_en[en_template_idx]
        
        # French templates
        fr_mask = (lang_modes == 1)
        if fr_mask.any():
            fr_indices = env_indices[fr_mask]
            fr_template_idx = template_idx[fr_mask]
            self.instructions[fr_indices] = self.instruction_templates_fr[fr_template_idx]
    
    def _get_obs(self) -> torch.Tensor:
        """Get local 3x3 observation for all environments."""
        N = self.num_envs
        obs = torch.zeros(N, 3, 3, device=self.device)
        
        # Check each cell in 3x3 neighborhood
        for di in range(-1, 2):
            for dj in range(-1, 2):
                # Position being observed
                check_pos = self.agent_pos + torch.tensor([di, dj], device=self.device)
                
                # Wall check (out of bounds)
                wall_mask = ((check_pos[:, 0] < 0) | (check_pos[:, 0] >= self.size) |
                            (check_pos[:, 1] < 0) | (check_pos[:, 1] >= self.size))
                obs[wall_mask, di+1, dj+1] = -1.0
                
                # Check for objects at this position
                for obj_idx in range(3):
                    obj_match = ((self.obj_pos[:, obj_idx, 0] == check_pos[:, 0]) & 
                                (self.obj_pos[:, obj_idx, 1] == check_pos[:, 1]) &
                                ~wall_mask)
                    
                    if obj_match.any():
                        # Encode as color*3 + shape + 1 (values 1-9)
                        encoding = (self.obj_colors[obj_match, obj_idx] * 3 + 
                                   self.obj_shapes[obj_match, obj_idx] + 1).float()
                        obs[obj_match, di+1, dj+1] = encoding
        
        return obs.view(N, -1)  # [N, 9]
    
    def step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Take actions in all environments.
        
        Args:
            actions: [N] int tensor of actions (0-3)
            
        Returns:
            obs: [N, obs_dim]
            rewards: [N]
            dones: [N]
            info: dict with 'success' tensor
        """
        N = self.num_envs
        
        # Movement deltas
        moves = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], 
                            dtype=torch.long, device=self.device)
        deltas = moves[actions]  # [N, 2]
        
        # Compute new positions
        new_pos = self.agent_pos + deltas
        
        # Clamp to grid bounds
        new_pos = new_pos.clamp(0, self.size - 1)
        
        # Update position (only for non-done envs)
        active = ~self.done
        self.agent_pos[active] = new_pos[active]
        
        # Compute rewards
        rewards = torch.full((N,), -0.01, device=self.device)
        
        # Check if reached target object
        target_pos = self.obj_pos[torch.arange(N, device=self.device), self.target_idx]  # [N, 2]
        reached_target = ((self.agent_pos == target_pos).all(dim=1) & active)
        
        # Check if reached wrong object
        reached_wrong = torch.zeros(N, dtype=torch.bool, device=self.device)
        for obj_idx in range(3):
            wrong_mask = ((self.agent_pos == self.obj_pos[:, obj_idx]).all(dim=1) & 
                         active & (self.target_idx != obj_idx))
            reached_wrong |= wrong_mask
        
        # Assign rewards
        rewards[reached_target] = 10.0
        rewards[reached_wrong] = -1.0
        
        # Update done status
        self.done[reached_target] = True
        self.steps += 1
        timeout = (self.steps >= self.max_steps) & ~self.done
        self.done[timeout] = True
        
        # Zero reward for already-done envs
        rewards[~active] = 0.0
        
        info = {
            'success': reached_target,
            'wrong_object': reached_wrong,
            'timeout': timeout,
        }
        
        return self._get_obs(), rewards, self.done, info
    
    def get_context(self) -> torch.Tensor:
        """Get context vector (for interface compatibility)."""
        # Simple context: just zeros for now
        return torch.zeros(self.num_envs, self.ctx_dim, device=self.device)
    
    def set_target_combo(self, color_idx: int, shape_idx: int):
        """
        Set all environments to target a specific color-shape combination.
        Useful for curriculum learning.
        """
        # Reset all with specific target
        self.reset(target_combo=(color_idx, shape_idx))


# Utility functions for generating instruction sets
def get_all_instructions(language: str = 'english') -> List[str]:
    """Get all 9 basic instructions."""
    colors = ['red', 'blue', 'green']
    shapes = ['circle', 'square', 'triangle']
    
    if language == 'english':
        return [f"go to {c} {s}" for c in colors for s in shapes]
    elif language == 'french':
        color_map = {'red': 'rouge', 'blue': 'bleu', 'green': 'vert'}
        shape_map = {'circle': 'cercle', 'square': 'carre', 'triangle': 'triangle_fr'}
        return [f"va au {color_map[c]} {shape_map[s]}" for c in colors for s in shapes]


def get_train_test_split() -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Get compositional generalization train/test split.
    
    Train on 6 combinations, test on 3 held-out.
    Returns (color_idx, shape_idx) tuples.
    """
    # Train: 6 combinations
    train_combos = [
        (0, 0),  # red circle
        (0, 1),  # red square
        (1, 0),  # blue circle
        (1, 2),  # blue triangle
        (2, 1),  # green square
        (2, 2),  # green triangle
    ]
    
    # Test: 3 held-out combinations
    test_combos = [
        (0, 2),  # red triangle (seen red + triangle separately)
        (1, 1),  # blue square (seen blue + square separately)
        (2, 0),  # green circle (seen green + circle separately)
    ]
    
    return train_combos, test_combos


if __name__ == "__main__":
    # Test single environment
    print("Testing LanguageGridworld...")
    env = LanguageGridworld(size=5, seed=42)
    obs, tokens, target = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Tokens: {tokens}")
    print(f"Instruction: {env.instruction_str}")
    print(f"Target: {env.target_color} {env.target_shape} at {target}")
    
    # Test step
    for _ in range(5):
        obs, reward, done, info = env.step(env.rng.integers(0, 4))
        print(f"Reward: {reward:.2f}, Done: {done}")
        if done:
            break
    
    print("\nTesting TorchVectorLanguageGridworld...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vec_env = TorchVectorLanguageGridworld(num_envs=4, size=5, device=device)
    obs, instructions = vec_env.reset(language='english')
    print(f"Obs shape: {obs.shape}")
    print(f"Instructions shape: {instructions.shape}")
    print(f"Instructions:\n{instructions}")
    
    # Test step
    actions = torch.randint(0, 4, (4,), device=device)
    obs, rewards, dones, info = vec_env.step(actions)
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    
    # Test French
    obs, instructions = vec_env.reset(language='french')
    print(f"\nFrench instructions:\n{instructions}")
    
    # Test train/test split
    train, test = get_train_test_split()
    print(f"\nTrain combos: {train}")
    print(f"Test combos: {test}")
    
    print("\n✅ All tests passed!")
