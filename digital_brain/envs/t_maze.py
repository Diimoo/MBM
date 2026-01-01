import numpy as np
import torch

class TMaze:
    """
    T-Maze: Agent must remember a cue shown at start,
    then navigate to the correct arm at the end.
    
    Layout:
        G_L ← ← ← ← ← C (cue)
                    ↓
                    S (start)
                    ↓
        G_R ← ← ← ←
    
    Agent sees cue at C (left or right), then must
    navigate through corridor and choose correct arm.
    Tests working memory over delay.
    """
    def __init__(self, corridor_length=5, seed=None):
        self.length = corridor_length
        self.cue = None
        self.pos = 0
        self.rng = np.random.default_rng(seed)
        self.action_space_n = 3 # 0: Forward, 1: Left, 2: Right
    
    def reset(self):
        self.cue = self.rng.choice(['left', 'right'])
        self.pos = 0
        return self._get_obs()
    
    def _get_obs(self):
        # One-hot: [pos, cue_left, cue_right, at_junction]
        # We'll use a fixed size observation vector
        obs = np.zeros(self.length + 3, dtype=np.float32)
        if self.pos < self.length:
            obs[self.pos] = 1.0
        
        if self.pos == 0:  # Show cue at start
            obs[-2] = 1.0 if self.cue == 'left' else 0.0
            obs[-1] = 1.0 if self.cue == 'right' else 0.0
        
        if self.pos == self.length:  # At junction
            obs[-3] = 1.0
        return obs
    
    def step(self, action):
        # Actions: 0=forward, 1=left, 2=right
        reward = -0.01
        done = False
        
        if action == 0:  # Forward
            if self.pos < self.length:
                self.pos += 1
            else:
                # Penalty for trying to move forward at junction
                reward = -0.1
        elif self.pos == self.length:  # At junction
            if (action == 1 and self.cue == 'left') or \
               (action == 2 and self.cue == 'right'):
                reward = 10.0
                done = True
            else:
                reward = -1.0
                done = True
        else:
            # Penalty for turning before junction
            reward = -0.1
        
        return self._get_obs(), float(reward), bool(done), {}

class TorchVectorTMaze:
    """Vectorized T-Maze on GPU."""
    def __init__(self, num_envs, corridor_length=5, device="cuda", seed=42):
        self.num_envs = num_envs
        self.length = corridor_length
        self.device = torch.device(device)
        self.max_steps = corridor_length + 5
        
        torch.manual_seed(seed)
        
        self.pos = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.cue = torch.zeros(num_envs, dtype=torch.long, device=self.device) # 0: left, 1: right
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        
        num_reset = len(indices)
        self.pos[indices] = 0
        self.cue[indices] = torch.randint(0, 2, (num_reset,), device=self.device)
        self.steps[indices] = 0
        return self._get_obs()

    def _get_obs(self):
        # obs shape: (num_envs, length + 3)
        obs = torch.zeros((self.num_envs, self.length + 3), device=self.device)
        
        # Position bits
        # Only set if pos < length
        mask_pos = self.pos < self.length
        obs[torch.where(mask_pos)[0], self.pos[mask_pos]] = 1.0
        
        # Cue bits at pos=0
        mask_start = self.pos == 0
        obs[mask_start, -2] = (self.cue[mask_start] == 0).float()
        obs[mask_start, -1] = (self.cue[mask_start] == 1).float()
        
        # Junction bit at pos=length
        mask_junction = self.pos == self.length
        obs[mask_junction, -3] = 1.0
        
        return obs

    def step(self, actions):
        # actions: (num_envs,)
        rewards = torch.full((self.num_envs,), -0.01, device=self.device)
        
        # Forward (0)
        mask_fwd = (actions == 0)
        mask_can_fwd = mask_fwd & (self.pos < self.length)
        self.pos[mask_can_fwd] += 1
        
        # Penalty for fwd at junction
        rewards[mask_fwd & (self.pos == self.length) & (actions == 0)] = -0.1
        
        # Choice at junction
        mask_at_junction = self.pos == self.length
        mask_turn = (actions == 1) | (actions == 2)
        
        # Correct choice
        mask_correct = mask_at_junction & (
            ((actions == 1) & (self.cue == 0)) | 
            ((actions == 2) & (self.cue == 1))
        )
        rewards[mask_correct] = 10.0
        
        # Incorrect choice at junction
        mask_wrong = mask_at_junction & mask_turn & (~mask_correct)
        rewards[mask_wrong] = -1.0
        
        # Penalty for turning before junction
        rewards[(~mask_at_junction) & mask_turn] = -0.1
        
        self.steps += 1
        dones = mask_correct | mask_wrong | (self.steps >= self.max_steps)
        
        if dones.any():
            indices = torch.where(dones)[0]
            self.reset(indices)
            
        return self._get_obs(), rewards, dones, {}
