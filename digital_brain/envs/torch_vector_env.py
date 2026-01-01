import torch

class TorchVectorPOMDP:
    """
    Pure PyTorch implementation of POMDPGridworld for massive parallelization on GPU.
    """
    def __init__(self, num_envs, size=5, device="cuda", seed=42):
        self.num_envs = num_envs
        self.size = size
        self.device = torch.device(device)
        self.max_steps = size * size * 2
        
        torch.manual_seed(seed)
        
        # State tensors
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.key_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.door_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.goal_pos = torch.full((num_envs, 2), size - 1, dtype=torch.long, device=self.device)
        
        self.has_key = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.door_open = torch.zeros(num_envs, dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        
        num_reset = len(indices)
        if num_reset == 0:
            return self._get_obs()

        # Simple random placement avoiding goal
        # Goal is at (size-1, size-1)
        for i in range(num_reset):
            idx = indices[i]
            forbidden = {(self.size-1, self.size-1)}
            
            # Agent
            while True:
                p = torch.randint(0, self.size, (2,), device=self.device)
                if tuple(p.tolist()) not in forbidden:
                    self.agent_pos[idx] = p
                    forbidden.add(tuple(p.tolist()))
                    break
            
            # Door
            while True:
                p = torch.randint(0, self.size, (2,), device=self.device)
                if tuple(p.tolist()) not in forbidden:
                    self.door_pos[idx] = p
                    forbidden.add(tuple(p.tolist()))
                    break
            
            # Key
            while True:
                p = torch.randint(0, self.size, (2,), device=self.device)
                if tuple(p.tolist()) not in forbidden:
                    self.key_pos[idx] = p
                    break

        self.has_key[indices] = False
        self.door_open[indices] = False
        self.steps[indices] = 0
        
        return self._get_obs()

    def _get_obs(self):
        # Flattened 3x3 neighborhood: (num_envs, 9)
        # 0: empty, 1: key, 2: door, 3: goal, 4: wall
        obs = torch.zeros((self.num_envs, 3, 3), dtype=torch.float32, device=self.device)
        
        # Batch coordinates for 3x3 grid around agent
        offsets = torch.tensor([[-1, -1], [-1, 0], [-1, 1],
                                [0, -1],  [0, 0],  [0, 1],
                                [1, -1],  [1, 0],  [1, 1]], device=self.device)
        
        # (num_envs, 9, 2)
        grid_pos = self.agent_pos.unsqueeze(1) + offsets.unsqueeze(0)
        
        # Out of bounds (Wall = 4.0)
        oob = (grid_pos < 0).any(dim=-1) | (grid_pos >= self.size).any(dim=-1)
        
        # Flatten to (num_envs * 9, 2) for easier comparison if needed, or stay at (num_envs, 9, 2)
        # We'll use broadasting
        
        # Check Key (1.0)
        is_key = (grid_pos == self.key_pos.unsqueeze(1)).all(dim=-1) & (~self.has_key.unsqueeze(1))
        
        # Check Door (2.0 if not open)
        is_door = (grid_pos == self.door_pos.unsqueeze(1)).all(dim=-1) & (~self.door_open.unsqueeze(1))
        
        # Check Goal (3.0)
        is_goal = (grid_pos == self.goal_pos.unsqueeze(1)).all(dim=-1)
        
        # Fill obs_flat (num_envs, 9)
        obs_flat = torch.zeros((self.num_envs, 9), device=self.device)
        obs_flat[oob] = 4.0
        obs_flat[is_key] = 1.0
        obs_flat[is_door] = 2.0
        obs_flat[is_goal] = 3.0
        
        return obs_flat

    def step(self, actions):
        # actions: (num_envs,) tensor
        move = torch.tensor([[-1, 0], [1, 0], [0, -1], [0, 1]], device=self.device)
        delta = move[actions.long()]
        
        new_pos = self.agent_pos + delta
        
        # Boundary check
        in_bounds = (new_pos >= 0).all(dim=-1) & (new_pos < self.size).all(dim=-1)
        
        # Door check
        hitting_door = (new_pos == self.door_pos).all(dim=-1) & (~self.door_open)
        
        can_move = in_bounds & (~hitting_door)
        
        # Update position
        self.agent_pos[can_move] = new_pos[can_move]
        
        rewards = torch.full((self.num_envs,), -0.01, device=self.device)
        
        # Key pickup
        at_key = (self.agent_pos == self.key_pos).all(dim=-1) & (~self.has_key)
        self.has_key[at_key] = True
        self.door_open[at_key] = True
        rewards[at_key] += 1.0
        
        # Goal reaching
        at_goal = (self.agent_pos == self.goal_pos).all(dim=-1) & self.door_open
        rewards[at_goal] += 10.0
        
        self.steps += 1
        dones = at_goal | (self.steps >= self.max_steps)
        
        # Auto-reset
        if dones.any():
            indices = torch.where(dones)[0]
            self.reset(indices)
            
        return self._get_obs(), rewards, dones, {}
