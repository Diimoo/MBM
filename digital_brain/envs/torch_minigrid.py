import torch
import numpy as np

class TorchMiniGrid:
    """
    GPU-accelerated implementation of MiniGrid-Empty environments.
    Mimics the logic of gymnasium-minigrid but runs purely on Torch tensors.
    
    Supports:
    - Egocentric 7x7 observations
    - Directional agent (0=Right, 1=Down, 2=Left, 3=Up)
    - Actions: 0=TurnLeft, 1=TurnRight, 2=Forward
    - Objects: Wall, Goal, Empty, Key, Ball
    """
    
    # Constants mapping (matches minigrid.core.constants)
    OBJECT_TO_IDX = {
        'unseen': 0,
        'empty': 1,
        'wall': 2,
        'floor': 3,
        'door': 4,
        'key': 5,
        'ball': 6,
        'box': 7,
        'goal': 8,
        'lava': 9,
        'agent': 10,
    }
    COLOR_TO_IDX = {
        'red': 0,
        'green': 1,
        'blue': 2,
        'purple': 3,
        'yellow': 4,
        'grey': 5,
    }
    
    # Actions
    ACTION_LEFT = 0
    ACTION_RIGHT = 1
    ACTION_FORWARD = 2
    
    def __init__(self, num_envs, width=None, height=None, size=None, device="cuda", seed=42):
        self.num_envs = num_envs
        if size is not None:
            width = width or size
            height = height or size
        self.width = width or 8
        self.height = height or 8
        self.device = torch.device(device)
        self.max_steps = 4 * self.width * self.height
        
        torch.manual_seed(seed)
        
        # State
        self.agent_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.agent_dir = torch.zeros(num_envs, dtype=torch.long, device=self.device) # 0-3
        self.goal_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # Memory-specific state (cue matching)
        self.correct_obj_idx = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.correct_obj_color = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        # Grid: (num_envs, W, H, 3)
        self.grid = torch.zeros((num_envs, self.width, self.height, 3), dtype=torch.long, device=self.device)
        
        self._init_static_grid()
        self.reset()
        
    def _init_static_grid(self):
        # Fill with walls
        self.grid[..., 0] = self.OBJECT_TO_IDX['wall']
        self.grid[..., 1] = self.COLOR_TO_IDX['grey']
        
        # Clear inside
        self.grid[:, 1:self.width-1, 1:self.height-1, 0] = self.OBJECT_TO_IDX['empty']
        self.grid[:, 1:self.width-1, 1:self.height-1, 1] = 0
        
    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        n = len(indices)
        if n == 0: return self.get_obs()
        
        self.steps[indices] = 0
        
        # Goal at bottom-right
        self.goal_pos[indices] = torch.tensor([self.width-2, self.height-2], device=self.device)
        
        # Reset inside to empty
        self.grid[indices, 1:self.width-1, 1:self.height-1, 0] = self.OBJECT_TO_IDX['empty']
        self.grid[indices, 1:self.width-1, 1:self.height-1, 1] = 0
        
        # Place Goal
        gx = self.goal_pos[indices, 0]
        gy = self.goal_pos[indices, 1]
        self.grid[indices, gx, gy, 0] = self.OBJECT_TO_IDX['goal']
        self.grid[indices, gx, gy, 1] = self.COLOR_TO_IDX['green']
        
        # Place Agent
        while True:
            px = torch.randint(1, self.width-1, (n,), device=self.device)
            py = torch.randint(1, self.height-1, (n,), device=self.device)
            overlap = (px == gx) & (py == gy)
            if not overlap.any():
                self.agent_pos[indices] = torch.stack([px, py], dim=1)
                break
            self.agent_pos[indices] = torch.stack([px, py], dim=1)
            self.agent_pos[indices[overlap]] = torch.tensor([1, 1], device=self.device)
            break
            
        self.agent_dir[indices] = torch.randint(0, 4, (n,), device=self.device)
        return self.get_obs()

    def step(self, actions):
        # 1. Handle Turns
        turn_left = (actions == self.ACTION_LEFT)
        turn_right = (actions == self.ACTION_RIGHT)
        self.agent_dir[turn_left] = (self.agent_dir[turn_left] - 1) % 4
        self.agent_dir[turn_right] = (self.agent_dir[turn_right] + 1) % 4
        
        # 2. Handle Forward
        forward = (actions == self.ACTION_FORWARD)
        dx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        dy = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        mask = (self.agent_dir == 0) & forward; dx[mask] = 1
        mask = (self.agent_dir == 1) & forward; dy[mask] = 1
        mask = (self.agent_dir == 2) & forward; dx[mask] = -1
        mask = (self.agent_dir == 3) & forward; dy[mask] = -1
        
        new_pos = self.agent_pos.clone()
        new_pos[:, 0] += dx
        new_pos[:, 1] += dy
        
        # Check collisions
        batch_idx = torch.arange(self.num_envs, device=self.device)
        target_objs = self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1], 0]
        target_colors = self.grid[batch_idx, new_pos[:, 0], new_pos[:, 1], 1]
        
        blocked = (target_objs == self.OBJECT_TO_IDX['wall'])
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Goal (8)
        at_goal = (target_objs == self.OBJECT_TO_IDX['goal'])
        rewards[at_goal] += 1.0
        dones[at_goal] = True
        
        # Memory objects (Ball=6, Key=5)
        is_obj = (target_objs == self.OBJECT_TO_IDX['ball']) | (target_objs == self.OBJECT_TO_IDX['key'])
        if is_obj.any():
            # In Memory task, reward only in the target room (far right)
            # We can check if new_pos.x > width / 2 or similar
            # For base TorchMiniGrid, width is 8, target room starts at 6? 
            # Actually, let's just make it general: if it's a Memory env, check X.
            # Or better, just check if it matches and it's not the cue position.
            match = (target_objs == self.correct_obj_idx) & (target_colors == self.correct_obj_color)
            
            # Identify cue collisions (X < width - 5)
            # Target room in S7 starts at X=11. Width is 15.
            # So X > 10 is target room.
            in_target_room = (new_pos[:, 0] > (self.width - 5))
            
            rewards[is_obj & match & in_target_room] += 1.0
            dones[is_obj & in_target_room] = True
            
            # Objects are always blocking
            blocked = blocked | is_obj
            
        # Allow move if not blocked
        can_move = forward & (~blocked)
        self.agent_pos[can_move] = new_pos[can_move]
        
        self.steps += 1
        timeout = self.steps >= self.max_steps
        dones = dones | timeout
        
        if dones.any():
            self.reset(torch.where(dones)[0])
            
        return self.get_obs(), rewards, dones, {}

    def get_obs(self):
        view_size = 7
        padding = view_size
        padded_W = self.width + 2 * padding
        padded_H = self.height + 2 * padding
        padded_grid = torch.zeros((self.num_envs, padded_W, padded_H, 3), dtype=torch.long, device=self.device)
        padded_grid[:, padding:padding+self.width, padding:padding+self.height, :] = self.grid
        
        vx = torch.arange(view_size, device=self.device)
        vy = torch.arange(view_size, device=self.device)
        grid_vy, grid_vx = torch.meshgrid(vy, vx, indexing='ij')
        rel_x = grid_vx - 3
        rel_y = grid_vy - 6
        rel_x = rel_x.flatten()
        rel_y = rel_y.flatten()
        
        fwd = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        right = torch.zeros((self.num_envs, 2), dtype=torch.long, device=self.device)
        mask = (self.agent_dir == 0); fwd[mask] = torch.tensor([1, 0], device=self.device); right[mask] = torch.tensor([0, 1], device=self.device)
        mask = (self.agent_dir == 1); fwd[mask] = torch.tensor([0, 1], device=self.device); right[mask] = torch.tensor([-1, 0], device=self.device)
        mask = (self.agent_dir == 2); fwd[mask] = torch.tensor([-1, 0], device=self.device); right[mask] = torch.tensor([0, -1], device=self.device)
        mask = (self.agent_dir == 3); fwd[mask] = torch.tensor([0, -1], device=self.device); right[mask] = torch.tensor([1, 0], device=self.device)
        
        global_offsets = (rel_x.view(1, -1, 1) * right.unsqueeze(1)) - (rel_y.view(1, -1, 1) * fwd.unsqueeze(1))
        center_pos = self.agent_pos + padding
        sample_pos = center_pos.unsqueeze(1) + global_offsets
        batch_ids = torch.arange(self.num_envs, device=self.device).view(-1, 1).expand(-1, 49)
        obs_flat = padded_grid[batch_ids, sample_pos[:,:,0], sample_pos[:,:,1]]
        obs = obs_flat.view(self.num_envs, view_size, view_size, 3)
        return obs.float()

    def render_ascii(self, env_idx=0):
        mapping = {0: ' ', 1: '.', 2: 'W', 8: 'G', 10: 'A', 5: 'K', 6: 'B'}
        grid_data = self.grid[env_idx, :, :, 0].cpu().numpy()
        ax, ay = self.agent_pos[env_idx].cpu().numpy()
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                if x == ax and y == ay: line += 'A'
                else: line += mapping.get(grid_data[x, y], '?')
            lines.append(line)
        return "\n".join(lines)

class TorchMiniGridMemory(TorchMiniGrid):
    def __init__(self, num_envs, corridor_length=7, device="cuda", seed=42):
        self.corridor_length = corridor_length
        w = 1 + 3 + corridor_length + 3 + 1
        h = 5
        super().__init__(num_envs, width=w, height=h, device=device, seed=seed)
        self.max_steps = 4 * (w + h)
        
    def _init_static_grid(self):
        self.grid[..., 0] = self.OBJECT_TO_IDX['wall']
        self.grid[..., 1] = self.COLOR_TO_IDX['grey']
        self.grid[:, 1:4, 1:4, 0] = self.OBJECT_TO_IDX['empty']
        cl = self.corridor_length
        self.grid[:, 4:4+cl, 2, 0] = self.OBJECT_TO_IDX['empty']
        self.grid[:, 4+cl:4+cl+3, 1:4, 0] = self.OBJECT_TO_IDX['empty']
        
    def reset(self, indices=None):
        if indices is None: indices = torch.arange(self.num_envs, device=self.device)
        n = len(indices)
        if n == 0: return self.get_obs()
        self.steps[indices] = 0
        
        # Reset grid regions
        self.grid[indices, 1:4, 1:4, 0] = self.OBJECT_TO_IDX['empty']
        cl = self.corridor_length
        self.grid[indices, 4:4+cl, 2, 0] = self.OBJECT_TO_IDX['empty']
        self.grid[indices, 4+cl:4+cl+3, 1:4, 0] = self.OBJECT_TO_IDX['empty']
        
        obj_types = torch.tensor([self.OBJECT_TO_IDX['ball'], self.OBJECT_TO_IDX['key']], device=self.device)
        colors = torch.tensor([self.COLOR_TO_IDX['green'], self.COLOR_TO_IDX['red']], device=self.device)
        type_idx = torch.randint(0, 2, (n,), device=self.device)
        color_idx = torch.randint(0, 2, (n,), device=self.device)
        cue_type = obj_types[type_idx]
        cue_color = colors[color_idx]
        dist_color = colors[1 - color_idx]
        
        self.correct_obj_idx[indices] = cue_type
        self.correct_obj_color[indices] = cue_color
        
        self.grid[indices, 2, 2, 0] = cue_type
        self.grid[indices, 2, 2, 1] = cue_color
        self.agent_pos[indices] = torch.tensor([1, 2], device=self.device).repeat(n, 1)
        self.agent_dir[indices] = 0
        
        target_x = 4 + cl + 2
        side = torch.randint(0, 2, (n,), device=self.device)
        top_mask = (side == 0)
        bot_mask = (side == 1)
        
        if top_mask.any():
            idx = indices[top_mask]
            self.grid[idx, target_x, 1, 0] = self.correct_obj_idx[idx]
            self.grid[idx, target_x, 1, 1] = self.correct_obj_color[idx]
            self.grid[idx, target_x, 3, 0] = self.correct_obj_idx[idx]
            self.grid[idx, target_x, 3, 1] = dist_color[top_mask]
            
        if bot_mask.any():
            idx = indices[bot_mask]
            self.grid[idx, target_x, 3, 0] = self.correct_obj_idx[idx]
            self.grid[idx, target_x, 3, 1] = self.correct_obj_color[idx]
            self.grid[idx, target_x, 1, 0] = self.correct_obj_idx[idx]
            self.grid[idx, target_x, 1, 1] = dist_color[bot_mask]
            
        return self.get_obs()

