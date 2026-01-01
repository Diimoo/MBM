import numpy as np
import torch

class RadialArmMaze:
    """
    Radial Arm Maze: Agent starts in center and must visit all arms once.
    Visiting an arm for the first time gives a reward.
    Re-visiting an arm gives no reward or a penalty.
    Tests episodic memory (remembering which arms were visited).
    """
    def __init__(self, num_arms=8, arm_length=3, seed=None):
        self.num_arms = num_arms
        self.arm_length = arm_length
        self.rng = np.random.default_rng(seed)
        self.action_space_n = num_arms + 1 # 0: stay/center, 1-N: go to arm N
        self.reset()

    def reset(self):
        self.pos = 0 # 0 is center
        self.visited = [False] * self.num_arms
        self.steps = 0
        self.max_steps = self.num_arms * self.arm_length * 2
        return self._get_obs()

    def _get_obs(self):
        # One-hot position + visited status
        obs = np.zeros(1 + self.num_arms + self.num_arms, dtype=np.float32)
        obs[self.pos] = 1.0
        for i, v in enumerate(self.visited):
            if v:
                obs[1 + self.num_arms + i] = 1.0
        return obs

    def step(self, action):
        self.steps += 1
        reward = -0.01
        done = False

        if action == 0:
            self.pos = 0
        elif 1 <= action <= self.num_arms:
            arm_idx = action - 1
            if not self.visited[arm_idx]:
                self.visited[arm_idx] = True
                reward += 1.0
            self.pos = action
        
        if all(self.visited):
            reward += 10.0
            done = True
        
        if self.steps >= self.max_steps:
            done = True
            
        return self._get_obs(), float(reward), bool(done), {}

class TorchVectorRadialArmMaze:
    """Vectorized Radial Arm Maze on GPU."""
    def __init__(self, num_envs, num_arms=8, arm_length=3, device="cuda", seed=42):
        self.num_envs = num_envs
        self.num_arms = num_arms
        self.arm_length = arm_length
        self.device = torch.device(device)
        self.max_steps = num_arms * arm_length * 2
        
        torch.manual_seed(seed)
        
        self.pos = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.visited = torch.zeros((num_envs, num_arms), dtype=torch.bool, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        
        self.pos[indices] = 0
        self.visited[indices] = False
        self.steps[indices] = 0
        return self._get_obs()

    def _get_obs(self):
        # obs shape: (num_envs, 1 + num_arms + num_arms)
        obs = torch.zeros((self.num_envs, 1 + self.num_arms * 2), device=self.device)
        
        # Position bits
        obs[torch.arange(self.num_envs, device=self.device), self.pos] = 1.0
        
        # Visited bits
        obs[:, 1+self.num_arms:] = self.visited.float()
        
        return obs

    def step(self, actions):
        # actions: (num_envs,)
        rewards = torch.full((self.num_envs,), -0.01, device=self.device)
        
        # Action 0: return to center
        mask_center = (actions == 0)
        self.pos[mask_center] = 0
        
        # Action 1-N: go to arm
        mask_arm = (actions > 0) & (actions <= self.num_arms)
        arm_indices = (actions[mask_arm] - 1).long()
        
        # Update visited and rewards
        # We need to handle per-env visited status
        # Get current visited status for the chosen arms
        env_indices = torch.where(mask_arm)[0]
        chosen_arms = arm_indices
        
        # Check if already visited
        # visited has shape (num_envs, num_arms)
        already_visited = self.visited[env_indices, chosen_arms]
        
        # Reward for new visit
        rewards[env_indices[~already_visited]] += 1.0
        # Update visited
        self.visited[env_indices, chosen_arms] = True
        
        # Update position
        self.pos[mask_arm] = actions[mask_arm].long()
        
        self.steps += 1
        
        # Done if all arms visited or max steps
        all_visited = self.visited.all(dim=1)
        rewards[all_visited] += 10.0
        
        dones = all_visited | (self.steps >= self.max_steps)
        
        if dones.any():
            res_indices = torch.where(dones)[0]
            self.reset(res_indices)
            
        return self._get_obs(), rewards, dones, {}
