import torch
import numpy as np

class TorchVectorCartPole:
    """
    Pure PyTorch implementation of CartPole for massive parallelization on GPU.
    """
    def __init__(self, num_envs, device="cuda", seed=42):
        self.num_envs = num_envs
        self.device = torch.device(device)
        
        # CartPole parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        
        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * np.pi / 360
        self.x_threshold = 2.4
        
        self.max_steps = 500
        
        torch.manual_seed(seed)
        
        # State tensors: (x, x_dot, theta, theta_dot)
        self.state = torch.zeros((num_envs, 4), dtype=torch.float32, device=self.device)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.reset()

    def reset(self, indices=None):
        if indices is None:
            indices = torch.arange(self.num_envs, device=self.device)
        
        num_reset = len(indices)
        # Uniform initialization like OpenAI Gym
        self.state[indices] = torch.rand((num_reset, 4), device=self.device) * 0.1 - 0.05
        self.steps[indices] = 0
        return self.state.clone()

    def step(self, actions):
        # actions: (num_envs,) tensor of 0 or 1
        x, x_dot, theta, theta_dot = self.state[:, 0], self.state[:, 1], self.state[:, 2], self.state[:, 3]
        
        force = torch.where(actions == 1, self.force_mag, -self.force_mag)
        costheta = torch.cos(theta)
        sintheta = torch.sin(theta)
        
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass
        
        # Euler integration
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        
        self.state = torch.stack([x, x_dot, theta, theta_dot], dim=1)
        self.steps += 1
        
        # Done conditions
        dones = (x < -self.x_threshold) | (x > self.x_threshold) | \
                (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians) | \
                (self.steps >= self.max_steps)
        
        # Reward is 1.0 for every step until done
        rewards = torch.ones(self.num_envs, device=self.device)
        
        # Auto-reset
        if dones.any():
            res_indices = torch.where(dones)[0]
            self.reset(res_indices)
            
        return self.state.clone(), rewards, dones, {}
