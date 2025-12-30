import numpy as np
import torch
from .pomdp_gridworld import POMDPGridworld

class VectorPOMDP:
    """
    Simple vectorized wrapper for POMDPGridworld.
    Runs N environments in parallel.
    """
    def __init__(self, num_envs, size=5, seed=42):
        self.num_envs = num_envs
        self.size = size
        self.envs = [POMDPGridworld(size=size, seed=seed + i) for i in range(num_envs)]
        
    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)
        
    def step(self, actions):
        """
        actions: (num_envs,)
        """
        results = [env.step(a) for env, a in zip(self.envs, actions)]
        obs, rewards, dones, infos = zip(*results)
        return np.array(obs), np.array(rewards), np.array(dones), infos
