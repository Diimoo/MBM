import numpy as np
import torch
from .pomdp_gridworld import POMDPGridworld

class VectorPOMDP:
    """
    CPU vector wrapper (python list). IMPORTANT:
    - Auto-resets envs that are done.
    - Returns obs for the next state (post-step; for done envs = first obs of next episode).
    """
    def __init__(self, num_envs, size=5, seed=42):
        self.num_envs = num_envs
        self.size = size
        self.envs = [POMDPGridworld(size=size, seed=seed + i) for i in range(num_envs)]

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.asarray(obs, dtype=np.float32)

    def step(self, actions):
        obs = np.zeros((self.num_envs, 9), dtype=np.float32)
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        dones = np.zeros((self.num_envs,), dtype=np.bool_)
        infos = [None] * self.num_envs

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, d, info = env.step(int(a))
            # auto-reset
            if d:
                o = env.reset()
                info = dict(info or {})
                info["reset"] = True
            obs[i] = o
            rewards[i] = r
            dones[i] = d
            infos[i] = info

        return obs, rewards, dones, infos
