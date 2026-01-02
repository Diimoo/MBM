import gymnasium as gym
import minigrid
import numpy as np

env = gym.make("MiniGrid-Empty-5x5-v0", render_mode="rgb_array")
obs, _ = env.reset()
print("Keys:", obs.keys())
print("Image shape:", obs['image'].shape)
print("Image dtype:", obs['image'].dtype)
print("Image sample:\n", obs['image'])

env.close()
