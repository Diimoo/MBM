import gymnasium as gym
import minigrid
import matplotlib.pyplot as plt

env = gym.make("MiniGrid-MemoryS7-v0", render_mode="rgb_array")
obs, _ = env.reset()

print("Grid size:", env.unwrapped.width, env.unwrapped.height)
print("Observation shape:", obs['image'].shape)

# Print a simplified grid representation
# Wall=2, Floor=?, Key=?, Ball=?
# We need to map the image indices to names if possible, or just print the simplified grid
grid = env.unwrapped.grid
print("Grid objects:")
for i in range(env.unwrapped.width):
    for j in range(env.unwrapped.height):
        obj = grid.get(i, j)
        if obj is not None and obj.type != 'wall':
            print(f"({i},{j}): {obj.type} {obj.color}")

env.close()
