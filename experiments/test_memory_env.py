import torch
from digital_brain.envs.torch_minigrid import TorchMiniGridMemory

def test_memory_env():
    num_envs = 1
    corridor_length = 7
    device = "cpu"
    env = TorchMiniGridMemory(num_envs=num_envs, corridor_length=corridor_length, device=device)
    
    print("Initial State:")
    print(env.render_ascii(0))
    
    # Move forward a few steps
    for _ in range(5):
        obs, reward, done, _ = env.step(torch.tensor([2]))
        print(f"Step, Reward: {reward.item()}, Done: {done.item()}")
        print(env.render_ascii(0))
    
    # Check if cue is visible in obs (should be in the beginning)
    obs = env.get_obs()
    print("Obs shape:", obs.shape)
    
    # Reset and check randomization
    env.reset()
    print("Reset State:")
    print(env.render_ascii(0))

if __name__ == "__main__":
    test_memory_env()
