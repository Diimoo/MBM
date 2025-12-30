import torch
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

def run_demo(max_steps: int = 200, seed: int = 0):
    config = {'d_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = POMDPGridworld(size=5, seed=seed)
    brain = DigitalBrain(config).to(device)

    ckpts = ["brain_phase3_best.pth", "brain_phase4.pth",
             "brain_phase3.pth", "brain_phase2.pth"]
    loaded = False
    for ckpt in ckpts:
        try:
            brain.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
            print(f"Loaded {ckpt}")
            loaded = True
            break
        except Exception:
            pass
    if not loaded:
        print("No checkpoint loaded, running with initial weights.")
        
    brain.to(device)

    obs_np = env.reset()
    brain.reset(1, device=device)

    print("\n--- Digital Brain Demo Run ---")
    print(f"{'Step':<5} | {'Action':<6} | {'Reward':<6} | {'DA':<7} | {'NE':<6} | {'ACh':<6} | {'Surp':<7}")
    print("-" * 72)

    total_reward = 0.0
    prev_reward = torch.tensor([[0.0]], device=device)
    prev_done = torch.tensor([[False]], device=device)

    for step in range(max_steps):
        obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
        obs = Obs(x=obs_t)
        
        # We need to handle the returns from brain.step
        # It returns: action, log_prob, value, state, log
        action, _, _, _, log = brain.step(obs, prev_reward, prev_done)
        action_idx = int(action.item())

        next_obs_np, reward, done, info = env.step(action_idx)

        print(
            f"{step:<5} | {action_idx:<6} | {reward:<6.2f} | "
            f"{log.mod_signals.get('DA', 0.0):<7.2f} | {log.mod_signals.get('NE', 0.0):<6.2f} | "
            f"{log.mod_signals.get('ACh', 0.0):<6.2f} | {log.pred_error:<7.4f}"
        )

        total_reward += reward
        obs_np = next_obs_np
        prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
        prev_done = torch.tensor([[done]], device=device)

        if done:
            break

    print("-" * 72)
    print(f"Goal Reached: {done}")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Has Key: {env.has_key}, Door Open: {env.door_open}")

if __name__ == "__main__":
    run_demo()
