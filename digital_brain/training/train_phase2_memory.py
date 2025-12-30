import torch
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld
import numpy as np

def train_phase2():
    """
    Phase 2 MVP:
    - Populate hippocampus with real Z(t) from cortex.
    - Evaluate recall with delay + noisy cues (NON-trivial).
    - No gradient training: current hippocampus has no learnable weights.
    """
    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'episodes': 50, 'max_steps': 100,
        'noise_std': 0.05, 'delay': 10, 'seed': 0,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    env = POMDPGridworld(size=5, seed=config['seed'])
    brain = DigitalBrain(config).to(device)
    brain.cortex.load_state_dict(torch.load("cortex_phase1.pth", map_location="cpu"))
    brain.to(device)

    print("Starting Phase 2: Episodic Memory (populate + evaluate)...")

    recall_errors = []
    hits = 0
    total = 0

    for ep in range(config['episodes']):
        obs_np = env.reset()
        brain.reset(1, device=device)

        prev_reward = torch.tensor([[0.0]], device=device)
        prev_done = torch.tensor([[False]], device=device)

        z_history = []
        done = False
        steps = 0

        while not done and steps < config['max_steps']: # Limit steps
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            obs = Obs(x=obs_t)
            
            # Use policy to step (we need to explore/move to get cues)
            # Or use random policy? The phase 2 focus is memory capacity.
            _action, _log_prob, _value, _, _log = brain.step(obs, prev_reward, prev_done)
            z_history.append(brain.state.z.detach().clone())

            # random action for exploration
            action_idx = np.random.randint(env.action_space_n)
            next_obs_np, reward, done, _ = env.step(action_idx)

            obs_np = next_obs_np
            prev_reward = torch.tensor([[reward]], dtype=torch.float32)
            prev_done = torch.tensor([[done]])
            steps += 1

            # delayed noisy query
            if len(z_history) > config['delay']:
                target_z = z_history[-config['delay']][0]
                cue = target_z + torch.randn_like(target_z) * config['noise_std']
                cue = cue.unsqueeze(0)

                z_ret = brain.hippocampus.retrieve(cue)
                mse = torch.mean((z_ret[0] - target_z) ** 2).item()
                recall_errors.append(mse)

                if mse < 1e-3:
                    hits += 1
                total += 1

        if (ep + 1) % 10 == 0:
            avg = float(np.mean(recall_errors)) if recall_errors else float("nan")
            hr = (hits / total) if total else 0.0
            print(f"Episode {ep+1}/{config['episodes']} | Avg Recall MSE: {avg:.6f} | HitRate: {hr:.3f}")

    avg = float(np.mean(recall_errors)) if recall_errors else float("nan")
    hr = (hits / total) if total else 0.0
    print("Phase 2 complete.")
    print(f"Final Avg Recall MSE: {avg:.6f} | Final HitRate: {hr:.3f}")

    torch.save(brain.state_dict(), "brain_phase2.pth")

if __name__ == "__main__":
    train_phase2()
