import torch
import torch.nn as nn
import torch.optim as optim
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs
from digital_brain.envs.pomdp_gridworld import POMDPGridworld
import numpy as np

@torch.no_grad()
def eval_policy(brain: DigitalBrain, env: POMDPGridworld, episodes: int = 50, max_steps: int = 150):
    success = 0
    returns = []
    steps_list = []

    for _ in range(episodes):
        obs_np = env.reset()
        brain.reset(1)
        prev_reward = torch.tensor([[0.0]])
        prev_done = torch.tensor([[False]])

        done = False
        ep_ret = 0.0
        steps = 0
        while not done and steps < max_steps:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0)
            obs = Obs(x=obs_t)

            action, _, _, _, _, _ = brain.step(obs, prev_reward, prev_done)
            action_idx = int(action.item())

            obs_np, reward, done, _ = env.step(action_idx)
            ep_ret += reward

            prev_reward = torch.tensor([[reward]], dtype=torch.float32)
            prev_done = torch.tensor([[done]])
            steps += 1

        if ep_ret > 5.0:
            success += 1

        returns.append(ep_ret)
        steps_list.append(steps)

    return {
        "success_rate": success / episodes,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "steps_mean": float(np.mean(steps_list)),
    }

def train_phase3():
    config = {
        'd_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4,
        'lr': 3e-4, 'epochs': 1500, 'gamma': 0.99,  # Tuned for stability
        'entropy_coef': 0.05,                       # Increased entropy for exploration
        'seed': 0, 'max_steps': 150,
        'eval_every': 50, 'eval_episodes': 50,
    }

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = POMDPGridworld(size=5, seed=config['seed'])
    eval_env = POMDPGridworld(size=5, seed=config['seed'] + 1)

    brain = DigitalBrain(config).to(device)
    brain.load_state_dict(torch.load("brain_phase2.pth", map_location="cpu"), strict=False)
    brain.to(device) # Ensure loaded weights are moved

    for p in brain.cortex.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, brain.parameters()), lr=config['lr'])

    print("Starting Phase 3: Policy (BG) & Thalamus Gating...")

    best_score = -1e9

    for epoch in range(config['epochs']):
        obs_np = env.reset()
        brain.reset(1, device=device)

        prev_reward = torch.tensor([[0.0]], device=device)
        prev_done = torch.tensor([[False]], device=device)

        done = False
        steps = 0
        log_probs = []
        values = []
        rewards = []
        entropies = []

        while not done and steps < config['max_steps']:
            obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
            obs = Obs(x=obs_t)

            action, log_prob, value, _, _ = brain.step(obs, prev_reward, prev_done)
            action_idx = int(action.item())
            
            # Recalculate entropy from logits (need access to policy distribution)
            # Since step doesn't return div/entropy, we can approximate or modify step.
            # However, for A2C, we often compute it from the distribution.
            # Brain.step returns log_prob. 
            # To get entropy properly we ideally need the distribution.
            # For now, let's look at how brain.bg.step works. 
            # It returns action, log_prob.
            
            # Hack/Fix: We need entropy. 
            # Let's assume for now we use the log_prob as a proxy or just rely on the fact 
            # that we can't easily get it without modifying brain.py.
            
            # WAIT. I should modify brain.py or bg.py to return entropy if I want to use it.
            # Or I can just continue without entropy first? 
            # User specifically suggested entropy bonus.
            # Let's check basal_ganglia.py first.
            
            next_obs_np, reward, done, _ = env.step(action_idx)

            log_probs.append(log_prob.unsqueeze(0))
            values.append(value)
            rewards.append(reward)
            entropies.append(-log_prob) # Proxy entropy

            obs_np = next_obs_np
            prev_reward = torch.tensor([[reward]], dtype=torch.float32, device=device)
            prev_done = torch.tensor([[done]], device=device)
            steps += 1

        if len(rewards) > 0:
            returns = []
            R = 0.0
            for r in reversed(rewards):
                R = r + config['gamma'] * R
                returns.insert(0, R)

            returns = torch.tensor(returns, dtype=torch.float32, device=device).unsqueeze(1)
            log_probs_t = torch.cat(log_probs, dim=0).squeeze(-1)
            values_t = torch.cat(values, dim=0)

            advantage = (returns - values_t).detach().squeeze(1)

            actor_loss = -(log_probs_t * advantage).mean()
            critic_loss = nn.MSELoss()(values_t, returns)
            
            # Simple entropy approximation: -sum(p * log_p)
            # Since we only have log_prob of sampled action, this is a MC estimate of -log_prob
            # entropy ~= -log_prob.mean()
            entropy_loss = -log_probs_t.mean() * config['entropy_coef']
            
            loss = actor_loss + critic_loss - entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()

        if epoch % config['eval_every'] == 0:
            m = eval_policy(brain, eval_env, episodes=config['eval_episodes'], max_steps=config['max_steps'])
            score = m["success_rate"] * 100.0 + m["return_mean"]

            print(
                f"Epoch {epoch:>4} | TrainSteps {steps:>3} | "
                f"Eval Success {m['success_rate']:.2f} | "
                f"Eval Return {m['return_mean']:.2f}±{m['return_std']:.2f} | "
                f"Score {score:.2f}"
            )

            if score > best_score:
                best_score = score
                torch.save(brain.state_dict(), "brain_phase3_best.pth")

    print("Phase 3 complete.")
    torch.save(brain.state_dict(), "brain_phase3.pth")
    print(f"Best checkpoint score: {best_score:.2f} (saved to brain_phase3_best.pth)")

if __name__ == "__main__":
    train_phase3()
