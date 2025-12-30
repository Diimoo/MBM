import torch
import numpy as np
import sys
import os
from digital_brain.brain import DigitalBrain
from digital_brain.datatypes import Obs, ModSignals
from digital_brain.envs.pomdp_gridworld import POMDPGridworld

class AblationBrain(DigitalBrain):
    def __init__(self, config, mode="full"):
        super().__init__(config)
        self.mode = mode

    def step(self, obs, reward, done, learn: bool = False):
        B = obs.x.shape[0]
        if self.state is None:
            self.reset(B, obs.x.device)

        # Handle per-environment resets for finished episodes
        done_mask = done.view(B, 1)
        if torch.any(done_mask):
            with torch.no_grad():
                self.state.z = self.state.z * (1 - done_mask.float())
                e_act, i_act, trace_ee = self.state.cortex_state
                e_act = e_act * (1 - done_mask.float())
                i_act = i_act * (1 - done_mask.float())
                self.state.cortex_state = (e_act, i_act, trace_ee)
                self.state.bg_state['prev_value'] = self.state.bg_state['prev_value'] * (1 - done_mask.float())
                self._prev_selection = self._prev_selection * (1 - done_mask.float())
                self._prev_pred = self._prev_pred * (1 - done_mask.float())
                mask = (1 - done_mask.squeeze().float())
                if mask.dim() == 0: mask = mask.unsqueeze(0)
                self._prev_mods.DA = self._prev_mods.DA * mask
                self._prev_mods.NE = self._prev_mods.NE * mask
                self._prev_mods.ACh = self._prev_mods.ACh * mask + (1 - mask) * 0.5
                self._prev_mods.HT5 = self._prev_mods.HT5 * mask + (1 - mask) * 0.5

        reward_b = reward.squeeze(1) if (reward.dim() == 2 and reward.shape[1] == 1) else reward
        done_b = done.squeeze(1).float() if (done.dim() == 2 and done.shape[1] == 1) else done.float()

        pred_err_vec = torch.mean((self._prev_pred - obs.x) ** 2, dim=1)

        # 1) Thalamus
        if self.mode == "no_thalamus":
            gated_x = obs.x
        else:
            gated_x = self.thalamus.gate(obs.x, self._prev_selection, self._prev_mods)

        # 2) Cortex
        z_t, pred_t, new_cortex_state = self.cortex.forward(gated_x, self.state.cortex_state)

        # 3) Hippocampus
        if self.mode == "no_hippo":
            novelty = torch.zeros(B, device=obs.x.device)
        else:
            novelty = self.hippocampus.novelty(z_t)
            _ = self.hippocampus.retrieve(z_t)

        # 4) Basal Ganglia
        selection, da, action, log_prob, value, entropy = self.bg.step(
            z_t, reward_b.unsqueeze(1), obs.ctx, done_b.unsqueeze(1), self.state.bg_state['prev_value']
        )
        if self.mode == "no_bg_selection":
            selection = torch.zeros_like(selection)

        # 5) Neuromodulators
        if self.mode == "no_neuromods":
            mods = ModSignals(
                DA=torch.zeros(B, device=obs.x.device),
                NE=torch.zeros(B, device=obs.x.device),
                ACh=torch.full((B,), 0.5, device=obs.x.device),
                HT5=torch.full((B,), 0.5, device=obs.x.device),
            )
        else:
            mods = self.neuromods.compute(z_t, da.squeeze(1), novelty, pred_err_vec)

        # 6) Plasticity (3-Factor Rule) - only if learn=True
        if learn and self.mode != "no_neuromods" and self.mode != "no_plasticity":
            self.cortex.update_weights(mods, new_cortex_state)

        # 7) Update recurrent runtime state
        self.state.z = z_t.detach()
        self.state.cortex_state = tuple(s.detach() for s in new_cortex_state)
        self.state.bg_state['prev_value'] = value.detach()

        self._prev_selection = selection.detach()
        self._prev_mods = ModSignals(
            DA=mods.DA.detach(),
            NE=mods.NE.detach(),
            ACh=mods.ACh.detach(),
            HT5=mods.HT5.detach(),
        )
        self._prev_pred = pred_t.detach()

        if learn and self.mode != "no_hippo":
            if torch.any(novelty > 0.6) or torch.any(reward_b != 0):
                self.hippocampus.encode(z_t)

        return action, log_prob, value, self.state, None, entropy

def eval_brain(brain, device, seeds, episodes=50, max_steps=150):
    succ_rates = []
    for seed in seeds:
        env = POMDPGridworld(size=5, seed=seed)
        succ = 0
        for _ in range(episodes):
            obs_np = env.reset()
            brain.reset(1, device=device)
            prev_reward = torch.tensor([[0.0]], device=device)
            prev_done = torch.tensor([[False]], device=device)
            done = False
            ep_ret = 0.0
            steps = 0
            while (not done) and steps < max_steps:
                obs = Obs(x=torch.from_numpy(obs_np).unsqueeze(0).to(device))
                action,_,_,_,_,_ = brain.step(obs, prev_reward, prev_done)
                obs_np, r, done, _ = env.step(int(action.item()))
                ep_ret += float(r)
                prev_reward = torch.tensor([[r]], dtype=torch.float32, device=device)
                prev_done = torch.tensor([[done]], device=device)
                steps += 1
            if ep_ret > 5.0: succ += 1
        succ_rates.append(succ / episodes)
    return np.mean(succ_rates), np.std(succ_rates)

def main():
    config = {'d_obs': 9, 'd_z': 32, 'd_sel': 4, 'd_act': 4}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    weights_path = "brain_phase3_best.pth"
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found")
        return

    modes = ["full", "no_hippo", "no_thalamus", "no_neuromods", "no_bg_selection"]
    seeds = [100, 101, 102, 103, 104] # Subset for speed
    
    results = {}
    for mode in modes:
        print(f"\nEvaluating mode: {mode}")
        brain = AblationBrain(config, mode=mode).to(device)
        brain.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        
        mean_sr, std_sr = eval_brain(brain, device, seeds)
        results[mode] = (mean_sr, std_sr)
        print(f"Result {mode}: SR = {mean_sr:.3f} +/- {std_sr:.3f}")

    print("\n--- Ablation Summary ---")
    for mode, (m, s) in results.items():
        print(f"{mode:<15}: {m:.3f} +/- {s:.3f}")

if __name__ == "__main__":
    main()
