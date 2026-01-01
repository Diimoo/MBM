import torch
import torch.nn as nn
from .datatypes import Obs, BrainState, StepLog, ModSignals

class DigitalBrain(nn.Module):
    """
    Closed-loop, modular 'digital brain' orchestrator.

    FIXES:
    - Uses real reward/done inputs (caller must pass last env reward/done).
    - Action computed from *current* latent state z_t (no 1-step lag).
    - pred_error = surprise: MSE(prev_pred, current_obs).
    - Stores prev_selection/prev_mods to gate the next perception step.
    """
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        from .modules.cortex import Cortex
        from .modules.thalamus import Thalamus
        from .modules.hippocampus import Hippocampus
        from .modules.basal_ganglia import BasalGanglia
        from .modules.cerebellum import Cerebellum
        from .modules.neuromodulators import Neuromodulators

        self.cortex = Cortex(
            config['d_obs'], 
            config['d_z'], 
            config['d_act'], 
            sparse=config.get('sparse_cortex', False),
            sparsity=config.get('sparsity', 0.01),
            locality_radius=config.get('locality_radius', None)
        )
        self.thalamus = Thalamus(config['d_obs'], config['d_sel'])
        self.hippocampus = Hippocampus(config['d_z'])
        self.bg = BasalGanglia(config['d_z'], config['d_sel'], config['d_act'])
        self.cerebellum = Cerebellum(config['d_z'], config['d_obs'], config['d_act'])
        self.neuromods = Neuromodulators()

        self.state: BrainState | None = None

        # Runtime (non-parameter) recurrent helpers
        self._prev_selection: torch.Tensor | None = None  # (B, d_sel)
        self._prev_mods: ModSignals | None = None         # per-batch scalars
        self._prev_pred: torch.Tensor | None = None       # (B, d_obs)

    def reset(self, batch_size: int, device: str | torch.device = "cpu"):
        device = torch.device(device)
        self.to(device) # Ensure modules are on device
        
        z0 = torch.zeros(batch_size, self.config['d_z'], device=device)
        
        # Determine trace shape based on whether cortex is sparse
        if hasattr(self.cortex.microcircuit, 'W_ee_values'):
            # Sparse: trace matches number of connections
            trace0 = torch.zeros(self.cortex.microcircuit.W_ee_values.shape[0], device=device)
        else:
            # Dense: trace is (d_z, d_z)
            trace0 = torch.zeros(self.config['d_z'], self.config['d_z'], device=device)
            
        self.state = BrainState(
            z=z0,
            cortex_state=(z0, z0, trace0),  
            bg_state={'prev_value': torch.zeros(batch_size, 1, device=device)},
            hip_state=None,
            cerebellum_state=None
        )
        self._prev_selection = torch.zeros(batch_size, self.config['d_sel'], device=device)
        self._prev_mods = ModSignals(
            DA=torch.zeros(batch_size, device=device),
            NE=torch.zeros(batch_size, device=device),
            ACh=torch.full((batch_size,), 0.5, device=device),
            HT5=torch.full((batch_size,), 0.5, device=device),
        )
        self._prev_pred = torch.zeros(batch_size, self.config['d_obs'], device=device)

    def step(self, obs: Obs, reward: torch.Tensor, done: torch.Tensor, learn: bool = True):
        """
        Args:
            obs: current observation (Obs.x shape (B, d_obs))
            reward: last env reward (shape (B,1) or (B,))
            done: last env done flag (shape (B,1) or (B,))
            learn: if True, updates plastic weights and encodes to hippocampus.

        Returns:
            action, log_prob, value, state, log, entropy
        """
        # Read granular control flags from config
        use_hip = self.config.get('use_hippocampus', True)
        use_plas = self.config.get('use_plasticity', True)
        use_mem_pol = self.config.get('use_memory_policy', True)
        use_cereb = self.config.get('use_cerebellum', True)

        B = obs.x.shape[0]
        if self.state is None:
            self.reset(B, obs.x.device)

        # Handle per-environment resets for finished episodes
        done_mask = done.view(B, 1)
        if torch.any(done_mask):
            with torch.no_grad():
                # Zero out internal states for finished envs
                self.state.z = self.state.z * (1 - done_mask.float())
                
                # cortex_state: (e_act, i_act, trace_ee)
                e_act, i_act, trace_ee = self.state.cortex_state
                e_act = e_act * (1 - done_mask.float())
                i_act = i_act * (1 - done_mask.float())
                # trace_ee is shared across batch in SynapticPlasticity.update_trace (averages over batch)
                # so we don't necessarily reset the global trace, but the local activities
                self.state.cortex_state = (e_act, i_act, trace_ee)
                
                self.state.bg_state['prev_value'] = self.state.bg_state['prev_value'] * (1 - done_mask.float())
                self._prev_selection = self._prev_selection * (1 - done_mask.float())
                self._prev_pred = self._prev_pred * (1 - done_mask.float())
                
                # Reset mods for done envs to default
                mask = (1 - done_mask.squeeze().float())
                if mask.dim() == 0: mask = mask.unsqueeze(0) # Handle B=1
                self._prev_mods.DA = self._prev_mods.DA * mask
                self._prev_mods.NE = self._prev_mods.NE * mask
                self._prev_mods.ACh = self._prev_mods.ACh * mask + (1 - mask) * 0.5
                self._prev_mods.HT5 = self._prev_mods.HT5 * mask + (1 - mask) * 0.5

        # Normalize reward/done shapes
        reward_b = reward.squeeze(1) if (reward.dim() == 2 and reward.shape[1] == 1) else reward
        done_b = done.squeeze(1).float() if (done.dim() == 2 and done.shape[1] == 1) else done.float()

        # Surprise: previous prediction vs current obs
        pred_err_vec = torch.mean((self._prev_pred - obs.x) ** 2, dim=1)

        # 1) Thalamus gating of current inputs using previous selection/mods
        gated_x = self.thalamus.gate(obs.x, self._prev_selection, self._prev_mods)

        # 2) Cortex update on gated inputs
        # update_trace controlled by global learn AND config plasticity flag
        z_t, pred_t, new_cortex_state = self.cortex.forward(gated_x, self.state.cortex_state, update_trace=(learn and use_plas))

        # 3) Hippocampus novelty + retrieval
        novelty = torch.zeros(B, device=obs.x.device)
        retrieved = None
        if use_hip:
            novelty = self.hippocampus.novelty(z_t)
            retrieved = self.hippocampus.retrieve(z_t)

        # 4) Basal Ganglia: action/selection + TD-RPE (DA) computed using last reward and current value
        # 7) Cerebellum: Compute correction signal
        correction = None
        if use_cereb:
            correction, _timing = self.cerebellum.forward(z_t, obs.x)

        # 4.1) Basal Ganglia: Step with memory-augmented policy and cerebellar correction
        selection, da, action, log_prob, value, entropy = self.bg.step(
            z_t, 
            reward_b.unsqueeze(1), 
            obs.ctx, 
            done_b.unsqueeze(1), 
            self.state.bg_state['prev_value'],
            memory_context=retrieved if use_mem_pol else None,
            cerebellum_correction=correction
        )

        # 5) Neuromodulators (use pred_error as ACh proxy)
        mods = self.neuromods.compute(z_t, da.squeeze(1), novelty, pred_err_vec)

        # 6) Learning-related side effects
        if learn:
            # Plasticity Update (3-Factor Rule)
            if use_plas:
                self.cortex.update_weights(mods, new_cortex_state)
            
            # Hippocampus encode on novelty/reward
            if use_hip:
                if torch.any(novelty > 0.6) or torch.any(reward_b != 0):
                    self.hippocampus.encode(z_t)
        self.state.z = z_t.detach()
        # Detach each element of the tuple state (E, I)
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

        log = StepLog(
            pred_error=pred_err_vec.mean().item(),
            rpe=da.mean().item(),
            novelty=novelty.mean().item(),
            gate_stats={
                'prev_selection_mean': self._prev_selection.mean().item(),
                'ACh_mean': mods.ACh.mean().item(),
            },
            mod_signals={
                'DA': mods.DA.mean().item(),
                'NE': mods.NE.mean().item(),
                'ACh': mods.ACh.mean().item(),
                '5HT': mods.HT5.mean().item(),
            }
        )

        return action, log_prob, value, self.state, log, entropy

    def act(self, obs: Obs, reward: torch.Tensor, done: torch.Tensor):
        """Convenience method for evaluation (learn=False)."""
        return self.step(obs, reward, done, learn=False)
