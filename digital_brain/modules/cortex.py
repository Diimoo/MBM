import torch
import torch.nn as nn
import torch.nn.functional as F
from .plasticity import SynapticPlasticity
from .sparse_cortex import SparseCorticalMicrocircuit

class CorticalMicrocircuit(nn.Module):
    """
    Biological Cortical Microcircuit (Rate-based E/I populations).
    Reference: BESCHREIBUNG_v1.1.md Section 4.3
    
    Structure:
    - E: Excitatory Pyramidal cells (80%)
    - I_fast: Fast inhibition (PV+ like, local gain control)
    - I_slow: Slow inhibition (SST+ like, lateral/divisive)
    
    Plasticity:
    - W_ee is plastic via 3-Factor Hebbian Rule.
    """
    def __init__(self, d_in, d_z, dt=0.1):
        super().__init__()
        self.d_z = d_z
        self.dt = dt
        
        # Weights
        self.W_ei = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1) # E -> I
        self.W_ie = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1) # I -> E
        self.W_ee = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1) # E -> E (recurrent, plastic)
        self.W_in = nn.Parameter(torch.randn(d_in, d_z) * 0.1)          # Input -> E
        
        # Time constants
        self.tau_e = 1.0   # 10-20ms
        self.tau_i = 0.5   # 5-10ms (fast)
        
        # Plasticity Rule
        self.plasticity = SynapticPlasticity(tau_e=50.0, learning_rate=1e-3)
        
    def forward(self, x, state=None, *, update_trace: bool = True):
        """
        Euler integration + Trace Update.
        x: (B, d_in)
        state: tuple(e_act, i_act, trace_ee)
          - trace_ee: (d_z, d_z) eligibility trace for W_ee
        update_trace: if False, skip expensive Hebbian trace computation (2*B*n² FLOPs)
        """
        B = x.shape[0]
        if state is None:
            e_act = torch.zeros(B, self.d_z, device=x.device)
            i_act = torch.zeros(B, self.d_z, device=x.device)
            trace_ee = torch.zeros(self.d_z, self.d_z, device=x.device)
        else:
            if len(state) == 2: # Legacy support
                e_act, i_act = state
                trace_ee = torch.zeros(self.d_z, self.d_z, device=x.device)
            else:
                e_act, i_act, trace_ee = state
            
        # 1. Compute inputs
        ext_drive = x @ self.W_in
        rec_drive = e_act @ self.W_ee
        inh_drive = i_act @ self.W_ie
        
        # 2. Dynamics
        de = (-e_act + F.relu(ext_drive + rec_drive - inh_drive)) / self.tau_e
        di = (-i_act + F.relu(e_act @ self.W_ei)) / self.tau_i
        
        e_act_new = e_act + self.dt * de
        i_act_new = i_act + self.dt * di
        
        # Activity clamping for stability (Priority 1)
        e_act_new = torch.clamp(e_act_new, 0.0, 50.0)
        i_act_new = torch.clamp(i_act_new, 0.0, 50.0)
        
        # 3. Plasticity: Update Eligibility Trace for W_ee (only when learning)
        # Skip when learn=False to save ~2*B*n² FLOPs per step
        if update_trace:
            trace_ee_new = self.plasticity.update_trace(trace_ee, e_act, e_act_new)
        else:
            trace_ee_new = trace_ee
        
        # No weight update here! Weights are updated by neuromodulators later.
        
        return e_act_new, (e_act_new, i_act_new, trace_ee_new)

    def apply_plasticity(self, mod_signals, state):
        """
        Apply global neuromodulation to plastic weights.
        Delta W = lr * Trace * Modulator
        """
        if state is None or len(state) < 3:
            return
            
        _, _, trace_ee = state
        
        # Use DA signal for reward-based learning
        da = mod_signals.DA
        
        # Delta W
        delta_w = self.plasticity.compute_delta_w(trace_ee, da)
        
        # Update weights (in-place)
        with torch.no_grad():
            self.W_ee.add_(delta_w)
            # Weight clamping for stability (Priority 1)
            self.W_ee.clamp_(-5.0, 5.0)

class Cortex(nn.Module):
    """
    World model using Cortical Microcircuits with Plasticity.
    """
    def __init__(self, d_obs, d_z, d_act, sparse=False, sparsity=0.01, locality_radius=None):
        super().__init__()
        self.d_z = d_z
        if sparse:
            self.microcircuit = SparseCorticalMicrocircuit(d_obs, d_z, sparsity=sparsity, locality_radius=locality_radius)
        else:
            self.microcircuit = CorticalMicrocircuit(d_obs, d_z)
        self.pred_head = nn.Linear(d_z, d_obs)
        
    def forward(self, x, state, *, update_trace: bool = True):
        z_t, new_state = self.microcircuit(x, state, update_trace=update_trace)
        pred_t = self.pred_head(z_t)
        return z_t, pred_t, new_state
        
    def update_weights(self, mod_signals, state):
        self.microcircuit.apply_plasticity(mod_signals, state)

    def predict(self, z_t):
        return self.pred_head(z_t)
