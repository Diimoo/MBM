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
    def __init__(self, d_in, d_z, dt=0.1, use_norm=True):
        super().__init__()
        self.d_z = d_z
        self.dt = dt
        self.use_norm = use_norm
        
        # Weights
        self.W_ei = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1) # E -> I
        self.W_ie = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1) # I -> E
        self.W_ee = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1) # E -> E (recurrent, plastic)
        self.W_in = nn.Parameter(torch.randn(d_in, d_z) * 0.1)          # Input -> E
        
        # Normalization
        if use_norm:
            self.norm = nn.LayerNorm(d_z)
        
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
        total_drive = ext_drive + rec_drive - inh_drive
        if self.use_norm:
            total_drive = self.norm(total_drive)
            
        de = (-e_act + F.relu(total_drive)) / self.tau_e
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

class HierarchicalCortex(nn.Module):
    """
    Multi-layer cortical hierarchy.
    Bottom-up flow: Input -> L1 -> L2 -> ... -> Ln -> Output
    Supports residual connections and layer normalization.
    """
    def __init__(self, d_in, layer_sizes, d_obs, sparse=False, sparsity=0.01, locality_radius=None, use_norm=True, use_residual=True):
        super().__init__()
        self.layers = nn.ModuleList()
        self.use_residual = use_residual
        
        current_in = d_in
        for i, sz in enumerate(layer_sizes):
            if sparse:
                layer = SparseCorticalMicrocircuit(current_in, sz, sparsity=sparsity, locality_radius=locality_radius)
            else:
                layer = CorticalMicrocircuit(current_in, sz, use_norm=use_norm)
            self.layers.append(layer)
            current_in = sz
            
        self.pred_head = nn.Linear(layer_sizes[-1], d_obs)
        self.d_z = layer_sizes[-1]

    def forward(self, x, states, *, update_trace: bool = True):
        """
        states: List of (e_act, i_act, trace_ee) for each layer
        """
        new_states = []
        current_input = x
        
        for i, layer in enumerate(self.layers):
            prev_input = current_input
            current_input, layer_state = layer(current_input, states[i], update_trace=update_trace)
            
            # Residual connection (if dimensions match)
            if self.use_residual and current_input.shape == prev_input.shape:
                current_input = current_input + prev_input
                
            new_states.append(layer_state)
            
        z_t = current_input
        pred_t = self.pred_head(z_t)
        return z_t, pred_t, new_states

    def apply_plasticity(self, mod_signals, states):
        for i, layer in enumerate(self.layers):
            layer.apply_plasticity(mod_signals, states[i])

class Cortex(nn.Module):
    """
    World model using Cortical Microcircuits with Plasticity.
    Supports single-layer or multi-layer (hierarchical) architectures.
    """
    def __init__(self, d_obs, d_z, d_act, sparse=False, sparsity=0.01, locality_radius=None, layer_sizes=None, use_norm=True, use_residual=True):
        super().__init__()
        if layer_sizes is not None:
            self.microcircuit = HierarchicalCortex(d_obs, layer_sizes, d_obs, sparse=sparse, sparsity=sparsity, locality_radius=locality_radius, use_norm=use_norm, use_residual=use_residual)
            self.d_z = self.microcircuit.d_z
        else:
            self.d_z = d_z
            if sparse:
                self.microcircuit = SparseCorticalMicrocircuit(d_obs, d_z, sparsity=sparsity, locality_radius=locality_radius)
            else:
                self.microcircuit = CorticalMicrocircuit(d_obs, d_z, use_norm=use_norm)
        
        # In Hierarchical mode, microcircuit already has pred_head
        if not isinstance(self.microcircuit, HierarchicalCortex):
            self.pred_head = nn.Linear(self.d_z, d_obs)
        
    def forward(self, x, state, *, update_trace: bool = True):
        if isinstance(self.microcircuit, HierarchicalCortex):
            return self.microcircuit(x, state, update_trace=update_trace)
        
        z_t, new_state = self.microcircuit(x, state, update_trace=update_trace)
        pred_t = self.pred_head(z_t)
        return z_t, pred_t, new_state
        
    def update_weights(self, mod_signals, state):
        if isinstance(self.microcircuit, HierarchicalCortex):
            self.microcircuit.apply_plasticity(mod_signals, state)
        else:
            self.microcircuit.apply_plasticity(mod_signals, state)

    def predict(self, z_t):
        if isinstance(self.microcircuit, HierarchicalCortex):
            return self.microcircuit.pred_head(z_t)
        return self.pred_head(z_t)
