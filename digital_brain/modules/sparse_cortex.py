import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .plasticity import SynapticPlasticity

class SparseCorticalMicrocircuit(nn.Module):
    def __init__(self, d_in, d_z, dt=0.1, sparsity=0.01, locality_radius=None):
        super().__init__()
        self.d_z = d_z
        self.dt = dt
        
        # Generate sparse connectivity
        if locality_radius is not None:
            # Spatially local connections (more bio-plausible)
            indices, values = self._local_connectivity(d_z, sparsity, locality_radius)
        else:
            # Random sparse connections
            indices, values = self._random_connectivity(d_z, sparsity)
        
        # Store as COO sparse tensor components
        self.register_buffer('W_ee_indices', indices)
        self.W_ee_values = nn.Parameter(values)
        
        # Dense weights (small or non-recurrent, keep dense for performance at this scale)
        self.W_in = nn.Parameter(torch.randn(d_in, d_z) * 0.1)
        self.W_ei = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1)
        self.W_ie = nn.Parameter(torch.abs(torch.randn(d_z, d_z)) * 0.1)
        
        # Eligibility trace (sparse, same structure as W_ee)
        # We'll store this in the BrainState during forward pass, not as a parameter
        
        self.tau_e = 1.0
        self.tau_i = 0.5
        
        # Plasticity Rule
        self.plasticity = SynapticPlasticity(tau_e=50.0, learning_rate=1e-3)

    def _random_connectivity(self, n, sparsity):
        """Random sparse connectivity."""
        n_connections = int(n * n * sparsity)
        src = torch.randint(0, n, (n_connections,))
        dst = torch.randint(0, n, (n_connections,))
        indices = torch.stack([src, dst])
        values = torch.randn(n_connections) * 0.1
        return indices, values

    def _local_connectivity(self, n, sparsity, radius):
        """Spatially local connections (assumes 2D grid layout)."""
        grid_size = int(np.sqrt(n))
        if grid_size * grid_size != n:
            return self._random_connectivity(n, sparsity)
        
        indices = []
        # Adjusted sparsity to account for the local neighborhood size
        # Neighborhood size is approx (2*radius+1)^2
        # To maintain overall sparsity S, local probability should be S * (N / neighborhood_size)
        neighborhood_size = (2 * radius + 1) ** 2
        local_prob = min(1.0, sparsity * (n / neighborhood_size))
        
        for i in range(n):
            i_x, i_y = i // grid_size, i % grid_size
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    j_x, j_y = i_x + dx, i_y + dy
                    if 0 <= j_x < grid_size and 0 <= j_y < grid_size:
                        j = j_x * grid_size + j_y
                        if np.random.rand() < local_prob:
                            indices.append([i, j])
        
        if not indices:
            return self._random_connectivity(n, sparsity)
            
        indices = torch.tensor(indices).t()
        values = torch.randn(indices.shape[1]) * 0.1
        return indices, values

    def get_W_ee_sparse(self):
        """Get sparse W_ee tensor."""
        return torch.sparse_coo_tensor(
            self.W_ee_indices,
            self.W_ee_values,
            (self.d_z, self.d_z)
        )

    def forward(self, x, state, *, update_trace: bool = True):
        B = x.shape[0]
        if state is None:
            e_act = torch.zeros(B, self.d_z, device=x.device)
            i_act = torch.zeros(B, self.d_z, device=x.device)
            trace_ee = torch.zeros(self.W_ee_values.shape[0], device=x.device)
        else:
            e_act, i_act, trace_ee = state
        
        # External drive
        ext_drive = x @ self.W_in
        
        # Recurrent drive (SPARSE MATMUL)
        # Note: torch.sparse.mm expects (sparse, dense) -> dense.
        # Workaround: PyTorch sparse mm on CUDA has limited dtype support (often missing BFloat16/Half).
        # We perform this specific operation in Float32 to ensure stability and compatibility.
        W_ee_sparse = self.get_W_ee_sparse()
        
        # Capture original dtype to restore after operation
        orig_dtype = e_act.dtype
        
        # Always use Float32 for sparse mm on CUDA to avoid "not implemented" errors
        # We explicitly disable autocast for this operation to prevent it from forcing BFloat16 back.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            rec_drive = torch.sparse.mm(W_ee_sparse.float(), e_act.t().float()).t().to(orig_dtype)
        
        # Inhibitory drive
        inh_drive = i_act @ self.W_ie
        
        # Dynamics
        de = (-e_act + F.relu(ext_drive + rec_drive - inh_drive)) / self.tau_e
        di = (-i_act + F.relu(e_act @ self.W_ei)) / self.tau_i
        
        e_act_new = e_act + self.dt * de
        i_act_new = i_act + self.dt * di
        
        # Eligibility trace update (SPARSE)
        if update_trace:
            trace_ee_new = self._update_sparse_trace(trace_ee, e_act, e_act_new)
        else:
            trace_ee_new = trace_ee
            
        return e_act_new, (e_act_new, i_act_new, trace_ee_new)

    def _update_sparse_trace(self, trace, e_act_old, e_act_new):
        """Update eligibility trace for sparse connections only."""
        src_idx = self.W_ee_indices[0]
        dst_idx = self.W_ee_indices[1]
        
        # Hebbian term: pre × post (averaged over batch)
        pre_act = e_act_old[:, src_idx]   # (B, n_conn)
        post_act = e_act_new[:, dst_idx]  # (B, n_conn)
        hebbian = (pre_act * post_act).mean(dim=0) # (n_conn,)
        
        # Trace dynamics
        delta_e = (-trace + hebbian) / self.tau_e
        trace_new = trace + self.dt * delta_e
        
        return trace_new

    def apply_plasticity(self, mod_signals, state):
        """Apply 3-factor rule to sparse connections."""
        if state is None or len(state) < 3:
            return
            
        _, _, trace_ee = state
        da = mod_signals.DA # (B,)
        
        # Update only the sparse connections
        # trace_ee: (n_conn,), da: (B,)
        # delta_w: (n_conn,)
        delta_w = self.plasticity.compute_delta_w(trace_ee, da)
        
        with torch.no_grad():
            self.W_ee_values.add_(delta_w)
            # Weight clamping for stability (Priority 1)
            self.W_ee_values.clamp_(-1.0, 1.0) 
