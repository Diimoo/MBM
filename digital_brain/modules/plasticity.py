import torch
import torch.nn as nn

class SynapticPlasticity(nn.Module):
    """
    3-Factor Learning Rule with Homeostatic Regulation:
    Delta W = AdaptiveLR * EligibilityTrace * Neuromodulator
    
    Eligibility Trace (e_t):
    de_t/dt = -e_t/tau_e + Pre * Post (Hebbian term)
    
    Stabilization features:
    - Adaptive learning rate based on weight magnitude
    - Soft weight clipping (tanh-based)
    - Metaplasticity (history-based LR scaling)
    """
    def __init__(self, tau_e: float = 10.0, learning_rate: float = 1e-4, 
                 w_max: float = 3.0, metaplasticity_window: int = 100):
        super().__init__()
        self.tau_e = tau_e
        self.lr = learning_rate
        self.base_lr = learning_rate
        self.w_max = w_max
        
        # Metaplasticity: track recent plasticity magnitude
        self.register_buffer('plasticity_history', torch.zeros(metaplasticity_window))
        self.history_ptr = 0
        self.metaplasticity_window = metaplasticity_window
        
    def update_trace(self, trace, pre, post, dt=1.0, average_batch=True):
        """
        Updates the eligibility trace based on pre- and post-synaptic activity.
        trace: (d_pre, d_post) or (B, d_pre, d_post)
        pre: (B, d_pre)
        post: (B, d_post)
        average_batch: if True, averages across batch (saves memory).
        """
        if average_batch:
            # Efficient average outer product: (d_pre, B) @ (B, d_post) -> (d_pre, d_post)
            hebbian = (pre.t() @ post) / pre.shape[0]
        else:
            # Per-sample outer product: (B, d_pre, 1) * (B, 1, d_post) -> (B, d_pre, d_post)
            hebbian = torch.bmm(pre.unsqueeze(2), post.unsqueeze(1))
        
        # dE = (-E/tau + Hebbian) * dt
        delta_e = (-trace + hebbian) / self.tau_e
        new_trace = trace + delta_e * dt
        
        return new_trace
        
    def compute_delta_w(self, trace, modulator, current_weights=None):
        """
        Computes weight change with homeostatic regulation.
        Delta W = adaptive_lr * trace * modulator
        modulator: (B,)
        trace: (d_pre, d_post) or (B, d_pre, d_post)
        current_weights: optional (d_pre, d_post) for adaptive LR
        """
        # Compute adaptive learning rate based on weight magnitude
        effective_lr = self.lr
        if current_weights is not None:
            w_norm = torch.norm(current_weights)
            # Reduce LR as weights approach max
            lr_scale = torch.clamp(1.0 - (w_norm / (self.w_max * current_weights.numel() ** 0.5)), min=0.1, max=1.0)
            effective_lr = self.lr * lr_scale.item()
        
        # Apply metaplasticity scaling
        avg_plasticity = self.plasticity_history.mean()
        if avg_plasticity > 0.1:  # Threshold for scaling
            meta_scale = 1.0 / (1.0 + avg_plasticity)
            effective_lr = effective_lr * meta_scale
        
        if trace.dim() == 3:
            # Per-sample trace: (B, d_pre, d_post)
            delta_w = effective_lr * (trace * modulator.view(-1, 1, 1)).mean(dim=0)
        else:
            # Averaged trace: (d_pre, d_post)
            mod_avg = modulator.mean()
            delta_w = effective_lr * trace * mod_avg
        
        # HARD CLAMP delta_w to prevent runaway (nuclear option)
        delta_w = torch.clamp(delta_w, min=-0.1, max=0.1)
        
        # Track plasticity magnitude for metaplasticity
        delta_w_norm = torch.norm(delta_w).item()
        self.plasticity_history[self.history_ptr] = delta_w_norm
        self.history_ptr = (self.history_ptr + 1) % self.metaplasticity_window
        
        return delta_w
    
    def apply_soft_clipping(self, weights):
        """
        Apply soft (differentiable) weight clipping using tanh.
        Prevents hard discontinuities while bounding weights.
        """
        return torch.tanh(weights / self.w_max) * self.w_max
    
    def reset_metaplasticity(self):
        """Reset plasticity history (e.g., at task boundaries)."""
        self.plasticity_history.zero_()
        self.history_ptr = 0
