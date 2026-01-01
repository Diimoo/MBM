import torch
import torch.nn as nn

class SynapticPlasticity(nn.Module):
    """
    3-Factor Learning Rule:
    Delta W = LearningRate * EligibilityTrace * Neuromodulator
    
    Eligibility Trace (e_t):
    de_t/dt = -e_t/tau_e + Pre * Post (Hebbian term)
    """
    def __init__(self, tau_e: float = 10.0, learning_rate: float = 1e-4):
        super().__init__()
        self.tau_e = tau_e
        self.lr = learning_rate
        
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
        
    def compute_delta_w(self, trace, modulator):
        """
        Computes weight change.
        Delta W = lr * trace * modulator
        modulator: (B,)
        trace: (d_pre, d_post) or (B, d_pre, d_post)
        """
        if trace.dim() == 3:
            # Per-sample trace: (B, d_pre, d_post)
            # Weighted sum: Σ_b (trace[b] * modulator[b]) / B
            # Using einsum for efficiency: b,bi,bj -> ij
            # Actually we want (B, d_pre, d_post) multiplied by (B, 1, 1) then mean over B
            delta_w = self.lr * (trace * modulator.view(-1, 1, 1)).mean(dim=0)
        else:
            # Averaged trace: (d_pre, d_post)
            mod_avg = modulator.mean()
            delta_w = self.lr * trace * mod_avg
        return delta_w
