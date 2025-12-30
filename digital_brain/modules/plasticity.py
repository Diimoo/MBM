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
        
    def update_trace(self, trace, pre, post, dt=1.0):
        """
        Updates the eligibility trace based on pre- and post-synaptic activity.
        trace: (d_pre, d_post) or batch-wise
        pre: (B, d_pre)
        post: (B, d_post)
        """
        # Hebbian coincidence: pre^T * post
        # For batch processing, we can average over batch or keep batch dim?
        # Weights are shared across batch, so traces should probably be accumulated or averaged?
        # In a biological setting, this happens per synapse. We'll average over batch for weight update.
        
        # Efficient average outer product: (d_pre, B) @ (B, d_post) -> (d_pre, d_post)
        # Instead of torch.bmm which creates (B, d_pre, d_post) and OOMs.
        hebbian_avg = (pre.t() @ post) / pre.shape[0]
        
        # dE = (-E/tau + Hebbian) * dt
        delta_e = (-trace + hebbian_avg) / self.tau_e
        new_trace = trace + delta_e * dt
        
        return new_trace
        
    def compute_delta_w(self, trace, modulator):
        """
        Computes weight change.
        Delta W = lr * trace * modulator
        modulator: scalar signal (e.g., DA - baseline)
        """
        # modulator is (B,). We probably want the mean signal over the batch?
        # Or if the trace is already averaged, we take mean of mod.
        mod_avg = modulator.mean()
        
        delta_w = self.lr * trace * mod_avg
        return delta_w
