import torch
import torch.nn as nn

class Cerebellum(nn.Module):
    """
    Residual correction + timing.
    Reference: BESCHREIBUNG_v1.1.md Section 8
    """
    def __init__(self, d_z, d_obs, d_act):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_z + d_obs, d_z),
            nn.ReLU(),
            nn.Linear(d_z, d_act)
        )

    def forward(self, plan: torch.Tensor, sensory: torch.Tensor, error_signal: torch.Tensor = None):
        """
        Cerebellum.forward(plan, sensory, error_signal) -> (correction, timing_offset)
        """
        x = torch.cat([plan, sensory], dim=-1)
        correction = self.fc(x)
        timing_offset = torch.zeros(plan.shape[0], 1, device=plan.device)
        return correction, timing_offset
