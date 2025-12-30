import torch
import torch.nn as nn
from ..datatypes import ModSignals

class Thalamus(nn.Module):
    """
    Routing/gating for inputs.
    Reference: BESCHREIBUNG_v1.1.md Section 5
    """
    def __init__(self, d_obs, d_sel):
        super().__init__()
        self.gate_fc = nn.Linear(d_sel, d_obs)
        # Initialize bias to 2.0 so sigmoid starts near 1.0 (gates open)
        nn.init.constant_(self.gate_fc.bias, 2.0)
        self.alpha_ach = 0.5
        self.alpha_ne = 0.5

    def gate(self, inputs: torch.Tensor, selection: torch.Tensor, mods: ModSignals) -> torch.Tensor:
        """
        gate = sigmoid(Ws*selection + Wf*feedback - b)
        gain = 1 + α_ACh*ACh + α_NE*NE
        gated = inputs * gate * gain
        """
        gate = torch.sigmoid(self.gate_fc(selection))
        gain = 1.0 + self.alpha_ach * mods.ACh + self.alpha_ne * mods.NE
        
        # Reshape mods to broadcast
        gain = gain.view(-1, 1)
        
        return inputs * gate * gain
