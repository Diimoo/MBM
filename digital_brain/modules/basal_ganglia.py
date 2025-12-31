import torch
import torch.nn as nn
from ..datatypes import Selection

class BasalGanglia(nn.Module):
    """
    Selection/policy, TD-RPE (DA).
    Reference: BESCHREIBUNG_v1.1.md Section 6
    """
    def __init__(self, d_z, d_sel, d_act):
        super().__init__()
        self.value_head = nn.Linear(d_z, 1)
        self.selection_head = nn.Linear(d_z, d_sel)
        # Policy takes z_t + selection -> gives selection_head direct RL gradient
        self.policy_head = nn.Linear(d_z + d_sel, d_act)
        self.gamma = 0.99
        self.d_z = d_z
        self.d_sel = d_sel

    def step(self, z_t, reward, ctx, done, prev_value=None):
        """
        BG.step(z_t, reward_t, ctx_t, done_t) -> (selection, DA_signal, action, action_log_prob, value, entropy)
        """
        value = self.value_head(z_t)
        selection = self.selection_head(z_t) # Removed tanh to allow full range in Thalamus sigmoid
        
        # Concatenate z_t + selection for policy -> selection_head gets RL gradient
        policy_input = torch.cat([z_t, selection], dim=-1)
        logits = self.policy_head(policy_input)
        # Numerical stability: clamp logits
        logits = torch.clamp(logits, min=-20, max=20)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs, validate_args=False)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # DA = RPE = r + gamma * V(s') - V(s)
        da = torch.zeros_like(value)
        if prev_value is not None:
            da = reward + (1 - done.float()) * self.gamma * value - prev_value
            
        return selection, da, action, log_prob, value, entropy
