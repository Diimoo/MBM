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
        self.policy_head = nn.Linear(d_z, d_act)
        # Memory-augmented policy head (Priority 5)
        self.memory_head = nn.Linear(d_z, d_act)
        self.gamma = 0.99

    def step(self, z_t, reward, ctx, done, prev_value=None, memory_context=None, cerebellum_correction=None, action_to_eval=None):
        """
        BG.step(z_t, reward_t, ctx_t, done_t) -> (selection, DA_signal, action, action_log_prob, value, entropy)
        """
        # Memory-Augmented Policy: Concatenate retrieved memory if provided
        if memory_context is not None:
            z_input = torch.cat([z_t, memory_context], dim=-1)
        else:
            z_input = z_t

        value = self.value_head(z_t) # Value still uses core state
        selection = self.selection_head(z_t)
        
        # Policy uses core state + memory context influence
        logits = self.policy_head(z_t)
        if memory_context is not None:
            logits = logits + self.memory_head(memory_context)
            
        # Cerebellum integration: Bias logits with residual correction
        if cerebellum_correction is not None:
            logits = logits + cerebellum_correction
            
        logits = torch.clamp(logits, min=-20, max=20) # Prevent NaN explosion
        
        dist = torch.distributions.Categorical(logits=logits)
        
        if action_to_eval is not None:
            action = action_to_eval
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        # DA = RPE = r + gamma * V(s') - V(s)
        da = torch.zeros_like(value)
        if prev_value is not None:
            # Clamp reward/values to help stability
            da = reward + (1 - done.float()) * self.gamma * value - prev_value
            da = torch.clamp(da, min=-10, max=10)
            
        return selection, da, action, log_prob, value, entropy
