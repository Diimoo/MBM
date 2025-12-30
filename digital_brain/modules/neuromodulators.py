import torch
import torch.nn as nn
from ..datatypes import ModSignals

class Neuromodulators(nn.Module):
    """
    MVP rules:
    - DA: TD-RPE from BG
    - NE: novelty from Hippocampus
    - ACh: attention proxy from surprise/pred_error
    - 5HT: placeholder constant for now
    """
    def __init__(self):
        super().__init__()

    def compute(
        self,
        z_t: torch.Tensor,
        da_rpe: torch.Tensor,
        novelty: torch.Tensor,
        pred_error: torch.Tensor | None = None
    ) -> ModSignals:
        B = z_t.shape[0]
        device = z_t.device

        da = da_rpe
        ne = novelty

        if pred_error is None:
            ach = torch.full((B,), 0.5, device=device)
        else:
            ach = pred_error / (pred_error + 1.0)
            ach = torch.clamp(ach, 0.0, 1.0)

        ht5 = torch.full((B,), 0.5, device=device)

        return ModSignals(DA=da, NE=ne, ACh=ach, HT5=ht5)
