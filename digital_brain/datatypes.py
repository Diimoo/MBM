import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

@dataclass
class Obs:
    x: torch.Tensor  # (B, d_obs)
    ctx: Optional[torch.Tensor] = None  # (B, d_ctx)

@dataclass
class ModSignals:
    DA: torch.Tensor  # Dopamine (RPE)
    NE: torch.Tensor  # Noradrenaline (Novelty/Surprise)
    ACh: torch.Tensor # Acetylcholine (Attention/Gain)
    HT5: torch.Tensor # Serotonin (Stability/Horizon)

@dataclass
class Selection:
    sel: torch.Tensor  # (B, d_sel) - gating selection
    commit: torch.Tensor # (B, 1) - action commitment

@dataclass
class BrainState:
    z: torch.Tensor  # (B, d_z) - latent representation
    cortex_state: Any
    bg_state: Any
    hip_state: Any
    cerebellum_state: Any = None

@dataclass
class StepLog:
    pred_error: float
    rpe: float
    novelty: float
    gate_stats: Dict[str, float]
    mod_signals: Dict[str, float]
