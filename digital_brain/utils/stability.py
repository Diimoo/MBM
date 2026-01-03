"""
Stability utilities for preventing NaN explosions and runaway dynamics.
"""
import torch


def safe_clamp(tensor, min_val=-20, max_val=20, name="tensor"):
    """Clamp tensor and log if values are extreme or NaN."""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}! Replacing with zeros.")
        return torch.zeros_like(tensor)
    
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}! Clamping.")
        tensor = torch.where(torch.isinf(tensor), torch.sign(tensor) * max_val, tensor)
    
    max_abs = tensor.abs().max().item()
    if max_abs > max_val:
        print(f"⚠️ Clamping {name}: max_abs={max_abs:.2f} -> {max_val}")
    
    return torch.clamp(tensor, min_val, max_val)


def check_nan(tensor, name="tensor", raise_error=True):
    """Check for NaN/Inf and optionally raise error."""
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()
    
    if has_nan or has_inf:
        msg = f"{'NaN' if has_nan else 'Inf'} detected in {name}!"
        msg += f" Stats: mean={tensor.nanmean():.4f}, max_abs={tensor.abs().max():.4f}"
        
        if raise_error:
            raise RuntimeError(msg)
        else:
            print(f"⚠️ {msg}")
            return False
    return True
