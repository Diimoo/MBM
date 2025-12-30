import torch
import torch.nn as nn

class Hippocampus(nn.Module):
    """
    Episodic memory, fast encode/retrieve/replay (MVP).
    Retrieval uses cosine similarity (normalized), so novelty does not collapse.
    """
    def __init__(self, d_z: int, capacity: int = 1000, eps: float = 1e-8):
        super().__init__()
        self.d_z = d_z
        self.capacity = capacity
        self.eps = eps

        self.register_buffer("memory", torch.zeros(capacity, d_z))
        self.ptr = 0
        self.count = 0

    def encode(self, z: torch.Tensor, ctx=None):
        """Vectorized encoding into the memory buffer."""
        B = z.shape[0]
        device = z.device
        z_d = z.detach()
        
        # Determine slice indices
        indices = (torch.arange(self.ptr, self.ptr + B, device=device) % self.capacity).long()
        self.memory[indices] = z_d
        
        self.ptr = (self.ptr + B) % self.capacity
        self.count = min(self.count + B, self.capacity)
        return torch.tensor([self.ptr], device=device)

    def _cosine_sim(self, cue: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        cue_n = cue / (cue.norm(dim=1, keepdim=True) + self.eps)  # (B, d_z)
        mem_n = mem / (mem.norm(dim=1, keepdim=True) + self.eps)  # (N, d_z)
        return cue_n @ mem_n.T                                    # (B, N)

    def retrieve(self, cue: torch.Tensor, topk: int = 1) -> torch.Tensor:
        if self.count == 0:
            return torch.zeros_like(cue)

        mem = self.memory[:self.count]  # (N, d_z)
        sim = self._cosine_sim(cue, mem)
        k = min(topk, self.count)
        idx = torch.topk(sim, k=k, dim=1).indices  # (B, k)
        gathered = mem[idx]                        # (B, k, d_z)

        if k == 1:
            return gathered[:, 0, :]
        return gathered.mean(dim=1)

    def replay(self, n: int = 5) -> torch.Tensor:
        if self.count == 0:
            return torch.empty(0, self.d_z, device=self.memory.device)

        indices = torch.randint(0, self.count, (n,), device=self.memory.device)
        return self.memory[indices]

    def novelty(self, z: torch.Tensor) -> torch.Tensor:
        """
        novelty in [0,1], based on cosine similarity.
        novelty = (1 - max_sim) / 2
        """
        if self.count == 0:
            return torch.ones(z.shape[0], device=z.device)

        mem = self.memory[:self.count]
        sim = self._cosine_sim(z, mem)
        max_sim = torch.max(sim, dim=1).values
        novelty = (1.0 - max_sim) * 0.5
        return torch.clamp(novelty, 0.0, 1.0)
