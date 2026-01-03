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

    def clear(self):
        """Clears the memory buffer."""
        self.memory.zero_()
        self.ptr = 0
        self.count = 0

    def encode(self, z: torch.Tensor, ctx=None, max_write: int = 128):
        """Vectorized encoding of a subset of the batch into the memory buffer."""
        B = z.shape[0]
        device = z.device
        
        # If batch size is larger than capacity or a specified max_write, 
        # we only store a random subset to prevent overwriting the whole buffer in one step.
        num_to_write = min(B, max_write, self.capacity)
        
        if num_to_write < B:
            indices_subset = torch.randperm(B, device=device)[:num_to_write]
            z_to_store = z[indices_subset].detach()
        else:
            z_to_store = z.detach()
            num_to_write = B

        # Determine slice indices in ring buffer
        indices = (torch.arange(self.ptr, self.ptr + num_to_write, device=device) % self.capacity).long()
        self.memory[indices] = z_to_store
        
        self.ptr = (self.ptr + num_to_write) % self.capacity
        self.count = min(self.count + num_to_write, self.capacity)
        return torch.tensor([self.ptr], device=device)

    def _cosine_sim(self, cue: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        # Normalize cue and mem
        cue_n = cue / (cue.norm(dim=1, keepdim=True) + self.eps)  # (B, d_z)
        mem_n = mem / (mem.norm(dim=1, keepdim=True) + self.eps)  # (N, d_z)
        return cue_n @ mem_n.T                                    # (B, N)

    def retrieve(self, cue: torch.Tensor, topk: int = 1, 
                 confidence_threshold: float = 0.0) -> torch.Tensor:
        """
        Retrieve memories with optional confidence thresholding.
        
        Args:
            cue: Query tensor (B, d_z)
            topk: Number of memories to retrieve
            confidence_threshold: If > 0, only retrieve when max similarity 
                                  exceeds this threshold. Returns zeros for 
                                  uncertain cases to prevent stale memory interference.
        """
        if self.count == 0:
            return torch.zeros_like(cue)

        mem = self.memory[:self.count]  # (N, d_z)
        sim = self._cosine_sim(cue, mem)
        max_sim = torch.max(sim, dim=1).values  # (B,)
        
        k = min(topk, self.count)
        idx = torch.topk(sim, k=k, dim=1).indices  # (B, k)
        gathered = mem[idx]                        # (B, k, d_z)

        if k == 1:
            retrieved = gathered[:, 0, :]
        else:
            retrieved = gathered.mean(dim=1)
        
        # Apply confidence thresholding
        if confidence_threshold > 0:
            confident_mask = (max_sim > confidence_threshold).unsqueeze(1)  # (B, 1)
            retrieved = retrieved * confident_mask.float()
        
        return retrieved

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
