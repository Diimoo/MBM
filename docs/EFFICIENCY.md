# Efficiency Analysis: MBM vs. Transformer (Rigorous Comparison)

## 1. Scope and Context

This analysis compares the Multimodal Brain Model (MBM) to the Transformer architecture specifically for **embodied agents in long-horizon reinforcement learning**. Transformers are superior for static sequence processing (LLMs), while MBM targets the niche of infinite-horizon, memory-constrained agency.

## 2. Temporal Complexity ($L$ = Sequence Length)

| Metric | Transformer (Standard) | MBM (Sparse) |
| :--- | :--- | :--- |
| **Training (per Sequence)** | $O(L^2 \cdot D)$ (Sequential) / $O(L \cdot D)$ (Parallel)* | $O(L \cdot N)$ (Sequential Only) |
| **Inference (per Step)** | $O(D^2)$ (with KV Cache) | $O(N)$ |

*\*Transformers parallelize training across the sequence dimension ($L$), making them significantly faster to train on current hardware for short-to-medium sequences. MBM's recurrence requires sequential unrolling.*

### The Training Sequence Scaling

MBM's primary training advantage is for **ultra-long sequences** ($10^6+$ steps) where Transformer $O(L^2)$ attention becomes computationally prohibitive. However, for most common sequence lengths, the Transformer's ability to parallelize over time makes it the faster training architecture.

## 3. Memory Footprint: The "Constant Memory" Advantage

MBM's most defensible efficiency win is its **constant memory footprint** during inference, regardless of how long the agent has been running.

| Sequence Length ($L$) | Transformer KV Cache ($D=768$) | MBM State ($N=16K, C=1K$) |
| :--- | :--- | :--- |
| 1,000 steps | ~6.1 MB | **~70 KB** |
| 100,000 steps | ~614 MB | **~70 KB** |
| 1,000,000 steps | **~6.1 GB** | **~70 KB** |

**Analysis**: After 1 million steps, MBM is **~87,000x** more memory-efficient. This is critical for robotics and lifelong learning where agents must run for months without memory exhaustion or context resets.

## 4. Parameter Efficiency vs. Capacity

MBM achieves $O(N)$ parameter scaling through 1-5% sparse connectivity.

- **Comparison**: A 16K-neuron sparse MBM (~2.6M params) is much smaller than a standard Transformer (~100M+ params).
- **Caveat**: Capacity is not equal. A single-layer MBM does not match the expressive depth of a 12-layer Transformer. To achieve comparable "intelligence," MBM would likely require hierarchical (multi-layer) structures. Even then, the sparse $O(N)$ scaling remains a long-term advantage for biological-scale models ($10^6$ neurons).

## 5. Summary of Claims

| Claim | Status | Caveat |
| :--- | :--- | :--- |
| **O(L) Sequence Scaling** | ✅ **True** | Only beats Transformers for ultra-long training sequences. |
| **Constant Memory** | ✅ **True** | Groundbreaking for robotics and infinite-horizon agency. |
| **Inference Speed** | ⚠️ **Conditional** | Only faster if MBM neuron count $N$ is smaller than Transformer dimension $D^2$. |
| **Training Speed** | ❌ **False** | Transformers are faster due to sequence-level parallelism. |
| **Online Learning** | 🔥 **Potential** | 3-factor plasticity allows weight updates during inference; avoids replay buffers. |
| **Neuromorphic Fit** | 🚀 **Future** | Naturally maps to event-driven hardware like Intel Loihi. |

## 6. Conclusion

MBM is not a "Transformer replacement" for general sequence tasks. It is a specialized architecture for **embodied agents requiring lifelong learning with bounded memory**. Its strengths lie in memory stability and the potential for pure online learning without the massive data/compute requirements of Transformer-based replay.
