# Final Experimental Results (Jan 2026)

This summary is based on the final validation experiments run on January 3rd, 2026.
**Source:** `experiments/decisive_validation_20260103_171140.json`

---

## Multi-Seed Validation (N=5)

### Overall Catastrophic Forgetting Index (CFI)

- **MBM (stabilized):**
  - **μ (mean): -0.07**
  - **σ (std dev): 0.43**
- **PPO (baseline):**
  - **μ (mean): -0.12**
  - **σ (std dev): 0.22**

**Note:** The standard deviation of MBM's CFI is approximately **2x higher** than PPO's, indicating greater instability. The "6x higher variance" figure from older documents is outdated.

---

## Task-Dependent Performance (Mean CFI)

This breakdown shows how performance changes based on the similarity of the tasks in the continual learning sequence.

| Task Transition | Task Similarity | MBM (stabilized) | PPO (baseline) | Winner |
| :--- | :--- | :--- | :--- | :--- |
| `5x5 -> 7x7` | **High** | **-0.44** (Strong Improvement) | -0.07 | **MBM** |
| `5x5 -> 10x10`| Medium | +0.09 (Slight Forgetting) | **-0.02** | **PPO** |
| `7x7 -> 10x10`| Low | +0.13 (Forgetting) | **-0.26** (Improvement) | **PPO** |

### Key Insights:

1.  **MBM excels on similar tasks:** The high plasticity of MBM leads to significant "backward transfer" (performance improvement on the old task) when the new task is similar.
2.  **PPO is more stable:** PPO's stability prevents catastrophic forgetting and even facilitates moderate performance gains across dissimilar tasks.
3.  **The Tradeoff is Clear:** MBM's high-risk, high-reward plasticity is only beneficial in specific, high-similarity scenarios. In all other cases, PPO's stability is superior.
