# Archived Explorations

This folder contains previous approaches and experiments that informed the current hierarchical language model design.

## Folder Structure

### `flat_language_models/`
Early attempts at German language modeling using flat architectures:
- Byte-level encoding (struggled with multi-byte German characters)
- Char-level without hierarchy (learned char statistics but not structure)
- Brain-inspired but non-hierarchical approaches

**Key learnings:** Flat models can minimize loss but don't capture compositional structure needed for meaningful generation.

### `rl_spatial_training/`
Reinforcement learning experiments for spatial reasoning and navigation:
- Vectorized environments
- Curriculum learning approaches
- Language grounding in gridworlds

**Key learnings:** RL works well for discrete action spaces; language needs different treatment.

### `evaluation_scripts/`
Evaluation and diagnostic tools from previous experiments:
- Seed validation
- Model comparison
- Stability verification

### `svg_utilities/`
SVG manipulation scripts for documentation/visualization.

---

## Current Approach

The active development uses a **hierarchical architecture** following human language learning:

```
Characters → Syllables → Morphemes → Words → Phrases → Sentences
```

See `/docs/hierarchical_language_architecture.md` for details.
