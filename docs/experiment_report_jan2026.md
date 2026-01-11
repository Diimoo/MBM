# Hierarchical German Language Model - Experiment Report

**Date:** January 9, 2026  
**Status:** Phase 8 Complete - Neuro-modulator Expansion

---

## 1. Objective

Develop a hierarchical language model for German that learns language the way humans do:

- Characters → Syllables → Morphemes → Words → Phrases → Sentences → Discourse
- Each level explicitly composes the previous level
- Integrate brain-inspired modules: Dopamine (reward), Hebbian (association), Hippocampus (memory), Acetylcholine (uncertainty)

---

## 2. Experiments Conducted

### Experiment 1: Character Encoder + Syllable Detector (v1)

**File:** `hierarchical_german_v1.py`

**Results:**

- ✅ Syllable boundary detection worked well
- ✅ Model learned German syllable patterns
- Loss converged to ~0.03

---

### Experiment 2: Morpheme Parser (v2)

**File:** `hierarchical_german_v2.py`

**Results:**

- ✅ Morpheme boundary detection worked
- ✅ Type classification (Prefix, Root, Suffix) succeeded
- ✅ Embeddings showed differentiation (similarity 0.79-0.98)

---

### Experiment 3: Word Composer (v3)

**File:** `hierarchical_german_v3.py`

**Results:**

- ✅ Word boundary detection worked
- ⚠️ Word embeddings started showing high similarity (early signs of collapse)

---

### Experiment 4: Phrase Chunker + Sentence Encoder (v4)

**File:** `hierarchical_german_v4.py`

**Results:**

- ✅ Phrase boundaries detected
- ❌ **Sentence embeddings collapsed** - all inputs produced identical vectors

---

### Experiment 5: Brain Integration (v1)

**File:** `hierarchical_german_brain.py`

**Results:**

- ❌ **Complete embedding collapse**
- ❌ Reward hit 100% immediately (overfitting)

---

### Experiment 6: Tabula Rasa Model (RECOVERY SUCCESS)

**File:** `hierarchical_german_tabula_rasa.py`

**Approach:**

- Added contrastive loss (weight 5.0)
- Multiple reconstruction decoders
- Strict collapse monitoring

**Results:**

- ✅ **Embeddings differentiated** (avg sim 0.25)
- ✅ 100% Capital preservation
- ✅ 87% Perfect reconstruction

---

### Experiment 7: Question & Answer (Phase 6)

**File:** `hierarchical_german_phase6_qa.py`

**Approach:**

- Trained on 100,000 unique German Q&A pairs
- Multi-task: Question detection, Answer generation, Question generation

**Results:**

- ✅ **100% Question/Statement Detection Accuracy**
- ✅ Coherent German answers with zero <UNK> tokens
- ✅ Balanced distribution across 7 question types

---

### Experiment 8: Acetylcholine Integration (Phase 7)

**File:** `hierarchical_german_phase7.py`

**Approach:**

- Added **Acetylcholine (ACh)** module for hierarchy modulation
- Modulates weights between bottom-up (sensory) and top-down (context) based on uncertainty

**Results:**

- ✅ **Improved Inference Latency:** ~41.8ms
- ✅ **Maintained 100% Detection Accuracy**
- ✅ Enhanced training stability under high loss variance

---

### Experiment 9: Discourse Level & Narrative (Phase 8)

**File:** `hierarchical_german_phase8.py`

**Approach:**

- Added **DiscourseComposer** (Level 6) using a GRUCell and Attention Memory Buffer.
- Mixed training on 100k Q&A pairs and 50k multi-sentence complex narratives.
- Enhanced ACh system using loss variance for dynamic weighting and precision signaling.

**Results:**

- ✅ **Context Persistence:** Model maintains narrative state across multi-turn interactions.
- ✅ **Zero Regression:** Maintained 100% accuracy on basic Q&A tasks.
- ✅ **Interactive Learning:** Model updates internal Hebbian associations in real-time from user feedback.

---

## 3. Root Cause Analysis (Historical)

### Why Embedding Collapse Occurred:

1. **Character prediction is too easy** -> Fixed with **Reconstruction Loss**
2. **Mean pooling destroys info** -> Fixed with **Attention Pooling**
3. **No diversity signal** -> Fixed with **Contrastive Loss**
4. **Deep layers wash out signal** -> Fixed with **Residual connections & Hierarchy specific decoders**

---

## 4. Lessons Learned

### Architecture:

- Keep lower levels simple (CNN, BiLSTM)
- Use explicit boundary labels
- **Brain modules** must modulate learning, not just generate signal
- **Discourse state** provides necessary context for resolving ambiguities across sentences.

---

## 5. Model Archive

| Model | File | Status |
| :--- | :--- | :--- |
| Phase 1 | `hierarchical_german_tabula_rasa.py` | L0-L3 Hierarchy |
| Phase 5 | `hierarchical_german_phase5.py` | Full Brain Integration |
| Phase 6 | `hierarchical_german_phase6_qa.py` | 100k Q&A Balanced |
| Phase 7 | `hierarchical_german_phase7.py` | ACh Modulation |
| Phase 8 | `hierarchical_german_phase8.py` | Discourse Integration |

**Current Best Checkpoint:** `checkpoints/phase8_discourse_best.pth`

---

*Last updated: January 9, 2026*
