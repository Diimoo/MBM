# Hierarchical German Language Model - Next Steps Roadmap

**Date:** January 8, 2026  
**Status:** Phase 5 Complete - Full Brain Integration + Generation  
**Current Model:** `hierarchical_german_phase5.py`

---

## Current State Summary

### What Works (Validated)

| Component | Status | Evidence |
|-----------|--------|----------|
| Level 0: Character Encoder | ✅ Working | 128-dim embeddings, position encoding |
| Level 1: Syllable Detector | ✅ Working | Correct German syllabification |
| Level 2: Morpheme Parser | ✅ Working | Prefix/root/suffix boundaries |
| Level 3: Word Composer | ✅ Working | POS tagging, word boundaries |
| Level 4: Phrase Chunker | ✅ Working | BiLSTM + attention pooling |
| Level 5: Sentence Encoder | ✅ Working | Transformer + CLS token |
| Capital Preservation | ✅ Working | 100% on 1000 sentence test |
| Embedding Differentiation | ✅ Working | Avg similarity 0.56 (target < 0.7) |
| Reconstruction | ✅ Working | 87% perfect on random sentences |
| **Dopamine System** | ✅ Working | Reward baseline 0.95, modulates LR |
| **Serotonin System** | ✅ Working | Controls exploration (temp=0.57) |
| **Norepinephrine System** | ✅ Working | Attention modulation (0.54) |
| **Hebbian Plasticity** | ✅ Working | 4 association traces active |
| **Hippocampus Memory** | ✅ Working | Stores novel vocabulary |
| **Text Generation** | ✅ Working | Autoregressive transformer decoder |

### Key Success Factors Identified

1. **Contrastive loss** with strong weight (5.0) prevents collapse
2. **Reconstruction decoders** at multiple levels ensure meaningful embeddings
3. **Explicit supervision** (boundaries, types, POS) guides learning
4. **Collapse monitoring** catches problems early

---

## Roadmap to Goal

### Ultimate Goal

A brain-inspired hierarchical language model that:

- Understands German at all linguistic levels (char → discourse)
- Uses dopamine-modulated learning (reward for correct predictions)
- Strengthens associations via Hebbian plasticity
- Maintains vocabulary memory via hippocampus module
- Can generate coherent German text

---

## Phase 2: Extend Hierarchy (Levels 4-5) ✅ COMPLETE

### Step 1: Add Phrase Chunker (Level 4) ✅

**Priority:** HIGH  
**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase2.py`

**Architecture:**

```
Level 4: PhraseChunker
  - Input: word_emb (256-dim)
  - BiLSTM (256 hidden, bidirectional)
  - Boundary head: detect phrase boundaries
  - Type head: NP, VP, PP, ADJP, ADVP, SBAR
  - Attention pooling (not mean pooling!)
  - Output: phrase_emb (512-dim)
```

**Key considerations:**

- Use attention pooling, NOT mean pooling (prevents collapse)
- Add contrastive loss from the start
- Train with phrase structure labels (can use heuristics initially)

**Phrase types for German:**

| Type | Example |
|------|---------|
| NP (Noun Phrase) | "der große Hund" |
| VP (Verb Phrase) | "läuft schnell" |
| PP (Prep Phrase) | "in dem Garten" |
| ADJP (Adj Phrase) | "sehr schön" |
| ADVP (Adv Phrase) | "heute morgen" |

---

### Step 2: Add Sentence Encoder (Level 5) ✅

**Priority:** HIGH  
**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase2.py`

**Architecture:**

```
Level 5: SentenceEncoder
  - Input: phrase_emb (512-dim)
  - Transformer encoder (2 layers, 4 heads)
  - Attention pooling over phrases
  - Output: sentence_emb (512-dim)
```

**Key considerations:**

- Use CLS token or learned query for pooling
- Strong contrastive loss at sentence level
- Reconstruction decoder: sentence_emb → characters

---

### Step 3: Validate Extended Hierarchy ✅

**Priority:** HIGH  
**Status:** COMPLETE

**Results:**

1. ✅ Phrase boundary detection working
2. ✅ Sentence embedding similarity: 0.56 (target < 0.7)
3. ✅ End-to-end reconstruction: 82% perfect

---

## Phase 3: Brain Integration ✅ COMPLETE

### Step 4: Add Dopamine Reward System ✅

**Priority:** MEDIUM  
**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase3.py`

**Architecture:**

```python
class DopamineSystem:
    def __init__(self):
        self.baseline = EMA(tau=0.99)
    
    def compute_reward(self, predictions, targets):
        accuracy = (predictions == targets).float().mean()
        return accuracy
    
    def get_dopamine(self, reward):
        # TD-RPE: reward prediction error
        delta = reward - self.baseline
        self.baseline.update(reward)
        return delta
```

**Integration points:**

- Modulate learning rate: `lr * (1 + dopamine * 0.5)`
- Higher dopamine → stronger gradient updates
- Monitor for reward hacking (100% reward = suspicious)

---

### Step 5: Add Hebbian Plasticity ✅

**Priority:** MEDIUM  
**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase3.py`

**Architecture:**

```python
class HebbianTrace:
    def __init__(self, d_pre, d_post):
        self.trace = torch.zeros(d_pre, d_post)
        self.tau = 0.95  # Decay rate
    
    def update(self, pre_activation, post_activation, dopamine):
        # Three-factor rule: pre × post × neuromodulator
        hebbian = torch.outer(pre_activation, post_activation)
        self.trace = self.tau * self.trace + (1 - self.tau) * hebbian * dopamine
```

**Traces to maintain:**

| Trace | From | To | Purpose |
|-------|------|-----|---------|
| char_syl | char_emb | syl_emb | Character-syllable associations |
| syl_morph | syl_emb | morph_emb | Syllable-morpheme associations |
| morph_word | morph_emb | word_emb | Morpheme-word associations |

---

### Step 6: Add Hippocampus Vocabulary Memory ✅

**Priority:** MEDIUM  
**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase3.py`

**Architecture:**

```python
class VocabularyMemory:
    def __init__(self, d_emb, capacity=10000):
        self.embeddings = torch.zeros(capacity, d_emb)
        self.words = []  # String representations
        self.count = 0
    
    def store(self, word_emb, word_str):
        # Fast encoding of new word
        self.embeddings[self.count] = word_emb
        self.words.append(word_str)
        self.count += 1
    
    def retrieve(self, query, k=5):
        # Similarity-based retrieval
        sims = F.cosine_similarity(query.unsqueeze(0), self.embeddings[:self.count])
        topk = sims.topk(k)
        return [(self.words[i], sims[i]) for i in topk.indices]
```

**Use cases:**

- Store new words encountered during training
- Retrieve similar words for completion/generation
- Track vocabulary growth over time

---

## Phase 4: Generation ✅ COMPLETE

### Step 7: Autoregressive Character Generation ✅

**Priority:** LOW  
**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase4.py`

**Approach:**

```python
def generate(model, prompt, max_chars=100):
    chars = text_to_indices(prompt)
    
    for _ in range(max_chars):
        outputs = model(chars)
        next_char_logits = outputs['next_char'][:, -1]
        next_char = sample(next_char_logits, temperature=0.8)
        chars = torch.cat([chars, next_char], dim=-1)
        
        if next_char == EOS:
            break
    
    return indices_to_text(chars)
```

---

### Step 8: Constrained Generation

**Priority:** LOW  
**Status:** PENDING

**Approach:**

- Generate given a morpheme template
- Generate given a phrase structure
- Generate given a semantic constraint

---

## Phase 5: Enhanced Brain + Fine-tuning ✅ COMPLETE

### Step 9: Add Serotonin & Norepinephrine ✅

**Status:** COMPLETE  
**Implementation:** `hierarchical_german_phase5.py`

| Module | Function | Current Value |
|--------|----------|---------------|
| Serotonin | Exploration vs exploitation | 0.79 (temp=0.57) |
| Norepinephrine | Attention/novelty | 0.54 |

### Step 10: Fine-tune with Regularization ✅

**Status:** COMPLETE

**Improvements applied:**

- Label smoothing (0.1)
- Increased weight decay (0.05)
- Lower learning rate (1e-4)
- Early stopping (patience=3)

---

## Phase 6: Evaluation & Refinement ⏳ PENDING

### Step 11: Benchmark Against Baselines

**Status:** PENDING

**Comparisons:**

- vs. Character-level LSTM (no hierarchy)
- vs. Subword tokenizer (BPE/WordPiece)
- vs. Pretrained German models (German BERT)

**Metrics:**

- Perplexity on held-out German text
- Morpheme boundary F1 score
- POS tagging accuracy
- Sentence similarity quality (STS benchmark)

---

### Step 12: Document & Publish

**Status:** PENDING

**Deliverables:**

- Complete documentation of architecture
- Training scripts with hyperparameters
- Pretrained model checkpoints
- Example usage notebooks

---

## Immediate Next Action

**Options for next steps:**

1. **Constrained Generation** - Generate given templates/constraints
2. **Benchmark** - Compare against baselines (LSTM, BPE, German BERT)
3. **Interactive Demo** - Build Gradio UI for testing
4. **Documentation** - Write up architecture and results
5. **Acetylcholine** - Add 4th neuromodulator for memory consolidation

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Embedding collapse at Level 4-5 | Use contrastive loss, attention pooling |
| Reward hacking in dopamine | Monitor reward curve, cap reward |
| Memory overflow in hippocampus | Use fixed capacity with LRU eviction |
| Overfitting on small dataset | Use dropout, data augmentation |
| German-specific issues | Consult German morphology resources |

---

## Success Criteria

The project is successful when:

1. ✅ All 6 hierarchy levels work without collapse
2. ✅ Brain modules (dopamine, Hebbian, hippocampus) integrate smoothly
3. ✅ Model can generate coherent German sentences
4. ✅ Vocabulary grows organically during training (hippocampus stores novel words)
5. ✅ Model shows "learning" behavior (loss: 1.46 → 1.08 over 20 epochs)

---

## Model Files Summary

| Phase | File | Key Features |
|-------|------|-------------|
| 1 | `hierarchical_german_tabula_rasa.py` | Base hierarchy L0-L3 |
| 2 | `hierarchical_german_phase2.py` | +Phrase/Sentence L4-L5 |
| 3 | `hierarchical_german_phase3.py` | +Dopamine, Hebbian, Hippocampus |
| 4 | `hierarchical_german_phase4.py` | +Autoregressive generator |
| 5 | `hierarchical_german_phase5.py` | +Serotonin, Norepinephrine |

**Best checkpoint:** `checkpoints/phase5_best.pth`

---

*Last updated: January 8, 2026*
