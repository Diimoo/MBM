# Hierarchical Language Model Architecture

## Philosophy

Human language learning follows a strict hierarchy:
```
Phonemes → Letters → Syllables → Morphemes → Words → Phrases → Sentences → Discourse
```

Traditional LLMs skip this hierarchy, jumping directly from tokens to prediction. This creates "stochastic parrots" - models that predict patterns without compositional understanding.

A **world model** must build representations at each level, where higher levels **compose** lower levels rather than bypassing them.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    HIERARCHICAL LANGUAGE MODEL                   │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Level 6: Discourse        [paragraph/document meaning]          │
│      ↑                                                           │
│  Level 5: Sentence         [proposition, full meaning]           │
│      ↑                                                           │
│  Level 4: Phrase           [NP, VP, PP chunks]                   │
│      ↑                                                           │
│  Level 3: Word             [lexical meaning]                     │
│      ↑                                                           │
│  Level 2: Morpheme         [root, prefix, suffix]                │
│      ↑                                                           │
│  Level 1: Syllable         [pronounceable units]                 │
│      ↑                                                           │
│  Level 0: Character        [alphabet]                            │
│      ↑                                                           │
│  Input: Raw text                                                 │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Level 0: Character Encoder

**Purpose:** Learn character representations (alphabet knowledge)

**Input:** Raw text as character indices
**Output:** Character embeddings with positional context

```python
class CharacterEncoder:
    # Embedding: vocab_size → d_char (e.g., 108 → 128)
    # Position encoding: learned or sinusoidal
    # Local context: small CNN or attention window
    
    # Output: [batch, seq_len, d_char]
```

**Training signal:** Self-supervised (masked char prediction)

---

## Level 1: Syllable Detector

**Purpose:** Group characters into pronounceable units

**German syllable rules:**
- Every syllable has exactly one vowel nucleus (a, e, i, o, u, ä, ö, ü)
- Consonant clusters follow sonority sequencing
- Examples: "Kat-ze", "Mög-lich-keit", "Früh-stück"

**Input:** Character embeddings [batch, seq_len, d_char]
**Output:** Syllable embeddings [batch, num_syllables, d_syllable]

```python
class SyllableDetector:
    # Boundary detection: predict syllable boundaries
    # Pooling: aggregate chars within syllable
    # Representation: syllable-level embedding
    
    # Key: Learn to segment "Möglichkeit" → ["Mög", "lich", "keit"]
```

**Training signal:** 
- Option A: Supervised with syllable dictionary
- Option B: Unsupervised via reconstruction + entropy regularization

---

## Level 2: Morpheme Parser

**Purpose:** Identify meaningful sub-word units (roots, prefixes, suffixes)

**German morphology:**
- Prefixes: un-, ver-, be-, ent-, er-, ge-, miss-, zer-
- Suffixes: -ung, -heit, -keit, -schaft, -lich, -ig, -bar, -sam
- Roots: The semantic core

**Examples:**
- "unmöglich" → [un- (negation), mög (root: possible), -lich (adj)]
- "Freundschaft" → [Freund (root: friend), -schaft (noun: -ship)]

**Input:** Syllable embeddings [batch, num_syllables, d_syllable]
**Output:** Morpheme embeddings [batch, num_morphemes, d_morpheme] + morpheme types

```python
class MorphemeParser:
    # Morpheme boundary detection
    # Type classification: ROOT, PREFIX, SUFFIX, INFLECTION
    # Semantic binding: connect affixes to meaning modifications
    
    # Key: Learn that "-keit" transforms adjectives → nouns
```

**Training signal:**
- Morphological analyzer for supervision (e.g., DEMorphy for German)
- Or unsupervised via meaning consistency

---

## Level 3: Word Composer

**Purpose:** Compose morphemes into word-level meaning

**Key insight:** Word meaning = f(root_meaning, affix_modifications)

**German compound words:**
- "Donaudampfschifffahrtsgesellschaftskapitän"
- = Donau + Dampf + Schiff + Fahrt + Gesellschaft + Kapitän
- Each component contributes meaning compositionally

**Input:** Morpheme embeddings + types
**Output:** Word embeddings [batch, num_words, d_word]

```python
class WordComposer:
    # Compositional semantics
    # Root provides base meaning
    # Affixes modify systematically
    # Compounds combine left-to-right (German: head-final)
```

**Training signal:** 
- Word similarity (synonyms should be close)
- Definition matching
- Context prediction

---

## Level 4: Phrase Chunker

**Purpose:** Group words into syntactic phrases

**Phrase types:**
- NP (Noun Phrase): "der kleine Hund"
- VP (Verb Phrase): "läuft schnell"
- PP (Prepositional Phrase): "in dem Haus"
- AP (Adjective Phrase): "sehr schön"

**Input:** Word embeddings + POS tags
**Output:** Phrase embeddings [batch, num_phrases, d_phrase] + phrase types

```python
class PhraseChunker:
    # Constituency parsing (simplified)
    # Head detection (which word is the head)
    # Modifier attachment
```

**Training signal:** Constituency parse trees or unsupervised chunking

---

## Level 5: Sentence Encoder

**Purpose:** Compose phrases into full propositional meaning

**Captures:**
- Subject-Verb-Object relationships
- Tense, aspect, modality
- Negation scope
- Question vs statement

**Input:** Phrase embeddings + phrase types
**Output:** Sentence embedding [batch, d_sentence]

```python
class SentenceEncoder:
    # Predicate-argument structure
    # Semantic role labeling (who did what to whom)
    # Full proposition representation
```

---

## Level 6: Discourse Model

**Purpose:** Track meaning across sentences

**Captures:**
- Coreference (what "he" refers to)
- Topic continuity
- Narrative structure

---

## Training Strategy

### Phase 1: Bottom-Up Pre-training

Train each level separately, frozen lower levels:

```
1. Train CharEncoder (masked char prediction)
2. Freeze CharEncoder, train SyllableDetector
3. Freeze 0-1, train MorphemeParser  
4. Freeze 0-2, train WordComposer
5. Continue upward...
```

### Phase 2: Joint Fine-tuning

Unfreeze all levels, train end-to-end with:
- Next-word prediction loss (top-down)
- Reconstruction losses at each level (bottom-up)
- Consistency losses (representations should align)

### Phase 3: Generation Training

Train decoder at each level:
```
Sentence meaning → Phrase sequence → Word sequence → Morpheme sequence → Char sequence
```

---

## Implementation Plan

### Increment 1: Char + Syllable (Current Focus)

```python
# File: hierarchical_german_v1.py

class HierarchicalGermanV1:
    def __init__(self):
        self.char_encoder = CharacterEncoder(vocab_size=108, d_char=128)
        self.syllable_detector = SyllableDetector(d_char=128, d_syllable=256)
        
    def forward(self, text):
        chars = self.char_encoder(text)           # [B, L, 128]
        syllables, boundaries = self.syllable_detector(chars)  # [B, S, 256]
        return syllables, boundaries
```

**Evaluation:**
- Can it correctly syllabify German words?
- Do syllable representations cluster by phonetic similarity?

### Increment 2: Add Morpheme Layer

After syllable layer works, add morpheme parsing.

### Increment 3: Add Word Composition

After morpheme layer works, add word composition.

---

## Key Differences from Flat Models

| Aspect | Flat Model | Hierarchical Model |
|--------|------------|-------------------|
| Representation | Single embedding | Embedding at each level |
| Composition | Implicit | Explicit |
| Interpretability | Black box | Can inspect each level |
| Generalization | Memorization | Compositional |
| Data efficiency | Needs billions of tokens | Should need less |

---

## German-Specific Considerations

1. **Compound words:** German creates new words by composition
   - Model must learn to decompose AND compose
   
2. **Case system:** Nominative, Accusative, Dative, Genitive
   - Morpheme layer should capture case markers
   
3. **Verb position:** V2 in main clauses, V-final in subordinate
   - Phrase/Sentence layers should capture this

4. **Separable verbs:** "aufmachen" → "Ich mache die Tür auf"
   - Word layer needs to handle discontinuous morphemes

---

## Success Criteria

### Level 1 (Syllable)
- [ ] 95%+ accuracy on syllable boundary detection
- [ ] Syllable embeddings cluster by vowel nucleus

### Level 2 (Morpheme)
- [ ] Correctly identifies German prefixes/suffixes
- [ ] Root extraction accuracy > 90%

### Level 3 (Word)
- [ ] Compound word decomposition works
- [ ] Word similarity correlates with human judgments

### Level 4+ (Phrase/Sentence)
- [ ] Basic constituency parsing
- [ ] Coherent sentence generation

---

## Next Steps

1. **Implement CharEncoder + SyllableDetector**
2. **Create German syllable training data** (using hyphenation dictionary)
3. **Train and evaluate Level 0-1**
4. **Add MorphemeParser once syllables work**
