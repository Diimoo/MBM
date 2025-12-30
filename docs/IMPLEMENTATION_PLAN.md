# Implementation Plan – Digital Brain (v1.1)

**Source of Truth:** `docs/BESCHREIBUNG_v1.1.md` (architecture + interfaces).  
**Goal:** Build a modular neuro-inspired “Digital Brain” in PyTorch (rate-based v0), trained in phases, with strict interface contracts and continual-learning safeguards.

---

## 0) Non-Goals (v1.1)
- No 86B-neuron simulation.
- No full biophysical HH neuron models.
- No “end-to-end monolith” training that ignores module boundaries.

---

## 1) Compute Level (v1.1 Baseline)

### v0 (mandatory): Rate-based modules (PyTorch)
- **Cortex:** recurrent world model (GRU / SSM / lightweight Transformer-decoder) producing latent state `Z(t)` and predictions.
- **Thalamus:** differentiable gating/routing (gain × gate) for input streams.
- **Hippocampus:** episodic key–value memory (encode/retrieve/replay), with top-k retrieval + prioritized replay.
- **Basal Ganglia:** selection/policy (actor-critic) producing `selection/commit` + TD-RPE (`DA`).
- **Cerebellum:** supervised residual corrector (correction/timing_offset) trained on error signal.
- **Neuromodulators:** global scalars `DA, NE, ACh, 5HT` gating learning rates and thalamic gain.

### v1 (optional add-on later)
- Spiking (LIF/STDP) for submodules only, behind a feature adapter. Not required for v1.1.

---

## 2) Repo Layout (minimal, clean)
```

digital_brain/
brain.py                  # orchestration + step()
datatypes.py              # typed containers (Obs, BrainState, ModSignals, etc.)
modules/
cortex.py
thalamus.py
hippocampus.py
basal_ganglia.py
cerebellum.py
neuromodulators.py
envs/
pomdp_gridworld.py       # minimal environment that needs memory + gating
training/
train_phase1_worldmodel.py
train_phase2_memory.py
train_phase3_policy.py
train_phase4_cerebellum.py
train_phase5_language.py
tests/
test_contracts.py
test_cortex.py
test_thalamus.py
test_hippocampus.py
test_basal_ganglia.py
test_cerebellum.py
test_integration_demo.py
run_demo.py
README.md

````

---

## 3) Data Contracts (MUST be explicit)

### 3.1 Core tensors (standard shapes)
- `B` = batch size
- `d_obs` = observation dim
- `d_z` = cortex latent dim
- `d_sel` = selection dim
- `d_ctx` = context dim
- `d_act` = action dim

**Required invariants**
- `Z(t)` is always shape `(B, d_z)` and finite (no NaN/Inf).
- Thalamus gating outputs shape matches cortex input shape.
- Hippocampus retrieval returns `(B, d_z)` (or `(B, k, d_z)` then pooled).

### 3.2 Dataclasses (in `datatypes.py`)
- `Obs(x: Tensor, ctx: Tensor|None)`
- `BrainState(z: Tensor, cortex_state: Any, bg_state: Any, hip_state: Any, ...)`
- `Selection(sel: Tensor, commit: Tensor|bool)`
- `ModSignals(DA: Tensor, NE: Tensor, ACh: Tensor, HT5: Tensor)`
- `StepLog(pred_error, rpe, novelty, gate_stats, ...)`

---

## 4) Module Interfaces (MUST match BESCHREIBUNG_v1.1.md)

### 4.1 Cortex
**Purpose:** world model, representation `Z(t)`, prediction.
```python
Cortex.forward(obs_t, gated_inputs_t, prev_state) -> (z_t, pred_t, new_state)
Cortex.predict(z_t) -> pred_t
````

**Required:** prediction loss decreases in Phase 1.

### 4.2 Thalamus

**Purpose:** routing/gating for inputs.

```python
Thalamus.gate(inputs_dict, selection, feedback, mods) -> gated_inputs_dict
```

**Define gating explicitly**

* `gate = sigmoid(Ws*selection + Wf*feedback - b)`
* `gain = 1 + α_ACh*ACh + α_NE*NE`
* `gated = inputs * gate * gain`

### 4.3 Hippocampus

**Purpose:** episodic memory, fast encode/retrieve/replay.

```python
encode(z, ctx) -> episode_id
retrieve(cue, topk=K) -> z_retrieved
replay(n, prioritized=True) -> list[episode]
novelty(z) -> score
```

**Required:** 1-shot recall works; replay improves cortex stability.

### 4.4 Basal Ganglia

**Purpose:** selection/policy, TD-RPE (`DA`).

```python
BG.step(z_t, reward_t, ctx_t, done_t) -> (selection, DA_signal)
```

**TD-RPE:**

* `δ = r + γ*V(s') - V(s)`
  **Required:** DA responds correctly to reward changes.

### 4.5 Cerebellum

**Purpose:** residual correction + timing.

```python
Cerebellum.forward(plan, sensory, error_signal) -> (correction, timing_offset)
```

**Required:** reduces error relative to baseline.

### 4.6 Neuromodulators

**Purpose:** global control signals (scalars) that gate learning and gain.

```python
Neuromods.compute(z_t, pred_error, reward, memory_novelty) -> ModSignals
```

**Minimal definitions**

* `DA = TD-RPE` (from BG)
* `NE = novelty` (from hippocampus or pred_error spike)
* `ACh = attention proxy` (e.g., normalized pred_error / uncertainty)
* `5HT = stability proxy` (e.g., running average / patience; optional)

---

## 5) The One Function That Must Exist: `brain.step()`

In `brain.py` implement a closed loop:

```python
DigitalBrain.step(obs_t, reward_t, done_t) -> (action_t, new_state, log)
```

**Order (must be consistent)**

1. Cortex produces provisional `z_t` (or uses prev `z_{t-1}` if needed).
2. Hippocampus novelty + optional retrieve/replay cues.
3. BG computes `selection/commit` + `DA`.
4. Neuromods compute `NE/ACh/5HT` (and accept `DA`).
5. Thalamus gates input streams using selection + mods.
6. Cortex updates `z_t` with gated inputs and outputs predictions.
7. Cerebellum corrects action/timing (optional in early phases).
8. Action output produced (policy head or BG output).

---

## 6) Environment (must require memory + gating)

### v1.1 Environment: POMDP Gridworld “Key → Door”

* **Partial observation:** agent sees local neighborhood only.
* **Task:** find key, then door unlocks, then reach goal.
* **Reward:** sparse success reward + small step penalty.
* **Why:** without hippocampus/replay and gating, policy fails or is slower.

Deliver `envs/pomdp_gridworld.py` with:

* `reset() -> obs`
* `step(action) -> obs, reward, done, info`

---

## 7) Training Phases (strict, with freeze rules)

### Phase 1 — World Model (Cortex)

**Train:** Cortex (+ minimal Thalamus passthrough), no BG, no hippocampus writes.
**Loss:** next-step prediction `L_pred` + sparsity/regularization.
**Done when:**

* `L_pred` drops by X% over N steps
* Z(t) finite + stable
* contract tests pass

### Phase 2 — Episodic Memory (Hippocampus) + Replay

**Freeze:** Cortex core weights (95–99%).
**Train:** hippocampus encode/retrieve + small adapters if needed.
**Loss:** recall loss + novelty classification.
**Replay:** prioritized replay to cortex (offline minibatches).
**Done when:**

* 1-shot recall hit-rate ≥ threshold
* replay improves cortex prediction stability

### Phase 3 — Policy/Selection (Basal Ganglia) + Thalamus Gating

**Freeze:** Cortex core, hippocampus store (allow replay policy tweaks), train BG + Thalamus gate params.
**RL:** PPO (or A2C) using `Z(t)` as state.
**Ablation requirement:**

* performance with gating > performance without gating
  **Done when:**
* average return exceeds baseline by threshold
* DA signal correlates with reward changes

### Phase 4 — Cerebellum Residual Corrector

**Freeze:** BG policy mostly; allow cerebellum + small adapters.
**Loss:** supervised residual on error signal.
**Done when:** error reduced vs no-cerebellum baseline.

### Phase 5 — Language & Multilingual Heads (optional v1.1)

**Freeze:** core dynamics; train language encoders/decoders + switch control.
**Losses:** semantic alignment (contrastive) + translation loss + switch loss.
**Done when:** bilingual QA/translation meets minimal metrics.

---

## 8) Continual Learning (mandatory)

After each phase:

* Run **Regression Tests**: Phase 1 prediction still works, memory still recalls, etc.
* Use **Replay + Distillation** when training new modules:

  * `L_distill = || old_outputs - new_outputs ||`
* Prefer **adapters** over unfrozen core weights.

---

## 9) Verification (automated, non-negotiable)

### 9.1 Contract tests (`tests/test_contracts.py`)

* module I/O shapes
* dtype checks
* no NaNs/Infs
* deterministic forward pass with fixed seed

### 9.2 Unit tests

* `test_cortex.py`: prediction loss decreases on synthetic sequences.
* `test_thalamus.py`: gating changes input magnitude; ablation works.
* `test_hippocampus.py`: encode→retrieve returns correct episode (1-shot).
* `test_basal_ganglia.py`: DA (TD-RPE) sign matches reward change.
* `test_cerebellum.py`: residual corrector reduces error on toy regression.

### 9.3 Integration test

* `test_integration_demo.py`: `python run_demo.py` runs end-to-end for N steps and logs expected signals.

---

## 10) Deliverables Checklist (Definition of Done)

### Code

* All modules implemented with defined interfaces.
* `DigitalBrain.step()` works end-to-end.
* `run_demo.py` runs in < 1 minute on CPU.

### Training

* Phase scripts run and save checkpoints.
* Logs contain: `pred_error`, `DA`, `NE`, `ACh`, `gate_stats`, `replay_stats`.

### Quality

* All tests pass.
* Ablations demonstrate that modules matter (not decorative).

---

## 11) Immediate Next Action (build order)

1. Implement `datatypes.py` + `brain.step()` skeleton with stubs.
2. Implement Cortex + Thalamus passthrough + Phase 1 training + tests.
3. Add Hippocampus + Phase 2 + replay + tests.
4. Add BG + gating + Phase 3 RL + ablation tests.
5. Add Cerebellum + Phase 4.
6. Optional Phase 5 language.

**Rule:** No new module until the previous phase has green tests + a running demo.


