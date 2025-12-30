# Digitales Gehirn – Technische Spezifikation (v1.0)

## Basierend auf: Kandel et al., Principles of Neural Science, 5th Edition

> **Regelwerk**: Jede Behauptung MUSS eine Quellenangabe haben. Nicht belegte Konzepte sind als "NICHT IM KANDEL" markiert und gelten als optionale Add-ons.

---

## 0. Ziel und Scope

Dieses Dokument spezifiziert eine neuro-inspirierte Architektur für ein digitales System, das:
- Wahrnehmung integriert, Vorhersagen generiert, Handlungen auswählt
- lernt (kurz- und langfristig), konsolidiert, vergisst
- Sprache(n) verarbeitet
- modulare Zuständigkeiten über Gating/Selection organisiert

**Nicht-Ziel**: 1:1 Simulation auf Molekülebene oder Skalierung auf 86B Neuronen.

---

## 1. Recheneinheiten (Neuron-Primitive)

### 1.1 Biologisches Prinzip

Neuronen integrieren synaptische Eingänge über passive Membraneigenschaften und generieren Aktionspotentiale bei Überschreitung eines Schwellenwerts.

> "The passive electrical properties of the neuron—its membrane resistance, capacitance, and axoplasmic resistance—determine the time course and efficiency of signal conduction." (Kandel5e, Kap. 6, S. 138–144)

> "When the membrane potential reaches threshold, voltage-gated Na+ channels open, initiating the action potential." (Kandel5e, Kap. 7, S. 148–156)

### 1.2 Digitales Pendant

**Option A: Leaky Integrate-and-Fire (LIF)**

```
Zustand: V(t) ∈ ℝ (Membranpotential)
Parameter: V_rest, V_th, V_reset, τ_m, R_m, τ_ref

Dynamik:
τ_m × dV/dt = -(V - V_rest) + R_m × I_syn(t)

Spike-Regel:
wenn V ≥ V_th:
    spike ausgeben
    V := V_reset
    Refraktärzeit τ_ref einhalten
```

**Option B: Izhikevich-Modell** (für reichere Dynamik)

```
v' = 0.04v² + 5v + 140 - u + I
u' = a(bv - u)

wenn v ≥ 30mV:
    v := c
    u := u + d

Parameter (a,b,c,d) bestimmen Neuronentyp
```

> Biologische Basis: Hodgkin-Huxley-Dynamik (Kandel5e, Kap. 7, S. 153–156)

### 1.3 Schnittstellen

| Input | Output | Zustände |
|-------|--------|----------|
| I_syn(t): synaptischer Strom | spike ∈ {0,1} | V(t): Membranpotential |
| I_ext(t): externer Strom | | u(t): Recovery-Variable (Izhikevich) |

### 1.4 Tests/Benchmarks

- Feuerrate im physiologischen Bereich (1–100 Hz)
- Refraktärzeit verhindert Bursts > 500 Hz
- Korrekte F-I-Kurve (Frequenz vs. Eingangsstrom)

---

## 2. Synapsen-Primitive

### 2.1 Biologisches Prinzip

Synapsen übertragen Signale durch Neurotransmitter-Freisetzung, die postsynaptische Ströme erzeugt.

> "The synaptic current depends on the conductance of the postsynaptic receptors and the driving force on the permeant ions." (Kandel5e, Kap. 10, S. 210–220)

> "Short-term synaptic plasticity can either facilitate or depress transmission depending on the pattern of presynaptic activity." (Kandel5e, Kap. 12, S. 285–290)

### 2.2 Digitales Pendant

**Basis-Synapse:**

```
I_syn(t) = Σ_j w_ij × g_j(t) × (V - E_rev)

g_j(t) = Σ_spikes exp(-(t - t_spike) / τ_syn)
```

**Mit Short-Term Plasticity (STP):**

```
# Tsodyks-Markram-Modell
dx/dt = (1-x)/τ_D - u×x×δ(t-t_spike)    # Depression
du/dt = (U-u)/τ_F + U×(1-u)×δ(t-t_spike)  # Facilitation

w_eff = w × u × x
```

> Biologische Basis: Facilitation und Depression (Kandel5e, Kap. 12, S. 285–290)

### 2.3 Schnittstellen

| Input | Output | Zustände |
|-------|--------|----------|
| pre_spike: präsynaptischer Spike | I_post: postsynaptischer Strom | w: Gewicht |
| V_post: postsynaptisches Potential | | x, u: STP-Variablen |

### 2.4 Tests/Benchmarks

- Facilitation bei hochfrequenter Stimulation (> 20 Hz)
- Depression bei anhaltender Aktivität
- Gewichte bleiben in physiologischem Bereich

---

## 3. Lernregeln (Plastizität)

### 3.1 Hebbianische Plastizität / LTP/LTD

#### Biologisches Prinzip

> "Long-term potentiation in the Schaffer collateral pathway follows Hebbian learning rules: It requires coincident pre- and postsynaptic activity." (Kandel5e, Kap. 67, S. 1497–1500)

> "The NMDA receptor acts as a coincidence detector, requiring both glutamate binding and postsynaptic depolarization to allow Ca²⁺ influx." (Kandel5e, Kap. 67, S. 1493–1497)

> "Long-term depression occurs when presynaptic activity is not correlated with strong postsynaptic depolarization." (Kandel5e, Kap. 67, S. 1500)

#### Digitales Pendant: STDP

```
Δw_ij = {
    +A_+ × exp(-Δt / τ_+)  wenn Δt > 0 (pre vor post)
    -A_- × exp(+Δt / τ_-)  wenn Δt < 0 (post vor pre)
}

wobei Δt = t_post - t_pre

Typische Parameter:
A_+ = 0.01, A_- = 0.012
τ_+ = τ_- = 20 ms
```

> **Hinweis**: Die explizite STDP-Formel mit Zeitfenstern ist NICHT IM KANDEL in dieser Form. Kandel beschreibt LTP/LTD konzeptuell; die Formalisierung stammt aus der Computational Neuroscience.

### 3.2 Drei-Faktor-Lernen (Eligibility Trace × Modulator)

#### Biologisches Prinzip

> "Dopamine neurons in the midbrain signal reward prediction errors... These signals influence synaptic plasticity in target structures." (Kandel5e, Kap. 49, S. 1110–1112)

> "The cAMP-PKA-CREB pathway converts short-term to long-term memory by initiating gene expression." (Kandel5e, Kap. 66, S. 1475–1485)

#### Digitales Pendant

```
# Eligibility Trace
de_ij/dt = -e_ij/τ_e + STDP(pre_i, post_j)

# Gewichtsupdate
Δw_ij = η × M(t) × e_ij

# Modulator M(t) kann sein:
# - Dopamin-Signal (RPE)
# - Noradrenalin (Surprise)
# - Acetylcholin (Attention)
```

> **Hinweis**: Die explizite 3-Faktor-Formel ist NICHT IM KANDEL. Kandel beschreibt Dopamin-Modulation von Plastizität, aber nicht als formale Lernregel.

### 3.3 Homöostatische Plastizität

#### Biologisches Prinzip

> "Synaptic scaling adjusts all of a neuron's excitatory synapses up or down to maintain stable firing rates." (Kandel5e, Kap. 56, S. 1270–1280)

#### Digitales Pendant

```
# Synaptic Scaling
w_i := w_i × (r_target / r_observed)^α

# Alternativ: Intrinsische Plastizität
V_th := V_th + β × (r_observed - r_target)
```

### 3.4 Schnittstellen

| Input | Output | Zustände |
|-------|--------|----------|
| pre_activity, post_activity | Δw | e_ij: Eligibility Trace |
| M(t): Modulatorsignal | | r_observed: Feuerrate |
| r_target: Zielrate | | |

### 3.5 Tests/Benchmarks

- LTP bei korrelativer Aktivität
- LTD bei antikorrelativer Aktivität
- Stabile Feuerraten nach homöostatischer Anpassung
- Keine Gewichtsdivergenz

---

## 4. Modul: KORTEX

### 4.1 Biologisches Prinzip

> "The cerebral cortex is organized into six layers, each with distinct cell types and connectivity patterns." (Kandel5e, Kap. 15, S. 372–380)

> "Pyramidal neurons are the principal excitatory neurons of the cortex and form long-range connections." (Kandel5e, Kap. 4, S. 71–85)

> "Inhibitory interneurons using GABA shape the temporal dynamics of cortical processing." (Kandel5e, Kap. 13, S. 295–300)

### 4.2 Rolle

- Hierarchische Repräsentationen
- Rekurrente Dynamik für Kontext und Sequenzen
- Generierung von Vorhersagen

### 4.3 Digitales Pendant

**Mikro-Schaltkreis:**

```
Komponenten pro Kortikaler Kolumne:
- E: Exzitatorische Pyramidenzellen (80%)
- I_fast: Schnelle Inhibition (Parvalbumin+)
- I_slow: Langsame Inhibition (Somatostatin+)

Konnektivität:
E → E (rekurrent, lernbar)
E → I_fast, I_slow
I_fast → E (Feedback-Inhibition)
I_slow → E (laterale Inhibition)
```

**Hierarchie:**

```
Feedforward: L4 → L2/3 → L5
Feedback: L5 → L2/3 → L4
Inter-Areal: L2/3 (FF) ↔ L5/L6 (FB)
```

### 4.4 Predictive Coding Framework

> **NICHT IM KANDEL**: Predictive Coding als formales Framework wird nicht beschrieben. Kandel erwähnt Vorhersagen im Kontext von Kleinhirn und Basalganglien, aber nicht als kortikales Organisationsprinzip.

**Optionales Add-on:**

```
Prediction: x̂ = g(z)
Error: ε = x - x̂
Update: z := z + α × ∂g/∂z × ε
```

### 4.5 Schnittstellen

| Input | Output |
|-------|--------|
| sensorische Features | Vorhersagen (top-down) |
| Gating-Signale (Thalamus) | Prediction Errors (bottom-up) |
| Selection (Basalganglien) | latente Zustände Z(t) |
| episodische Keys (Hippocampus) | |

### 4.6 Zustände

- Z(t): Aktivitätsmuster (Repräsentation)
- W: Synaptische Gewichte (lernbar)

### 4.7 Tests/Benchmarks

- Stabile Feuerraten (5–30 Hz)
- Sparsity: < 10% gleichzeitig aktive Neuronen
- Hierarchische Feature-Extraktion

---

## 5. Modul: THALAMUS

### 5.1 Biologisches Prinzip

> "The thalamus is the gateway to the cerebral cortex. Almost all sensory information passes through the thalamus before reaching the cortex." (Kandel5e, Kap. 15, S. 363–368)

> "Thalamic relay neurons can operate in two modes: tonic (relay) mode and burst mode." (Kandel5e, Kap. 46, S. 1038–1042)

> "The reticular nucleus of the thalamus provides inhibitory control over thalamic relay neurons." (Kandel5e, Kap. 46, S. 1042–1044)

### 5.2 Rolle

- Kontrolliert, welche Inputs den Kortex erreichen
- Implementiert Attention-Gating als dynamisches Routing

### 5.3 Digitales Pendant

```
Struktur:
- Relay-Neuronen (R): je ein Kern pro Modalität
- Reticular-Neuronen (TRN): inhibitorische Kontrolle

Dynamik:
R_out = gain × R_in × gate

gate = σ(TRN_input + cortical_feedback)
gain = f(modulators)  # ACh, NE erhöhen Gain

Modi:
- Tonic: gain hoch, gate offen → Relay
- Burst: gain niedrig, gate variabel → Gating
```

### 5.4 Schnittstellen

| Input | Output |
|-------|--------|
| sensorische Streams | gefilterter Input an Kortex |
| Kortex top-down Signale | Gain-Parameter |
| Modulatoren (ACh, NE) | |
| BG Selection-Signale | |

### 5.5 Zustände

- gate[k]: Gating-Zustand pro Kanal
- mode: tonic vs. burst

### 5.6 Tests/Benchmarks

- Selektive Weiterleitung bei Attention
- Blockierung irrelevanter Inputs
- Modusumschaltung funktional

---

## 6. Modul: BASALGANGLIEN

### 6.1 Biologisches Prinzip

> "The basal ganglia consist of the striatum (caudate and putamen), globus pallidus, subthalamic nucleus, and substantia nigra." (Kandel5e, Kap. 43, S. 982–985)

> "The direct pathway facilitates movement by disinhibiting the thalamus, while the indirect pathway suppresses movement." (Kandel5e, Kap. 43, S. 990–995)

> "Dopamine from the substantia nigra modulates the balance between direct and indirect pathways: D1 receptors excite the direct pathway, D2 receptors inhibit the indirect pathway." (Kandel5e, Kap. 43, S. 995–1000)

> "Dopaminergic neurons signal reward prediction errors: They fire above baseline for unexpected rewards and below baseline when expected rewards are omitted." (Kandel5e, Kap. 49, S. 1110–1112)

### 6.2 Rolle

- Auswahl von Handlungen und kognitiven Routinen
- Lernen über Reward Prediction Error (RPE)

### 6.3 Digitales Pendant

**Struktur:**

```
Striatum:
- D1-MSNs (direct pathway, "Go")
- D2-MSNs (indirect pathway, "NoGo")

GPi/SNr (Output):
- Tonisch aktiv, hemmt Thalamus
- Inhibition durch D1-MSNs → Disinhibition

STN:
- Erregt GPi → globale Inhibition ("Hold")
```

**Reward Prediction Error:**

```
# Temporal Difference (TD) RPE
δ_t = r_t + γ × V(s_{t+1}) - V(s_t)

# Dopamin-Signal
DA(t) = baseline + β × δ_t

# Effekt auf Striatum
D1_activity ∝ DA(t)  # verstärkt Go
D2_activity ∝ 1/DA(t)  # verstärkt NoGo bei niedrigem DA
```

**Policy Update:**

```
# Actor-Critic
V(s) := V(s) + α_critic × δ_t  # Critic lernt Value

# Policy über Striatum-Gewichte
w_Go := w_Go + α_actor × δ_t × e_Go
w_NoGo := w_NoGo - α_actor × δ_t × e_NoGo
```

### 6.4 Schnittstellen

| Input | Output |
|-------|--------|
| Kortex Z(t): State-Embedding | Go/NoGo Signale |
| Reward r_t | Selection an Thalamus |
| Dopamin δ_t | Action commitment |

### 6.5 Zustände

- V(s): Value-Funktion (im Critic)
- w_Go, w_NoGo: Policy-Gewichte
- selected_action: aktuelle Auswahl

### 6.6 Tests/Benchmarks

- Korrekte Action-Selection in Bandit-Tasks
- TD-Error korreliert mit Dopamin-Proxy
- Go/NoGo-Balance verhindert Überaktivität

---

## 7. Modul: HIPPOCAMPUS

### 7.1 Biologisches Prinzip

> "The hippocampus is essential for the formation of new declarative memories." (Kandel5e, Kap. 65, S. 1445–1455)

> "The hippocampus can rapidly encode new information in a single trial, in contrast to the slow learning in the neocortex." (Kandel5e, Kap. 67, S. 1500–1510)

> "Pattern separation in the dentate gyrus orthogonalizes similar inputs, while pattern completion in CA3 allows retrieval from partial cues." (Kandel5e, Kap. 67, S. 1505–1510)

> "During sleep, the hippocampus replays recently encoded memories, facilitating consolidation in the neocortex." (Kandel5e, Kap. 51, S. 1148–1150)

### 7.2 Rolle

- Schnelle Speicherung neuer Episoden (1-shot)
- Index für spätere Rekonstruktion
- Replay für Konsolidierung

### 7.3 Digitales Pendant

**Struktur:**

```
Entorhinal Cortex (EC):
- Input/Output Interface zum Kortex

Dentate Gyrus (DG):
- Sparse Coding, Pattern Separation
- Expansion: EC (klein) → DG (groß, sparse)

CA3:
- Rekurrente Verbindungen
- Pattern Completion, Autoassoziatives Netzwerk

CA1:
- Output zum Kortex
- Vergleicht EC-Input mit CA3-Recall
```

**Encoding:**

```
# Pattern Separation in DG
DG = sparse_encode(EC_input)  # k-winner-take-all

# Storage in CA3
CA3_pattern = DG
CA3_weights += outer(CA3_pattern, CA3_pattern)  # Hopfield-like
```

**Retrieval:**

```
# Partial cue → Pattern Completion
CA3_recalled = sign(CA3_weights @ partial_cue)

# Output via CA1
EC_output = CA1_decode(CA3_recalled)
```

**Replay:**

```
# Offline (Sleep/Rest)
for episode in recent_episodes:
    replay_sequence = hippocampus.retrieve(episode.cue)
    cortex.consolidate(replay_sequence)
```

### 7.4 Complementary Learning Systems (CLS)

> **Hinweis**: Das CLS-Framework als Theorie ist NICHT IM KANDEL explizit benannt, aber die Grundidee wird beschrieben:

> "The hippocampus rapidly encodes specific episodes, while the neocortex slowly extracts statistical regularities." (Kandel5e, Kap. 67, S. 1500–1510)

### 7.5 Schnittstellen

| Input | Output |
|-------|--------|
| Kortex-Zustände Z(t) | Keys/Indices für Recall |
| Kontextsignale | Replay-Sequenzen |
| Novelty-Signal | |

### 7.6 Zustände

- stored_episodes: Liste gespeicherter Muster
- CA3_weights: autoassoziative Gewichte

### 7.7 Tests/Benchmarks

- 1-shot Learning neuer Episoden
- Recall bei partiellen Cues (> 50% korrekt bei 30% Cue)
- Pattern Separation: ähnliche Inputs → distinkte Codes

---

## 8. Modul: KLEINHIRN

### 8.1 Biologisches Prinzip

> "Purkinje cells are the sole output of the cerebellar cortex." (Kandel5e, Kap. 42, S. 960–965)

> "Climbing fibers from the inferior olive carry error signals that trigger long-term depression of parallel fiber synapses onto Purkinje cells." (Kandel5e, Kap. 42, S. 965–977)

> "The cerebellum learns to predict the sensory consequences of movements and generates corrective signals." (Kandel5e, Kap. 42, S. 975–978)

### 8.2 Rolle

- Supervised Error-Driven Learning
- Timing und Präzision von Sequenzen
- Automatisierung von Abläufen

### 8.3 Digitales Pendant

**Struktur:**

```
Mossy Fibers (MF):
- Kontextuelle Eingänge (Efferenzkopie, sensorisch)
- Expandiert durch Granule Cells (GC)

Parallel Fibers (PF):
- GC → Purkinje Cell (PC) Synapsen
- Lernbare Gewichte

Climbing Fibers (CF):
- Error-Signal von Inferior Olive
- 1 CF pro PC

Purkinje Cells (PC):
- Integriert PF-Inputs
- Inhibiert Deep Cerebellar Nuclei (DCN)
```

**Lernregel (LTD):**

```
# Cerebellar LTD
Δw_PF = -η × PF_activity × CF_activity

# CF feuert bei Fehler
CF = error_signal  # z.B. |predicted - actual|

# Effekt: Gewichte reduziert, die bei Fehler aktiv waren
```

### 8.4 Schnittstellen

| Input | Output |
|-------|--------|
| Efferenzkopie (Plan) | Korrigierte Steuerwerte |
| Error-Signal | Timing-Parameter |
| Sensorisches Feedback | |

### 8.5 Zustände

- w_PF: Parallel Fiber Gewichte
- PC_activity: Purkinje-Zell-Aktivität

### 8.6 Tests/Benchmarks

- Fehlerreduktion über Trials
- Timing-Präzision < 50ms
- Generalisierung auf ähnliche Bewegungen

---

## 9. Modul: NEUROMODULATOREN

### 9.1 Biologisches Prinzip

> "Dopaminergic neurons from the VTA project to the striatum and prefrontal cortex, signaling reward prediction errors." (Kandel5e, Kap. 49, S. 1108–1113)

> "Noradrenergic neurons from the locus coeruleus project widely and increase arousal and attention in response to salient stimuli." (Kandel5e, Kap. 46, S. 1046–1050)

> "Cholinergic neurons modulate attention and enhance synaptic plasticity in the cortex." (Kandel5e, Kap. 46, S. 1054–1056)

> "Serotonergic neurons from the raphe nuclei influence mood, impulsivity, and temporal discounting of rewards." (Kandel5e, Kap. 46, S. 1050–1054)

### 9.2 Digitales Pendant

```
Modulator-System:
- DA: Dopamin → Lernsignal (RPE)
- NE: Noradrenalin → Arousal/Reset
- ACh: Acetylcholin → Attention/Plasticity Gate
- 5HT: Serotonin → Temporal Horizon

Effekte:
DA → BG Plastizität, Kortex Gain
NE → globaler Reset, Exploration
ACh → sensorischer Gain, LTP-Threshold
5HT → Discount-Faktor γ, Impulskontrolle
```

**Implementierung:**

```python
class Neuromodulators:
    def compute_DA(self, reward, value_pred):
        return reward - value_pred  # RPE
    
    def compute_NE(self, surprise):
        return sigmoid(surprise - threshold_NE)
    
    def compute_ACh(self, attention_signal):
        return attention_signal
    
    def compute_5HT(self, baseline=0.5):
        return baseline  # moduliert durch Kontext

    def apply_effects(self, modules):
        modules.BG.learning_rate *= (1 + self.DA)
        modules.cortex.gain *= (1 + self.ACh)
        modules.all.reset_probability *= self.NE
        modules.BG.gamma = 0.9 + 0.09 * self.5HT
```

### 9.3 Schnittstellen

| Input | Output |
|-------|--------|
| Reward r_t | DA: RPE-Signal |
| Surprise/Novelty | NE: Reset-Signal |
| Attention demand | ACh: Gain-Signal |
| Context | 5HT: Horizon-Parameter |

### 9.4 Tests/Benchmarks

- DA korreliert mit Reward-Überraschung
- NE erhöht nach unerwarteten Ereignissen
- ACh moduliert Lernrate messbar

---

## 10. Gedächtnis-Systeme

### 10.1 Working Memory

#### Biologisches Prinzip

> "Persistent activity in the prefrontal cortex maintains information over short delays in the absence of sensory input." (Kandel5e, Kap. 67, S. 1505–1515)

#### Digitales Pendant

```
Implementierung: Rekurrente Loops in PFC

# Sustained Activity
dV/dt = -V/τ + W_rec @ V + W_in @ input + noise

# Kapazitätsbegrenzung durch Inhibition
active_slots = top_k(V, k=4)  # ~4 Items

# Gating durch BG
if BG.gate_open:
    V := W_in @ new_input
```

### 10.2 Episodisches Gedächtnis

> "Episodic memory depends on the hippocampus and involves the encoding of specific events in their spatiotemporal context." (Kandel5e, Kap. 65, S. 1445–1455)

→ Siehe Modul HIPPOCAMPUS (Sektion 7)

### 10.3 Semantisches Gedächtnis

> "Semantic memory, or knowledge about the world, is thought to be stored in distributed representations across the neocortex." (Kandel5e, Kap. 65, S. 1450–1455)

#### Digitales Pendant

```
# Langsame Extraktion in Kortex
for replay_episode in hippocampus.replay():
    cortex.weights += η_slow × hebbian_update(replay_episode)

# Verteilte Repräsentationen
semantic_vector = cortex.encode(concept)
```

### 10.4 Prozedurales Gedächtnis

> "Procedural memory for skills and habits depends on the basal ganglia and cerebellum." (Kandel5e, Kap. 65, S. 1450–1455)

→ Siehe Module BASALGANGLIEN (Sektion 6) und KLEINHIRN (Sektion 8)

---

## 11. Taktisches Vergessen

### 11.1 Synaptischer Abbau/Decay

#### Biologisches Prinzip

> "Synaptic connections that are not used tend to weaken and may eventually be eliminated." (Kandel5e, Kap. 55, S. 1245–1255)

#### Digitales Pendant

```
# Weight Decay
w := (1 - λ) × w

# Threshold-basiertes Pruning
if |w| < θ_prune:
    w := 0
    synapse.remove()
```

### 11.2 Pruning / Strukturplastizität

> "During development and in response to experience, unnecessary synapses are eliminated through activity-dependent pruning." (Kandel5e, Kap. 55, S. 1250–1255)

#### Digitales Pendant

```
# Activity-based Pruning
usage[i,j] += |w[i,j] × activity[i] × activity[j]|

if usage[i,j] < θ_usage over T_window:
    prune(synapse[i,j])

# Rewiring
if random() < p_rewire:
    new_target = sample_by_activity()
    create_synapse(source, new_target)
```

### 11.3 Interferenz-Management

> "Pattern separation in the dentate gyrus helps distinguish similar memories and reduce interference." (Kandel5e, Kap. 67, S. 1505–1510)

#### Digitales Pendant

```
# Sparse Coding in DG
DG_code = k_winners(EC_input, k=sparse_k)

# Prioritized Replay
priority[episode] = |TD_error[episode]| + recency_bonus
replay_order = sorted(episodes, by=priority, descending=True)
```

---

## 12. Sprache und Mehrsprachigkeit

### 12.1 Biologisches Prinzip

> "Broca's area in the left frontal lobe is important for speech production and grammatical processing." (Kandel5e, Kap. 60, S. 1354–1365)

> "Wernicke's area in the left temporal lobe is important for speech comprehension." (Kandel5e, Kap. 60, S. 1365–1372)

> "There is a critical period for language acquisition; after puberty, learning a new language becomes more difficult." (Kandel5e, Kap. 60, S. 1356–1358)

### 12.2 Digitales Pendant

> **Hinweis**: Mehrsprachigkeit, Language Switching und Übersetzung sind NICHT IM KANDEL detailliert beschrieben. Die folgende Spezifikation ist ein optionales Add-on.

**Optionales Add-on: Spracharchitektur**

```
Komponenten:
- S: Gemeinsamer semantischer Latent-Space
- E_L: Encoder pro Sprache L (Tokens → S)
- D_L: Decoder pro Sprache L (S → Tokens)

Language Control:
- PFC/BG wählt aktive Sprache L_active
- Thalamus gatet E_L und D_L

Übersetzung:
Input_A → E_A → S → D_B → Output_B
```

---

## 13. Koordination und Gating

### 13.1 Salience Detection

#### Biologisches Prinzip

> "The amygdala evaluates the emotional significance of stimuli and can trigger rapid responses to threats." (Kandel5e, Kap. 48, S. 1079–1095)

> "Noradrenergic neurons in the locus coeruleus respond to salient or surprising stimuli." (Kandel5e, Kap. 46, S. 1046–1050)

### 13.2 Loop-Konkurrenz

#### Biologisches Prinzip

> "The basal ganglia select among competing motor programs by disinhibiting the appropriate thalamocortical loop." (Kandel5e, Kap. 43, S. 990–995)

#### Digitales Pendant

```
# Competition
proposals = [cortex_module.propose() for module in modules]
utilities = [BG.evaluate(p) for p in proposals]
winner = argmax(utilities)

# Selection
BG.select(winner)
thalamus.route_to(winner)
```

---

## 14. Trainingsregime

### Phase 1: Self-Supervised Weltmodell

```
Ziel: Vorhersage nächster Zustände
Loss: L_pred = ||x_{t+1} - f(x_t, a_t)||²
Ergebnis: Stabile latente Repräsentationen Z
```

### Phase 2: Reinforcement Learning

```
Ziel: Policy-Lernen
Algorithmus: Actor-Critic mit TD(λ)
Curriculum: Einfache → komplexe Tasks
```

### Phase 3: Memory und Replay

```
Ziel: Konsolidierung
Ablauf: 
1. Encoding neuer Episoden (Hippocampus)
2. Offline-Replay (Sleep-Phase)
3. Kortikale Konsolidierung
```

### Phase 4: Sprache (Optional)

```
Ziel: Sprachverarbeitung
Ablauf:
1. Phonologische Encoder
2. Semantik stabilisieren
3. Decoder pro Sprache
```

---

## 15. Definition of Done (Tests)

| Test | Kriterium | Kandel-Referenz |
|------|-----------|-----------------|
| Stabilität | Feuerraten 1–100 Hz, keine Divergenz | Kap. 7, S. 156–160 |
| LTP/LTD | Korrekte Richtung bei Korrelation | Kap. 67, S. 1497–1500 |
| Working Memory | Recall nach 5s Delay > 90% | Kap. 67, S. 1505–1515 |
| Episodisches Gedächtnis | 1-shot Learning funktional | Kap. 67, S. 1500–1510 |
| Action Selection | Bandit-Task > 80% optimal | Kap. 43, S. 990–998 |
| Error Learning | Fehlerreduktion im Kleinhirn | Kap. 42, S. 975–978 |
| Konsolidierung | Transfer Hippocampus → Kortex | Kap. 51, S. 1148–1150 |

---

## 16. Implementationshinweise

1. **Modularität**: Jedes Modul als eigenständige Komponente mit definierten Interfaces
2. **Skalierung**: Start mit 10⁴–10⁶ Neuronen, skalierbar
3. **Logging**: Spikes, Gewichte, Modulatoren, Replay-Traces, Policy-Decisions
4. **Hardware**: GPU/TPU für Matrixoperationen, neuromorphe Chips optional

---

## Anhang: Nicht im Kandel belegte Konzepte

Die folgenden Konzepte sind optionale Add-ons ohne direkte Kandel-Quelle:

| Konzept | Status | Alternative Quellen |
|---------|--------|---------------------|
| STDP-Formel mit expliziten Zeitkonstanten | Nur konzeptuell in Kandel | Bi & Poo (1998), Markram et al. (1997) |
| 3-Faktor-Lernregel | Impliziert, nicht formal | Frémaux & Gerstner (2016) |
| Predictive Coding | NICHT IM KANDEL | Rao & Ballard (1999), Friston (2010) |
| CLS als benanntes Framework | Konzeptuell beschrieben | McClelland et al. (1995) |
| Spracharchitektur/Switching | NICHT IM KANDEL | Abutalebi & Green (2007) |
| Eligibility Traces | Impliziert durch Dopamin-Delay | Sutton & Barto (2018) |

---

*Spezifikation erstellt basierend auf: Kandel ER, Schwartz JH, Jessell TM, Siegelbaum SA, Hudspeth AJ (2013). Principles of Neural Science, 5th Edition. McGraw-Hill.*
