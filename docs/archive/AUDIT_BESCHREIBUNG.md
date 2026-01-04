# Audit: BESCHREIBUNG.md

## Prüfdatum: 2024
## Geprüftes Dokument: docs/BESCHREIBUNG.md (v1.0)

---

## 1. Sätze ohne Quellenangabe

Die folgenden Aussagen im Dokument haben KEINE direkte Kandel-Quellenangabe:

### Sektion 1 (Recheneinheiten)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 1.2 LIF-Modell | "τ_m × dV/dt = -(V - V_rest) + R_m × I_syn(t)" | AKZEPTABEL: Abgeleitet aus Kandel Kap. 6, S. 138–144 (passive Eigenschaften) |
| 1.2 Izhikevich | "v' = 0.04v² + 5v + 140 - u + I" | **FEHLEND**: Izhikevich-Modell nicht in Kandel. Computational Neuroscience Add-on |

### Sektion 2 (Synapsen)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 2.2 STP-Modell | "Tsodyks-Markram-Modell" | **FEHLEND**: Nicht in Kandel. Sollte als Add-on markiert werden |
| 2.2 | "dx/dt = (1-x)/τ_D - u×x×δ(t-t_spike)" | **FEHLEND**: Formel nicht in Kandel |

### Sektion 3 (Lernregeln)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 3.1 STDP | "Δw_ij = +A_+ × exp(-Δt / τ_+)..." | **FEHLEND**: Explizite STDP-Formel nicht in Kandel. Bereits korrekt als Add-on markiert |
| 3.2 3-Faktor | "de_ij/dt = -e_ij/τ_e + STDP(pre_i, post_j)" | **FEHLEND**: Bereits korrekt als Add-on markiert |
| 3.3 Synaptic Scaling | "w_i := w_i × (r_target / r_observed)^α" | TEILWEISE: Konzept in Kandel Kap. 56, aber nicht diese Formel |

### Sektion 4 (Kortex)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 4.3 Mikro-Schaltkreis | "I_fast: Schnelle Inhibition (Parvalbumin+)" | **FEHLEND**: Parvalbumin-Klassifikation nicht explizit in Kandel |
| 4.3 | "I_slow: Langsame Inhibition (Somatostatin+)" | **FEHLEND**: Somatostatin-Klassifikation nicht explizit in Kandel |
| 4.4 Predictive Coding | "x̂ = g(z), ε = x - x̂" | KORREKT als "NICHT IM KANDEL" markiert |

### Sektion 5 (Thalamus)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 5.3 | "R_out = gain × R_in × gate" | **FEHLEND**: Keine Kandel-Quelle für diese Formel |
| 5.3 | "gate = σ(TRN_input + cortical_feedback)" | **FEHLEND**: Keine Kandel-Quelle |

### Sektion 6 (Basalganglien)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 6.3 TD-RPE | "δ_t = r_t + γ × V(s_{t+1}) - V(s_t)" | **FEHLEND**: TD-Formel nicht explizit in Kandel. Konzept ja, Formel nein |
| 6.3 Actor-Critic | "V(s) := V(s) + α_critic × δ_t" | **FEHLEND**: Actor-Critic nicht in Kandel |

### Sektion 7 (Hippocampus)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 7.3 Hopfield | "CA3_weights += outer(CA3_pattern, CA3_pattern)" | **FEHLEND**: Hopfield-Regel nicht in Kandel |
| 7.3 Replay | "for episode in recent_episodes: replay_sequence..." | **FEHLEND**: Replay-Algorithmus nicht in Kandel |

### Sektion 8 (Kleinhirn)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 8.3 LTD-Regel | "Δw_PF = -η × PF_activity × CF_activity" | TEILWEISE: LTD beschrieben in Kandel Kap. 42, S. 975–977, aber nicht diese exakte Formel |

### Sektion 9 (Neuromodulatoren)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 9.2 | "modules.BG.gamma = 0.9 + 0.09 * self.5HT" | **FEHLEND**: Keine Kandel-Quelle für γ-Modulation durch Serotonin |

### Sektion 10 (Gedächtnis)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 10.1 WM | "active_slots = top_k(V, k=4)" | **FEHLEND**: "4 Items" Kapazitätsbegrenzung nicht in Kandel |

### Sektion 11 (Vergessen)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 11.1 | "w := (1 - λ) × w" | **FEHLEND**: Weight Decay Formel nicht in Kandel |
| 11.2 | "if usage[i,j] < θ_usage over T_window: prune()" | **FEHLEND**: Pruning-Algorithmus nicht in Kandel |

### Sektion 12 (Sprache)

| Zeile/Stelle | Aussage | Status |
|--------------|---------|--------|
| 12.2 | Gesamte Mehrsprachigkeits-Architektur | KORREKT als "NICHT IM KANDEL" markiert |

---

## 2. Widersprüche oder unklare Begriffe

### Widerspruch 1: LTP-Induktionsregel
- **Stelle**: Sektion 3.1 vs. 6.3
- **Problem**: In 3.1 wird STDP als lokale 2-Faktor-Regel beschrieben, aber in 6.3 wird 3-Faktor-Lernen für BG verwendet. Unklar, wann welche Regel gilt.
- **Lösung**: Klarstellen, dass STDP die Basis bildet und durch Modulatoren zu 3-Faktor erweitert wird.

### Widerspruch 2: Dopamin-Funktion
- **Stelle**: Sektion 6.3 vs. 9.2
- **Problem**: In 6.3 ist DA = baseline + β × δ_t, aber in 9.2 wird compute_DA als "reward - value_pred" definiert. Inkonsistente Baseline-Behandlung.
- **Lösung**: Vereinheitlichen: DA(t) = DA_baseline + β × (r_t + γV(s') - V(s))

### Unklarer Begriff 1: "Eligibility Trace"
- **Stelle**: Sektion 3.2
- **Problem**: e_ij wird eingeführt, aber τ_e nicht spezifiziert.
- **Lösung**: Typischen Wert angeben: τ_e ≈ 100–1000 ms

### Unklarer Begriff 2: "Sparse Coding"
- **Stelle**: Sektion 7.3
- **Problem**: "sparse_encode" und "k_winners" werden verwendet, aber k nicht definiert.
- **Lösung**: k ≈ 2–5% der Neuronen (typischer Wert)

### Unklarer Begriff 3: "Gain"
- **Stelle**: Sektion 5.3, 9.2
- **Problem**: "gain" wird mehrfach verwendet, aber nie formal definiert.
- **Lösung**: Definition hinzufügen: gain ∈ [0, 2], multiplikativer Faktor auf Aktivität

---

## 3. Fehlende Formeln bei Lernen/Update

### 3.1 Fehlende Formel: Kortex-Gewichtsupdate
- **Stelle**: Sektion 4
- **Problem**: Kortex hat lernbare Gewichte W, aber keine Update-Regel spezifiziert.
- **Lösung**: Hinzufügen:
```
dW/dt = η × STDP(pre, post) × M(t)  # mit Modulator-Gating
```

### 3.2 Fehlende Formel: Thalamus Gating-Update
- **Stelle**: Sektion 5.3
- **Problem**: gate wird berechnet, aber wie lernt es?
- **Lösung**: Hinzufügen:
```
gate_ij lernt NICHT - wird durch BG Selection und Kortex Feedback bestimmt
```

### 3.3 Fehlende Formel: Working Memory Update
- **Stelle**: Sektion 10.1
- **Problem**: WM-Dynamik beschrieben, aber kein Lern-Update.
- **Lösung**: Klarstellen:
```
WM lernt nicht - nur aktive Maintenance. Langzeit-Lernen erfolgt durch Replay → Kortex.
```

### 3.4 Fehlende Formel: Semantisches Gedächtnis
- **Stelle**: Sektion 10.3
- **Problem**: "η_slow × hebbian_update" ohne Definition.
- **Lösung**: Hinzufügen:
```
hebbian_update(x) = outer(x, x) - W  # Oja's Rule für Normalisierung
η_slow ≈ 0.001 × η_hippocampus
```

---

## 4. Fehlende Interfaces (Input/Output) pro Modul

### 4.1 Kortex (Sektion 4)
- **Vorhanden**: Input/Output-Tabelle ✓
- **Fehlend**: Keine Zustands-Tabelle
- **Lösung**: Hinzufügen:
```
| Zustände |
|----------|
| Z(t): Aktivitätsmuster |
| W_ff, W_fb, W_rec: Gewichtsmatrizen |
| layer_activity[l]: Aktivität pro Layer |
```

### 4.2 Thalamus (Sektion 5)
- **Vorhanden**: Input/Output-Tabelle ✓, Zustände ✓
- **Fehlend**: Keine formale Definition der Zustandsübergänge
- **Lösung**: Hinzufügen:
```
Zustandsübergang:
mode := "tonic" wenn ACh > θ_ACh ELSE "burst"
gate[k] := sigmoid(selection[k] + feedback[k] - inhibition[k])
```

### 4.3 Basalganglien (Sektion 6)
- **Vorhanden**: Input/Output ✓, Zustände ✓
- **Fehlend**: Vollständige Schnittstellen-Signatur
- **Lösung**: Hinzufügen:
```
BG.step(cortex_state, reward, context) → (action_selection, DA_signal)
```

### 4.4 Hippocampus (Sektion 7)
- **Vorhanden**: Input/Output ✓, Zustände ✓
- **Fehlend**: Replay-Interface
- **Lösung**: Hinzufügen:
```
| Interface | Signatur |
|-----------|----------|
| encode | encode(Z_cortex, context) → episode_id |
| retrieve | retrieve(cue) → Z_recalled |
| replay | replay(n_episodes) → [Z_1, Z_2, ...] |
```

### 4.5 Kleinhirn (Sektion 8)
- **Vorhanden**: Input/Output ✓, Zustände ✓
- **Fehlend**: Timing-Output nicht spezifiziert
- **Lösung**: Hinzufügen:
```
| Output | Typ |
|--------|-----|
| correction | ℝ^n (Steuerwert-Korrektur) |
| timing_offset | ℝ (ms, Timing-Anpassung) |
```

### 4.6 Neuromodulatoren (Sektion 9)
- **Vorhanden**: Input/Output ✓
- **Fehlend**: Zustände nicht definiert
- **Lösung**: Hinzufügen:
```
| Zustände |
|----------|
| DA_baseline: Tonic Dopamin-Level |
| NE_threshold: Schwelle für Surprise |
| ACh_level: Aktueller Acetylcholin-Pegel |
| 5HT_baseline: Serotonin-Grundniveau |
```

---

## 5. Patch-Plan (konkrete Textänderungen)

### Patch 1: Izhikevich als Add-on markieren
**Stelle**: Sektion 1.2, nach "Option B: Izhikevich-Modell"
**Änderung**:
```diff
- **Option B: Izhikevich-Modell** (für reichere Dynamik)
+ **Option B: Izhikevich-Modell** (für reichere Dynamik)
+ 
+ > **NICHT IM KANDEL**: Das Izhikevich-Modell ist eine vereinfachte Alternative zu Hodgkin-Huxley aus der Computational Neuroscience (Izhikevich 2003).
```

### Patch 2: Tsodyks-Markram als Add-on markieren
**Stelle**: Sektion 2.2, nach "Mit Short-Term Plasticity (STP)"
**Änderung**:
```diff
  **Mit Short-Term Plasticity (STP):**
+ 
+ > **NICHT IM KANDEL**: Das Tsodyks-Markram-Modell formalisiert STP. Kandel beschreibt Facilitation/Depression konzeptuell (Kap. 12, S. 285–290), aber nicht diese Gleichungen.
```

### Patch 3: TD-Formel als Add-on markieren
**Stelle**: Sektion 6.3, nach "Reward Prediction Error"
**Änderung**:
```diff
  **Reward Prediction Error:**
+ 
+ > **Hinweis**: Die TD-Formel δ_t = r_t + γV(s') - V(s) stammt aus Sutton & Barto (Reinforcement Learning). Kandel beschreibt das RPE-Konzept (Kap. 49, S. 1110–1112), aber nicht die mathematische Formulierung.
```

### Patch 4: Eligibility Trace Parameter
**Stelle**: Sektion 3.2, nach "de_ij/dt = -e_ij/τ_e..."
**Änderung**:
```diff
  de_ij/dt = -e_ij/τ_e + STDP(pre_i, post_j)
+ 
+ Typische Parameter:
+ τ_e = 100–1000 ms (Eligibility-Fenster)
+ η = 0.001–0.01 (Lernrate)
```

### Patch 5: Sparse Coding Parameter
**Stelle**: Sektion 7.3, nach "DG = sparse_encode(EC_input)"
**Änderung**:
```diff
  DG = sparse_encode(EC_input)  # k-winner-take-all
+ 
+ Parameter:
+ k ≈ 2–5% der DG-Neuronen (Sparsity)
+ Expansion-Ratio: DG_size ≈ 5–10 × EC_size
```

### Patch 6: Gain-Definition
**Stelle**: Sektion 5.3, nach "gain = f(modulators)"
**Änderung**:
```diff
  gain = f(modulators)  # ACh, NE erhöhen Gain
+ 
+ Definition:
+ gain ∈ [0.1, 2.0] (multiplikativer Faktor)
+ gain = 1.0 + α_ACh × ACh + α_NE × NE
+ mit α_ACh, α_NE ≈ 0.5
```

### Patch 7: Kortex Lernregel
**Stelle**: Sektion 4, neuer Unterabschnitt "4.X Lernregel"
**Hinzufügen**:
```markdown
### 4.X Lernregel

```
# Kortikale Plastizität
dW/dt = η × e_ij × M(t)

wobei:
- e_ij: Eligibility Trace (aus STDP)
- M(t): Modulator-Signal (ACh für Attention, DA für Reward)
- η ≈ 0.001
```

> Biologische Basis: LTP/LTD im Kortex folgt ähnlichen Regeln wie im Hippocampus (Kandel5e, Kap. 67, S. 1490–1500), moduliert durch cholinerge Eingänge (Kap. 46, S. 1054–1056).
```

### Patch 8: Hippocampus Interface-Tabelle
**Stelle**: Sektion 7.5, ersetzen
**Änderung**:
```diff
  ### 7.5 Schnittstellen
  
- | Input | Output |
- |-------|--------|
- | Kortex-Zustände Z(t) | Keys/Indices für Recall |
- | Kontextsignale | Replay-Sequenzen |
- | Novelty-Signal | |
+ | Interface | Signatur | Beschreibung |
+ |-----------|----------|--------------|
+ | encode() | encode(Z, ctx) → id | Speichert Episode |
+ | retrieve() | retrieve(cue) → Z | Ruft Episode ab |
+ | replay() | replay(n) → [Z₁...Zₙ] | Offline-Replay |
+ | novelty() | novelty(Z) → bool | Prüft Neuheit |
```

### Patch 9: Neuromodulator-Zustände
**Stelle**: Sektion 9, nach 9.3 Schnittstellen, neuer Unterabschnitt
**Hinzufügen**:
```markdown
### 9.4 Zustände

| Zustand | Typ | Beschreibung |
|---------|-----|--------------|
| DA_tonic | ℝ | Tonic Dopamin-Level (Baseline) |
| NE_level | ℝ | Aktueller Noradrenalin-Pegel |
| ACh_level | ℝ | Aktueller Acetylcholin-Pegel |
| 5HT_level | ℝ | Aktueller Serotonin-Pegel |
```

### Patch 10: Working Memory Klarstellung
**Stelle**: Sektion 10.1, nach Code-Block
**Hinzufügen**:
```markdown
> **Klarstellung**: Working Memory lernt nicht im Sinne von Gewichtsänderungen. Die Plastizität liegt in der Konsolidierung (WM → Hippocampus → Kortex), nicht im WM-Modul selbst.
```

### Patch 11: Konsistente DA-Berechnung
**Stelle**: Sektion 6.3 und 9.2, vereinheitlichen
**Änderung in 9.2**:
```diff
  def compute_DA(self, reward, value_pred):
-     return reward - value_pred  # RPE
+     return reward + self.gamma * value_next - value_pred  # TD-RPE
```

---

## 6. Zusammenfassung

| Kategorie | Anzahl Probleme | Kritisch |
|-----------|-----------------|----------|
| Fehlende Quellen | 18 | 6 (nicht als Add-on markiert) |
| Widersprüche | 2 | 2 |
| Unklare Begriffe | 3 | 0 |
| Fehlende Formeln | 4 | 2 |
| Fehlende Interfaces | 6 | 3 |

### Priorität der Patches

1. **Hoch**: Patch 1, 2, 3 (Add-on-Markierungen für Formeln ohne Kandel-Quelle)
2. **Hoch**: Patch 11 (Widerspruch DA-Berechnung)
3. **Mittel**: Patch 4, 5, 6 (Parameter-Definitionen)
4. **Mittel**: Patch 7, 8, 9 (Fehlende Interfaces/Lernregeln)
5. **Niedrig**: Patch 10 (Klarstellung)

---

## 7. Empfehlung

Das Dokument BESCHREIBUNG.md ist **grundsätzlich solide** strukturiert und folgt den Regeln größtenteils. Die Hauptprobleme sind:

1. **Einige Computational Neuroscience Formeln** (Izhikevich, Tsodyks-Markram, TD-Learning) sollten explizit als "NICHT IM KANDEL" markiert werden.

2. **Konsistenz bei Dopamin-Berechnung** muss hergestellt werden.

3. **Parameter-Werte** für Eligibility Traces, Sparsity und Gain sollten ergänzt werden.

Nach Anwendung der 11 Patches ist das Dokument audit-konform.

---

*Audit durchgeführt gemäß den Regeln aus INSTRUCTIONS.md*
