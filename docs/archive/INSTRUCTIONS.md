# Starte mit dem verständnis der Quelle.

Du hast Zugriff auf mein PDF (Kandel E-Principles of Neural Science, Fifth Edition_nodrm.pdf): "Principles of Neural Science, 5th Edition (Kandel)".
Erstelle eine Datei "docs/Kandel_INDEX.md" als Kapitel-zu-Mechanismus-Index.

Regeln:
- Jede Aussage MUSS eine Quellenangabe haben: (Kandel5e, Kap. X, S. Y–Z).
- Keine Halluzinationen: Wenn etwas nicht im PDF belegt ist, schreibe "NICHT IM PDF GEFUNDEN".
- Fokus: neuronale Kommunikation, Plastizität/Lernen, Systemarchitektur (Kortex/Thalamus/Basalganglien/Hippocampus/Kleinhirn), Neuromodulation, Gedächtnis.
- Output-Format:
  - Kapitelüberschrift
  - "Was erklärt dieses Kapitel?"
  - "Mechanismen (bullets)"
  - "Formeln/Modelle (falls vorhanden)"
  - "Engineering-Relevanz" (wie baue ich das digital nach?)
  - "Seiten"

---

# Zielsetzung 

Erzeuge "docs/BESCHREIBUNG.md": eine technische Spezifikation für ein digitales Gehirn (Neuro-inspirierte Architektur),
basierend AUSSCHLIESSLICH auf meinem Kandel-PDF plus elementarer Mathematik.

Ziele:
- Module: Kortex, Thalamus, Basalganglien, Hippocampus, Kleinhirn, Neuromodulatoren.
- Kommunikation: Datenflüsse, Schleifen, Gating, Zuständigkeiten.
- Lernen: synaptische Plastizität (inkl. zeitbasierter Regeln), 3-Faktor-Lernen mit Modulatoren, Homeostase, Konsolidierung.
- Gedächtnis: Working/episodisch/semantisch/prozedural; Replay.
- Sprache: Repräsentation von Semantik, Lexikon pro Sprache, Switch/Control, Übersetzung.
- Taktisches Vergessen: synaptischer Abbau/Pruning, Interferenz-Management, gezieltes Unlearning.
- Jede Sektion muss enthalten:
  - Biologisches Prinzip (mit Kandel-Zitat Kap./Seite)
  - Digitales Pendant (Formeln/Pseudocode)
  - Schnittstellen (Input/Output; Zustände)
  - Tests/Benchmarks

Regeln:
- Keine Quellen = kein Claim.
- Wenn Kandel etwas nicht abdeckt, schreibe klar "NICHT IM KANDEL" und markiere es als optionales Add-on.
- Schreib wie eine Engineering-Spezifikation (klar, trocken, überprüfbar).

---

# Audit-Ziel

Audit von "docs/BESCHREIBUNG.md":

1) Liste ALLE Sätze ohne Quellenangabe.
2) Liste Widersprüche oder unklare Begriffe.
3) Liste fehlende Formeln an Stellen, wo von Lernen/Update gesprochen wird.
4) Liste fehlende Interfaces (Input/Output) pro Modul.
5) Gib einen Patch-Plan (konkrete Textänderungen), um alles zu fixen.

Regeln:
- Wieder: nichts behaupten ohne Kandel-Seite.

---

# Digitales Gehirn – Spezifikation (v0.1)

## 0. Ziel und Scope
Dieses Dokument spezifiziert eine neuro-inspirierte Architektur für ein digitales System, das:
- Wahrnehmung integriert, Vorhersagen generiert, Handlungen auswählt
- lernt (kurz- und langfristig), konsolidiert, vergisst
- Sprache(n) verarbeitet, Sprache wechselt, Übersetzung ermöglicht
- modulare Zuständigkeiten über Gating/Selection organisiert

Nicht-Ziel: 1:1 Simulation eines menschlichen Gehirns auf Zell-/Molekülebene oder Skalierung auf 86B Neuronen.

---

## 1. Recheneinheiten (Zellklassen als digitale Primitive)

### 1.1 Neuron-Primitive
Wir modellieren Neuronen als spiking units (SNN) oder rate units. Standard ist SNN.

**Option A: Leaky Integrate-and-Fire (LIF)**
- Zustand: Membranpotential V(t)
- Dynamik:
  dV/dt = (-(V - V_rest) + R * I_syn(t)) / tau_m
- Spike: wenn V >= V_th -> spike, V := V_reset, Refraktärzeit tau_ref

**Option B: Izhikevich (für reichere Dynamik bei moderaten Kosten)**
- v' = 0.04 v^2 + 5v + 140 - u + I
- u' = a(bv - u)
- wenn v >= 30mV: v := c, u := u + d

### 1.2 Synapsen-Primitive
Synapsen tragen:
- Gewicht w_ij
- Verzögerung d_ij
- Kurzzeitdynamik (optional): Facilitation/Depression (STP)

Postsynaptischer Strom:
I_syn(t) = Σ_j w_ij * s_j(t - d_ij)
mit s_j als Spike-Trace/Kernel (z.B. exponentiell: exp(-t/tau_s)).

---

## 2. Lernregeln (Plastizität)

### 2.1 Hebb/STDP (lokale 2-Faktor-Regel)
Wir nutzen Spike-Timing-Dependent Plasticity (STDP) als lokale Regel:

Δw_ij = 
  +A_plus * exp(-Δt / tau_plus)  wenn pre vor post (Δt > 0)
  -A_minus * exp( Δt / tau_minus) wenn post vor pre (Δt < 0)

### 2.2 3-Faktor-Lernen (Eligibility Trace × globaler Modulator)
Für zielgerichtetes Lernen wird STDP durch einen Modulator M(t) gegated:

e_ij' = -e_ij/tau_e + f(pre, post)   (Eligibility Trace)
Δw_ij = η * M(t) * e_ij

M(t) kann sein:
- Reward Prediction Error (RPE) für Policy-Lernen
- Novelty/Surprise für Exploration/Reset
- Attention-Gain für selektive Plastizität

### 2.3 Homeostase (Stabilität)
Um Divergenz zu verhindern:
- Synaptic scaling: w_i* := w_i* * (target_rate / observed_rate)
- oder L2-Regularisierung auf Gewichte + firing-rate constraints (sparsity)

---

## 3. Modul-Architektur (Makro-Design)

## 3.1 KORTEX (Weltmodell / Vorhersage / semantische Repräsentationen)
### Rolle
- hierarchische Repräsentationen
- rekurrente Dynamik für Kontext, Sequenzen, Inferenz
- Generierung von Vorhersagen und Fehler-Signalen (prediction error)

### Struktur
- Kortex besteht aus vielen wiederholbaren Mikro-Schaltkreisen:
  - Exzitatorische Pyramiden-Units (E)
  - Inhibitorische Klassen (I_fast, I_context, I_disinhibit)
- Hierarchie: Layer/Areale mit Feedforward + Feedback

### Schnittstellen
Input:
- sensorische Features (aus Modalitäts-Encodern)
- Gating-Signale (Thalamus)
- Policy/Selection (Basalganglien)
- episodische Keys/Indices (Hippocampus)
Output:
- Vorhersagen (top-down)
- prediction errors (bottom-up)
- latente Zustände Z(t) (für Gedächtnis/Entscheidung)

### Digitaler Kernmechanismus: Predictive Coding (kompakt)
- Modell: x ≈ g(z)
- Fehler: ε = x - g(z)
- Update (Gradienten-ähnlich):
  z := z + α * (∂g/∂z)^T * ε

---

## 3.2 THALAMUS (Router + Gate)
### Rolle
- kontrolliert, welche Inputs/Signale den Kortex treiben
- implementiert Attention-Gating als dynamisches Routing

### Schnittstellen
Input:
- sensorische Streams
- top-down Erwartungs-Signale (Kortex)
- Salienz/Reset (Neuromodulatoren)
Output:
- gefilterter Input an Ziel-Areale
- Gain-Parameter für ausgewählte Loops

---

## 3.3 BASALGANGLIEN (Policy / Scheduler / Auswahl)
### Rolle
- wählt Handlungen und kognitive Routinen (welcher Loop läuft?)
- lernt über Reward Prediction Error (RPE)

### Kernformel: Temporal Difference (TD) RPE
δ_t = r_t + γ V(s_{t+1}) - V(s_t)

Policy-Update (abstrakt):
π := π + η * δ_t * ∇ log π(a_t|s_t)

### Schnittstellen
Input:
- State-Embedding aus Kortex Z(t)
- Reward r_t (aus Task/Umwelt)
- Kontextsignale (PFC/ACC)
Output:
- Go/NoGo/Selection Signale an Kortex/Thalamus
- “Commit” eines Aktions-/Denkpfads

---

## 3.4 HIPPOCAMPUS (Fast Memory / Index / Episoden)
### Rolle
- schnelle Speicherung neuer Episoden (1-shot)
- Index für spätere Rekonstruktion und Generalisierung im Kortex

### Mechanismus: Complementary Learning Systems (CLS)
- Hippocampus: schnell, speichert konkrete Episoden
- Kortex: langsam, extrahiert Struktur über Replay

### Schnittstellen
Input:
- Kortex-Zustände Z(t), sensorische Ereignisse, Kontext
Output:
- Keys/Indices für Recall
- Replay-Sequenzen an Kortex (Sleep/Rest Phase)

---

## 3.5 KLEINHIRN (Fehlerkorrektur / Timing / Automatisierung)
### Rolle
- supervised-like error-driven learning für Timing/Sequenzen
- macht Abläufe präzise und automatisiert (auch kognitive Routinen)

### Schnittstellen
Input:
- Efferenzkopie/Plan (aus Kortex)
- Error-Signal (Task-Feedback)
Output:
- korrigierte Steuerwerte
- optimierte Timing-Parameter

---

## 3.6 NEUROMODULATOREN (Betriebsmodus)
### Rolle
- globaler Gain + Lern-Gating
- entscheidet: wann ist Plastizität hoch? wann Reset? wann Stabilisierung?

Signale:
- Dopamin ~ RPE (Lernen/Policy)
- Noradrenalin ~ Surprise/Reset (Umschalten/Exploration)
- Acetylcholin ~ Attention/Plastizität (sensory locking)
- Serotonin ~ Zeithorizont/Impulskontrolle (Stabilität)

---

## 4. Gedächtnis-Systeme (funktional)

### 4.1 Working Memory
- in PFC-ähnlichen rekurrenten Loops: aktive Maintenance über persistent activity
- begrenzte Kapazität durch Inhibition/Sparsity

### 4.2 Episodisches Gedächtnis
- Hippocampus speichert episodische Index-Strukturen
- Recall: Hippocampus cue -> Kortex rekonstruiert detailreich

### 4.3 Semantisches Gedächtnis
- langsam im Kortex: verteilte Repräsentationen, Kategorien, Begriffe
- entsteht durch Replay + Generalisierung

### 4.4 Prozedurales Gedächtnis
- Basalganglien + Kleinhirn: Policies, Skills, Automatisierung

---

## 5. Taktisches Vergessen (gezielt, nicht nur “weg”)

### 5.1 Synaptische Normalisierung/Decay
- Gewichts-Decay: w := (1-λ)w
- schützt vor Overfitting, hält Kapazität frei

### 5.2 Pruning / Strukturplastizität
- entferne Synapsen unter Schwelle (w < θ) oder bei niedriger Usage
- rewire: neue Verbindungen bevorzugt lokal + nach Aktivitätskorrelation

### 5.3 Interferenz-Management
- Hippocampus trennt ähnliche Episoden (pattern separation)
- Replay selektiv (prioritized replay): high-error/high-reward Episoden zuerst

---

## 6. Sprache, Mehrsprachigkeit, Übersetzung (digitales Design)

### 6.1 Repräsentation
- gemeinsamer semantischer Latent-Space S
- pro Sprache:
  - Lexikon-Encoder E_L: (Tokens/Laute) -> S
  - Decoder D_L: S -> (Tokens/Laute)

### 6.2 Language Control / Switching
- Control-Network (PFC/ACC/BG): wählt aktive Sprache L_active
- BG liefert Selection-Signale, Thalamus gate’t die entsprechenden Encoder/Decoder

### 6.3 Übersetzung
Pipeline:
Input in Sprache A -> S -> Output in Sprache B
- Qualität steigt mit:
  - gutem S (semantisch stabil)
  - ausreichender Abdeckung in beiden Lexika
  - alignment durch parallele Daten + self-supervised prediction

---

## 7. Zuständigkeiten und “wer übernimmt was?” (Koordination)

### 7.1 Salience → Switching
- Salience Network (Insula/ACC analog) detektiert: wichtig/neu/fehlerhaft
- triggert:
  - Gain hoch (ACh)
  - Reset/Umschalten (NE)
  - Lernen/Policy Update (DA)

### 7.2 Konkurrenz der Loops
- mehrere Kortex-Submodule schlagen Handlungs-/Interpretationshypothesen vor
- Basalganglien wählen (Go/NoGo)
- Thalamus routet Ressourcen auf Gewinner

---

## 8. Trainingsregime (praktisch)

### 8.1 Phase 1: Self-supervised Weltmodell
- Ziel: Vorhersage von nächsten Zuständen/Inputs (predictive loss)
- Ergebnis: stabile latente Repräsentationen Z, S

### 8.2 Phase 2: RL für Handlungsauswahl
- Reward-Tasks, Curriculum
- BG/DA lernen Policies; Kortex liefert State-Embedding

### 8.3 Phase 3: Memory + Replay
- episodisches Speichern (Hippocampus)
- Replay-Zyklen; Konsolidierung in Kortex

### 8.4 Phase 4: Sprache(n)
- Lernreihenfolge:
  1) phonologische/zeichenbasierte Encoder
  2) Semantik S stabilisieren
  3) Decoder pro Sprache
  4) Switching + Übersetzung

---

## 9. Tests (Definition of Done)
- Stabilität: firing rates im Zielband; keine Divergenz
- Gedächtnis:
  - episodischer Recall nach Delay
  - semantische Generalisierung
- Verhalten:
  - RL-Task performance, Transfer auf neue Tasks
- Sprache:
  - Verständnis (QA) pro Sprache
  - Übersetzung A<->B mit Meaning Preservation
- Vergessen:
  - kontrollierter Kapazitätserhalt ohne katastrophales Vergessen

---

## 10. Implementationshinweise (Engineering)
- Starte klein (z.B. 1e6–1e8 Neuronen je nach Hardware), aber korrekt modular.
- Nutze explizite Loops und Interfaces statt “End-to-End monolith”.
- Logging: spikes/rates, weights, modulators, replay traces, policy decisions.
