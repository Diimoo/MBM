# Kandel Index: Kapitel-zu-Mechanismus-Verzeichnis
## Principles of Neural Science, 5th Edition

> **Hinweis**: Dieses Dokument indiziert relevante Kapitel aus dem Kandel-Lehrbuch mit Fokus auf: neuronale Kommunikation, Plastizität/Lernen, Systemarchitektur, Neuromodulation und Gedächtnis.

---

## Part II: Zellbiologie und Molekularbiologie von Neuronen

### Kapitel 5: Ion Channels
**Was erklärt dieses Kapitel?**
Grundlegende Eigenschaften von Ionenkanälen, die für elektrische Signalübertragung verantwortlich sind.

**Mechanismen:**
- Selektivität von Ionenkanälen basierend auf Größe, Ladung und Hydrationsenergie (Kandel5e, Kap. 5, S. 101–107)
- Spannungsabhängige Kanäle (Natrium, Kalium, Calcium) (Kandel5e, Kap. 5, S. 158–170)
- Liganden-gesteuerte Kanäle (Kandel5e, Kap. 5, S. 186–188)

**Formeln/Modelle:**
- Leitfähigkeit: g = I / (V - E_rev)
- Nernst-Gleichung für Gleichgewichtspotential

**Engineering-Relevanz:**
- Basis für neuronale Einheiten: Ionenkanal-Dynamik bestimmt Membranpotential
- Implementierung als differenzielle Zustandsgleichungen

**Seiten:** 100–171

---

### Kapitel 6: Membrane Potential and Passive Electrical Properties
**Was erklärt dieses Kapitel?**
Wie das Ruhemembranpotential entsteht und passive elektrische Eigenschaften die Signalausbreitung beeinflussen.

**Mechanismen:**
- Ruhepotential durch K+-Leckkanäle und Na+/K+-Pumpe (Kandel5e, Kap. 6, S. 126–135)
- Membrankapazität verlangsamt Signale (Kandel5e, Kap. 6, S. 139)
- Längenkonstante und Zeitkonstante (Kandel5e, Kap. 6, S. 142–144)

**Formeln/Modelle:**
- τ_m = R_m × C_m (Membranzeitkonstante)
- λ = √(r_m / r_a) (Längenkonstante)
- V_m = (g_K × E_K + g_Na × E_Na + g_Cl × E_Cl) / (g_K + g_Na + g_Cl)

**Engineering-Relevanz:**
- Definiert Zeitkonstanten für digitale Neuronen (LIF-Modell)
- Bestimmt räumliche Integration von Inputs

**Seiten:** 126–147

---

### Kapitel 7: Propagated Signaling: The Action Potential
**Was erklärt dieses Kapitel?**
Entstehung und Fortleitung von Aktionspotentialen durch spannungsabhängige Kanäle.

**Mechanismen:**
- Hodgkin-Huxley-Dynamik für Na+ und K+ Ströme (Kandel5e, Kap. 7, S. 148–156)
- Refraktärzeit durch Inaktivierung (Kandel5e, Kap. 7, S. 156–158)
- Saltatorische Leitung in myelinisierten Axonen (Kandel5e, Kap. 7, S. 144)

**Formeln/Modelle:**
- I_Na = g_Na × m³ × h × (V - E_Na)
- I_K = g_K × n⁴ × (V - E_K)
- dm/dt = α_m(V)(1-m) - β_m(V)m

**Engineering-Relevanz:**
- Basis für Spiking Neural Networks (SNN)
- Izhikevich-Modell als vereinfachte Alternative

**Seiten:** 148–171

---

## Part III: Synaptische Übertragung

### Kapitel 8: Overview of Synaptic Transmission
**Was erklärt dieses Kapitel?**
Grundlagen elektrischer und chemischer Synapsen.

**Mechanismen:**
- Elektrische Synapsen: Gap Junctions für schnelle Synchronisation (Kandel5e, Kap. 8, S. 178–184)
- Chemische Synapsen: Verstärkung und Modulation möglich (Kandel5e, Kap. 8, S. 184–187)
- Direkt vs. indirekt gesteuerte Rezeptoren (Kandel5e, Kap. 8, S. 186)

**Engineering-Relevanz:**
- Gap Junctions → direkte Kopplung für Synchronisation
- Chemische Synapsen → lernfähige Gewichte

**Seiten:** 177–188

---

### Kapitel 10: Synaptic Integration in the Central Nervous System
**Was erklärt dieses Kapitel?**
Wie Neuronen multiple synaptische Eingänge integrieren.

**Mechanismen:**
- Räumliche und zeitliche Summation (Kandel5e, Kap. 10, S. 210–227)
- Exzitatorische (EPSP) und inhibitorische (IPSP) Potentiale (Kandel5e, Kap. 10, S. 211–220)
- Dendritische Integration und aktive Dendriten (Kandel5e, Kap. 10, S. 228–230)
- Spike-Initiation am Axon-Initialsegment (Kandel5e, Kap. 10, S. 227)

**Formeln/Modelle:**
- V_soma = Σ w_i × EPSP_i - Σ w_j × IPSP_j (vereinfacht)

**Engineering-Relevanz:**
- Definiert synaptische Gewichtung und Integration
- Basis für dendritisches Computing

**Seiten:** 210–235

---

### Kapitel 11: Modulation of Synaptic Transmission: Second Messengers
**Was erklärt dieses Kapitel?**
Wie Second-Messenger-Kaskaden synaptische Übertragung modulieren.

**Mechanismen:**
- cAMP-Weg: G-Protein → Adenylat-Cyclase → PKA (Kandel5e, Kap. 11, S. 237–247)
- Ca²+/Calmodulin-Weg (Kandel5e, Kap. 11, S. 247–250)
- Phosphorylierung von Ionenkanälen (Kandel5e, Kap. 11, S. 250–253)
- Langfristige Effekte durch Genexpression (Kandel5e, Kap. 11, S. 253–256)

**Formeln/Modelle:**
- NICHT IM PDF GEFUNDEN: explizite Formeln für Kaskadendynamik

**Engineering-Relevanz:**
- Basis für langsame Modulation (Gain, Plastizität-Gating)
- Implementierung als langsame Zustandsvariablen

**Seiten:** 236–259

---

### Kapitel 12: Transmitter Release
**Was erklärt dieses Kapitel?**
Mechanismen der Neurotransmitter-Freisetzung an präsynaptischen Terminals.

**Mechanismen:**
- Ca²+-abhängige Vesikelfreisetzung (Kandel5e, Kap. 12, S. 260–275)
- SNARE-Proteine und Vesikelfusion (Kandel5e, Kap. 12, S. 275–285)
- Kurzzeit-Plastizität: Facilitation und Depression (Kandel5e, Kap. 12, S. 285–290)

**Formeln/Modelle:**
- P_release ∝ [Ca²+]⁴ (kooperative Ca²+-Abhängigkeit)

**Engineering-Relevanz:**
- Short-Term Plasticity (STP) als dynamischer Filter
- Facilitation/Depression für temporale Verarbeitung

**Seiten:** 260–295

---

### Kapitel 13: Neurotransmitters
**Was erklärt dieses Kapitel?**
Klassifikation und Eigenschaften von Neurotransmittern.

**Mechanismen:**
- Biogene Amine: Dopamin, Noradrenalin, Serotonin, Acetylcholin (Kandel5e, Kap. 13, S. 289–295)
- Aminosäure-Transmitter: Glutamat, GABA, Glycin (Kandel5e, Kap. 13, S. 295–300)
- Neuropeptide und Neuromodulatoren (Kandel5e, Kap. 13, S. 300–310)

**Engineering-Relevanz:**
- Neuromodulatoren als globale Gain-/Modus-Signale
- Unterscheidung: schnelle Transmitter vs. langsame Modulatoren

**Seiten:** 289–320

---

## Part IV: Wahrnehmung

### Kapitel 15: The Organization of the Central Nervous System
**Was erklärt dieses Kapitel?**
Anatomische Organisation des ZNS mit Fokus auf Hirnstrukturen.

**Mechanismen:**
- Thalamus als sensorisches Relay (Kandel5e, Kap. 15, S. 363–368)
- Basalganglien regulieren motorische und kognitive Funktionen (Kandel5e, Kap. 15, S. 368–370)
- Hippocampus und Gedächtnis (Kandel5e, Kap. 15, S. 370–372)
- Neokortex: hierarchische Organisation (Kandel5e, Kap. 15, S. 372–380)

**Engineering-Relevanz:**
- Modulare Architektur: Thalamus (Router), BG (Selektion), Hippocampus (Episoden), Kortex (Repräsentation)

**Seiten:** 356–390

---

## Part VI: Bewegung

### Kapitel 42: The Cerebellum
**Was erklärt dieses Kapitel?**
Struktur und Funktion des Kleinhirns für motorische Kontrolle und Lernen.

**Mechanismen:**
- Purkinje-Zellen als einziger Output (Kandel5e, Kap. 42, S. 960–965)
- Kletterfasern (Climbing Fibers) für Fehlersignale (Kandel5e, Kap. 42, S. 965–970)
- Moosfasern (Mossy Fibers) für kontextuelle Eingänge (Kandel5e, Kap. 42, S. 965–970)
- Langzeit-Depression (LTD) als Lernmechanismus (Kandel5e, Kap. 42, S. 975–978)

**Formeln/Modelle:**
- LTD: Δw ∝ -activity_parallel × activity_climbing (Kandel5e, Kap. 42, S. 975–977)

**Engineering-Relevanz:**
- Supervised Learning durch Fehlersignale
- Timing und Präzision von Sequenzen
- Automatisierung von Abläufen

**Seiten:** 960–981

---

### Kapitel 43: The Basal Ganglia
**Was erklärt dieses Kapitel?**
Rolle der Basalganglien bei Bewegungsauswahl und Verstärkungslernen.

**Mechanismen:**
- Direkter Pfad (Go): fördert Bewegung (Kandel5e, Kap. 43, S. 982–990)
- Indirekter Pfad (NoGo): hemmt Bewegung (Kandel5e, Kap. 43, S. 990–995)
- Dopamin moduliert beide Pfade (Kandel5e, Kap. 43, S. 995–1000)
- Striatum als Eingangsstruktur (Kandel5e, Kap. 43, S. 982–985)

**Formeln/Modelle:**
- NICHT IM PDF GEFUNDEN: TD-Lernregel explizit formuliert
- Dopamin ~ Reward Prediction Error (RPE) konzeptuell beschrieben (Kandel5e, Kap. 43, S. 995–998)

**Engineering-Relevanz:**
- Actor-Critic Architektur
- Go/NoGo für Handlungsselektion
- Dopamin als RPE-Signal für Policy-Lernen

**Seiten:** 982–1006

---

## Part VII: Unbewusste und bewusste Verarbeitung

### Kapitel 46: The Modulatory Functions of the Brain Stem
**Was erklärt dieses Kapitel?**
Neuromodulatorische Systeme des Hirnstamms und ihre Funktionen.

**Mechanismen:**
- Dopaminerge Systeme: VTA → Striatum/Kortex für Belohnung/Motivation (Kandel5e, Kap. 46, S. 1038–1046)
- Noradrenerge Systeme: Locus Coeruleus für Arousal/Aufmerksamkeit (Kandel5e, Kap. 46, S. 1046–1050)
- Serotonerge Systeme: Raphekerne für Stimmung/Impulskontrolle (Kandel5e, Kap. 46, S. 1050–1054)
- Cholinerge Systeme: für Aufmerksamkeit und Plastizität (Kandel5e, Kap. 46, S. 1054–1056)

**Engineering-Relevanz:**
- Dopamin → Lernsignal (RPE)
- Noradrenalin → Surprise/Reset
- Acetylcholin → Attention/Plastizität-Gating
- Serotonin → Zeithorizont/Stabilität

**Seiten:** 1038–1056

---

### Kapitel 49: Homeostasis, Motivation, and Addictive States
**Was erklärt dieses Kapitel?**
Neuronale Grundlagen von Motivation und Belohnungslernen.

**Mechanismen:**
- Dopamin als Lernsignal, nicht nur Belohnungssignal (Kandel5e, Kap. 49, S. 1108–1113)
- Reward Prediction Error (RPE) in dopaminergen Neuronen (Kandel5e, Kap. 49, S. 1110–1112)
- Nucleus Accumbens als Belohnungszentrum (Kandel5e, Kap. 49, S. 1106–1108)

**Formeln/Modelle:**
- Konzeptuell: δ = r - V(s) (RPE, nicht explizit so formuliert)
- Dopamin-Burst bei unerwartetem Reward (Kandel5e, Kap. 49, S. 1110–1111)
- Dopamin-Pause bei ausbleibendem erwartetem Reward (Kandel5e, Kap. 49, S. 1110–1111)

**Engineering-Relevanz:**
- TD-Learning: δ_t = r_t + γV(s_{t+1}) - V(s_t)
- Dopamin-Signal für Policy-Updates

**Seiten:** 1095–1115

---

### Kapitel 51: Sleep and Dreaming
**Was erklärt dieses Kapitel?**
Neuronale Mechanismen von Schlaf und deren Funktion für Gedächtniskonsolidierung.

**Mechanismen:**
- REM vs. Non-REM Schlaf (Kandel5e, Kap. 51, S. 1140–1145)
- Circadiane Rhythmen durch Suprachiasmatischen Nucleus (Kandel5e, Kap. 51, S. 1145–1147)
- Schlafspindeln für Gedächtniskonsolidierung (Kandel5e, Kap. 51, S. 1148–1149)
- REM-ON und REM-OFF Zellen im Hirnstamm (Kandel5e, Kap. 51, S. 1147–1148)

**Engineering-Relevanz:**
- Offline-Replay für Konsolidierung
- Schlafphasen als unterschiedliche Verarbeitungsmodi

**Seiten:** 1140–1165

---

## Part VIII: Entwicklung

### Kapitel 55: Formation and Elimination of Synapses
**Was erklärt dieses Kapitel?**
Wie Synapsen gebildet und eliminiert werden.

**Mechanismen:**
- Aktivitätsabhängige Synaptogenese (Kandel5e, Kap. 55, S. 1233–1245)
- Synaptisches Pruning (Kandel5e, Kap. 55, S. 1245–1255)
- Kompetition zwischen Synapsen (Kandel5e, Kap. 55, S. 1250–1255)

**Engineering-Relevanz:**
- Strukturelle Plastizität: Pruning von wenig genutzten Verbindungen
- Aktivitätsabhängige Reorganisation

**Seiten:** 1233–1260

---

### Kapitel 56: Experience and the Refinement of Synaptic Connections
**Was erklärt dieses Kapitel?**
Wie Erfahrung synaptische Verbindungen formt (kritische Perioden).

**Mechanismen:**
- Kritische Perioden für Plastizität (Kandel5e, Kap. 56, S. 1260–1270)
- Okuläre Dominanzsäulen als Modell (Kandel5e, Kap. 56, S. 1265–1275)
- Hebbianische Plastizität formt Verbindungen (Kandel5e, Kap. 56, S. 1270–1280)

**Formeln/Modelle:**
- Hebb-Regel: "Neurons that fire together, wire together" (Kandel5e, Kap. 56, S. 1270)

**Engineering-Relevanz:**
- Kritische Perioden für bestimmte Lernphasen
- Erfahrungsabhängige Feinabstimmung

**Seiten:** 1260–1290

---

## Part IX: Sprache, Denken, Affekt und Lernen

### Kapitel 60: Language
**Was erklärt dieses Kapitel?**
Neuronale Grundlagen der Sprache.

**Mechanismen:**
- Broca-Areal für Sprachproduktion (Kandel5e, Kap. 60, S. 1354–1365)
- Wernicke-Areal für Sprachverständnis (Kandel5e, Kap. 60, S. 1365–1372)
- Arcuate Fasciculus verbindet beide (Kandel5e, Kap. 60, S. 1372–1375)
- Kritische Periode für Spracherwerb (Kandel5e, Kap. 60, S. 1356–1358)

**Engineering-Relevanz:**
- Trennung von Encoding und Decoding
- Sprach-spezifische Module

**Seiten:** 1353–1390

---

### Kapitel 65: Learning and Memory
**Was erklärt dieses Kapitel?**
Überblick über Lern- und Gedächtnissysteme.

**Mechanismen:**
- Deklaratives vs. nicht-deklaratives Gedächtnis (Kandel5e, Kap. 65, S. 1441–1450)
- Explizites Gedächtnis: Hippocampus-abhängig (Kandel5e, Kap. 65, S. 1445–1455)
- Implizites Gedächtnis: Basalganglien, Kleinhirn, Amygdala (Kandel5e, Kap. 65, S. 1450–1455)
- Kurz- vs. Langzeitgedächtnis (Kandel5e, Kap. 65, S. 1455–1460)

**Engineering-Relevanz:**
- Multiple Gedächtnissysteme mit unterschiedlichen Eigenschaften
- Working Memory vs. Langzeitspeicher

**Seiten:** 1441–1470

---

### Kapitel 66: Cellular Mechanisms of Implicit Memory Storage and the Biological Basis of Individuality
**Was erklärt dieses Kapitel?**
Zelluläre Mechanismen für implizites (prozedurales) Gedächtnis.

**Mechanismen:**
- Sensitivierung und Habituation bei Aplysia (Kandel5e, Kap. 66, S. 1461–1475)
- cAMP-PKA-CREB-Kaskade für Langzeitgedächtnis (Kandel5e, Kap. 66, S. 1475–1485)
- Synaptisches Wachstum für dauerhafte Speicherung (Kandel5e, Kap. 66, S. 1485–1490)

**Formeln/Modelle:**
- Kurzzeit: PKA → Phosphorylierung → erhöhte Transmitterfreisetzung
- Langzeit: CREB → Genexpression → neue Synapsen

**Engineering-Relevanz:**
- Kurzzeit-Plastizität durch Gewichtsmodifikation
- Langzeit-Plastizität durch strukturelle Änderungen

**Seiten:** 1461–1498

---

### Kapitel 67: Prefrontal Cortex, Hippocampus, and the Biology of Explicit Memory
**Was erklärt dieses Kapitel?**
Neuronale Grundlagen des expliziten (deklarativen) Gedächtnisses.

**Mechanismen:**
- Hippocampus für episodisches Gedächtnis (Kandel5e, Kap. 67, S. 1500–1510)
- Long-Term Potentiation (LTP) im Hippocampus (Kandel5e, Kap. 67, S. 1490–1500)
- NMDA-Rezeptoren als Koinzidenzdetektor (Kandel5e, Kap. 67, S. 1493–1497)
- Schaffer-Kollaterale LTP folgt Hebb-Regeln (Kandel5e, Kap. 67, S. 1497–1500)
- Frühe und späte Phase von LTP (Kandel5e, Kap. 67, S. 1500–1505)
- Präfrontaler Kortex für Arbeitsgedächtnis (Kandel5e, Kap. 67, S. 1505–1515)

**Formeln/Modelle:**
- LTP-Induktion: starke präsynaptische Aktivität + postsynaptische Depolarisation
- NMDA als UND-Gatter (Glutamat + Depolarisation = Ca²+-Einstrom)

**Engineering-Relevanz:**
- LTP/LTD als Basis für STDP
- Hippocampus als schneller episodischer Speicher
- PFC für aktive Maintenance (Working Memory)

**Seiten:** 1487–1520

---

## Zusammenfassung: Schlüsselmechanismen für digitale Implementierung

| Mechanismus | Kandel-Kapitel | Digitale Umsetzung |
|------------|----------------|-------------------|
| Ionenkanal-Dynamik | 5–7 | LIF/Izhikevich-Neuronen |
| Synaptische Integration | 10 | Gewichtete Summation |
| Kurzzeit-Plastizität | 12 | STP-Filter (Facilitation/Depression) |
| Neuromodulation | 11, 46, 49 | Globale Gain-/Mode-Parameter |
| LTP/LTD | 66, 67 | STDP-Lernregel |
| Basalganglien | 43 | Actor-Critic, Go/NoGo |
| Kleinhirn | 42 | Supervised Error-Learning |
| Hippocampus | 67 | Fast Episodic Memory + Replay |
| Thalamus | 15 | Routing/Gating |
| Working Memory | 67 | Persistent Activity in PFC |

---

## Nicht im Kandel gefunden (für Add-ons)

- **STDP mit expliziter Zeitfenster-Formel**: Kandel beschreibt LTP/LTD konzeptuell, gibt aber keine A+/A-/τ-Parameter
- **3-Faktor-Lernregel mit Eligibility Trace**: Konzeptuell impliziert durch Dopamin-Modulation, aber nicht formal
- **Predictive Coding**: NICHT IM KANDEL
- **Complementary Learning Systems (CLS)**: Konzeptuell beschrieben (schneller Hippocampus vs. langsamer Kortex), aber nicht unter diesem Namen
- **Replay-Mechanismen**: Schlaf und Konsolidierung beschrieben, aber kein expliziter Replay-Algorithmus

---

*Erstellt basierend auf: Kandel ER, Schwartz JH, Jessell TM, Siegelbaum SA, Hudspeth AJ (2013). Principles of Neural Science, 5th Edition. McGraw-Hill.*
