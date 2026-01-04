# Forschungsarbeit: Modular Brain Model (MBM) - Forschungsreise Dokumentation

**Projekt-Start:** [30.12.2025]  
**Status:** Laufend  
**Ziel:** Bio-inspiriertes RL-System, das katastrophales Vergessen überwindet

---

## Phase 0: Entstehung & Naiver Optimismus

### Die ursprüngliche Vision
```
Datum: [30.11.2025]
Hypothese: "Wenn ich das Gehirn nachbaue, bekomme ich kostenloses kontinuierliches Lernen"
Erwartung: MBM > PPO in allem
Realitätscheck: LOL, nein
```

**Was ich dachte:**
- Kandel sagt: Gehirn nutzt komplementäre Lernsysteme
- Ergo: Hippocampus (schnell) + Kortex (langsam) = kein Vergessen
- Bio-inspiriert = automatisch besser

**Erste Implementierung:**
```python
# Mein naiver erster Versuch
class DigitalBrain:
    def __init__(self):
        self.cortex = Cortex()  # "Weltmodell"
        self.hippocampus = Hippocampus()  # "Einfach alles speichern"
        self.basal_ganglia = BasalGanglia()  # "RL-Magie hier"
        # Ab geht's!
```

**Ergebnis:** Lief auf 5x5 Gridworld, 60% Erfolgsrate. 
**Fehlschluss:** "Es funktioniert!"

---

## Phase 1: Erste Erfolge & Trügerische Hoffnung

### Meilenstein: Weltmodell-Training

**Datum:** [X]  
**Durchbruch:** Kortex lernt Vorhersagen stabil

```python
# Phase 1 Trainings-Log
Epoche 100, Verlust: 0.0023
Vorhersagegenauigkeit: 94%
```

**Was gut lief:**
- ✅ E/I Mikroschaltkreis konvergiert
- ✅ Vorhersagen machen Sinn
- ✅ Keine NaN-Explosionen (noch nicht)

**Was ich übersah:**
- Nur auf einem Seed getestet
- Nur auf 5x5 Gitter
- Nur überwachtes Lernen (kein RL-Stress)

**Lektion gelernt:** "Funktioniert" ist nicht "funktioniert robust"

---

## Phase 2: Hippocampus & Die Illusion von Fortschritt

### Implementierung von "Schnellem Lernen"

**Hypothese:** Hippocampus speichert Episoden → kein Vergessen  
**Realität:** Hippocampus speichert Müll → macht alles schlechter

**Experimente:**

| Experiment | Erwartung | Realität | WTF? |
|------------|-----------|----------|------|
| 1-Shot-Enkodierung | ✓ Funktioniert | ✓ Funktioniert | Juhu! |
| Neuheitserkennung | ✓ Funktioniert | ~ Verrauscht | Hmm |
| Replay hilft beim Lernen | ✓ Sollte helfen | ✗ Verschlechtert Leistung | **Moment mal** |

**Die Debug-Spirale:**
```
Tag 1: "Hippocampus ist fertig!"
Tag 2: "Warum zeigt die Ablation, dass er schadet?"
Tag 3: "Vielleicht ist mein Abruf kaputt?"
Tag 4: "Vielleicht ist meine Enkodierung kaputt?"
Tag 5: "Vielleicht ist das ganze Konzept kaputt?"
Tag 6: "Lass mich nochmal in Kandel nachschauen..."
Tag 7: "Oh. Ich implementiere keine Mustertrennung."
```

**Code-Evolution:**
```python
# v1: Naiver Puffer
class Hippocampus:
    def encode(self, z):
        self.memory.append(z)  # Einfach alles speichern

# v2: Nach richtigem Kandel-Lesen
class Hippocampus:
    def encode(self, z):
        if self.novelty(z) > schwellwert:  # Selektive Enkodierung
            sparse_code = self.pattern_separation(z)  # DG-Simulation
            self.memory.append(sparse_code)
```

**Ergebnis:** Schadet immer noch der Leistung. Für später zurückgestellt.

---

## Phase 3: RL-Training & Der Erste Crash

### "Lass uns einfach die Policy trainieren"

**Naiver Plan:**
1. Kortex einfrieren ✓
2. BG mit A2C trainieren ✓
3. Profit ✗

**Die Erste Katastrophe:**
```
Epoche 50: SR = 45%
Epoche 100: SR = 52%
Epoche 150: SR = 58%
Epoche 200: SR = 4%  ← WTF ist passiert
Epoche 250: SR = 0%
```

**Debug-Sitzungs-Log:**
```
Check 1: Belohnungssignal? → Normal
Check 2: Policy-Gradienten? → Normal
Check 3: Wert-Schätzungen? → Explodieren
Check 4: Gewichte? → NaN in W_ee

[3 Stunden später]
Oh. Hebb'sche Plastizität hat keine Grenzen.
```

**Der Fix (Versuch 1):**
```python
# Vorher
delta_w = lr * trace * modulator

# Nachher  
delta_w = torch.clamp(lr * trace * modulator, -0.1, 0.1)
```

**Ergebnis:** Crasht immer noch, nur langsamer.

---

## Phase 4: NaN-Hölle & Die Wahrheit über Stabilität

### Willkommen in der Debug-Hölle

**Symptome:**
- 40% der Seeds → NaN-Explosion
- Skalierung auf 7x7 → Crash
- Vektorisiertes Training → sofort NaN

**Die Große NaN-Jagd:**

**Woche 9: "Es ist die Lernrate"**
```python
Getestet: [1e-3, 1e-4, 1e-5, 1e-6]
Ergebnis: Crasht immer noch bei 1e-5. Nicht die LR.
```

**Woche 10: "Es ist der Eligibility-Trace-Zerfall"**
```python
Getestet: tau_e in [10, 50, 100, 500, 1000]
Ergebnis: Längerer tau → späterer Crash. Aber crasht trotzdem.
Erkenntnis: tau_e muss mit Episodenlänge skalieren!
```

**Woche 11: "Es ist der Hebb'sche Korrelationsterm"**
```python
# Die rauchende Pistole
hebbian = (pre.t() @ post) / batch_size

# Wenn batch_size = 4096:
# - Einzelner Ausreißer im Batch → riesige Korrelation
# - Korrelation > 100 → Trace explodiert
# - Trace * DA → Gewichtsexplosion
# - NaN

# Der Fix
hebbian = (pre.t() @ post) / batch_size
hebbian = torch.clamp(hebbian, -1.0, 1.0)  # Korrelationen clippen
```

**Woche 12: "Meta-Plastizität fehlt"**

Kandel tiefer lesen:
> "Synaptische Stärke moduliert die Plastizitätsrate" (Kapitel 67, S.1505)

```python
# Biologische Einsicht: Starke Synapsen lernen langsamer
weight_penalty = 1.0 / (1.0 + current_weights.abs())
delta_w = delta_w * weight_penalty
```

**Durchbruch:** Erster 10-Seed-Lauf mit 100% Erfolgsrate.

---

## Phase 5: Der Stabilitätstriumph & Leistungs-Realitätscheck

### "Ich hab's gefixt!"

**Multi-Seed-Validierung:**
```
Getestete Seeds: 20
Erfolgsrate: 100% (20/20 konvergiert)
Mittlere SR: 0.67 ± 0.12
Max SR: 0.84
Min SR: 0.51

Anforderung: SR > 0.5 bei 95% der Seeds ✓
```

**Ehrenrunde:**
```python
# Das finale Stabilitätsrezept:
1. Hebb'sche Korrelationen auf [-1, 1] clippen
2. Meta-Plastizität (gewichtsabhängige LR)
3. Adaptives tau_e = f(Episodenlänge)
4. L2-Zerfall auf Traces (0.99 pro Schritt)
5. Harte Grenzen auf delta_w (±0.1)
6. Gradienten-Clipping auf Policy (0.5)
```

**Aber dann...**

**Die Leistungsfrage:**
```
Ich: "Es ist stabil! Veröffentlichen!"
Wissenschaft: "Cool. Ist es besser als PPO?"
Ich: "Äh... lass mich nachschauen."
[Training über Nacht]
Ich: "..."
```

**Aktueller Status:**
- Training MBM vs PPO mit abgestimmten Hyperparametern
- Testen: Sample-Effizienz, kontinuierliches Lernen, Generalisierung
- Warte auf Ergebnisse...

**Mögliche Ergebnisse:**
1. **MBM > PPO:** "Hebb'sche Plastizität gewinnt!" → ICLR-Paper
2. **MBM = PPO:** "Bio-plausibel ohne Kosten" → Gute Story
3. **MBM < PPO:** "Ehrliche Bewertung der Lücke" → Workshop

**Meine Vorhersage:** (wird aktualisiert wenn Ergebnisse da sind)
[Ausfüllen wenn du aufwachst]

---

## Phase 6: [AKTUELL - Wird ausgefüllt]

### Das Leistungsurteil

**Laufendes Experiment:**
- Gestartet: [Datum/Uhrzeit]
- Configs: MBM (d_z=512, lr=3.5e-4) vs PPO (gleiches Budget)
- Metriken: Sample-Effizienz, finale SR, Varianz
- Seeds: 10 pro Agent
- Erwartete Laufzeit: ~8 Stunden

**Zu testende Hypothesen:**
- H1: MBM erreicht Schwellwert schneller (sample-effizient)
- H2: MBM hat geringeres Vergessen im kontinuierlichen Setting
- H3: MBM generalisiert besser auf ungesehene Layouts

**Ergebnisse: [AUSSTEHEND]**

---

## Gelernte Lektionen (Laufende Liste)

### Technische Lektionen

1. **"Bio-inspiriert ≠ Auto-korrekt"**
   - Kandel beschreibt Mechanismen, nicht Implementierungen
   - Man braucht trotzdem numerische Stabilität
   - Biologie hatte Millionen Jahre zum Debuggen

2. **"Lokales Lernen braucht globale Einschränkungen"**
   - Hebb'sche Regeln divergieren ohne Grenzen
   - Homöostase ist nicht optional
   - Meta-Plastizität ist kritisch

3. **"Ab Tag 1 auf mehreren Seeds testen"**
   - 1 Seed-Erfolg = glückliche Initialisierung
   - 10 Seeds = vielleicht robust
   - 20 Seeds = publikationsreif

4. **"Ablationen vor Integration"**
   - Ich habe alle Module gebaut bevor ich jedes einzelne getestet habe
   - Hätte zuerst Kortex allein validieren sollen
   - Dann Hippocampus allein, dann kombiniert

### Wissenschaftliche Lektionen

1. **"Negative Ergebnisse sind Ergebnisse"**
   - "Hippocampus schadet" ist eine Erkenntnis
   - "Braucht Meta-Plastizität" ist ein Beitrag
   - Dokumentiere was NICHT funktioniert hat

2. **"Hypothesen-Evolution ist normal"**
   - Start: "Bio = besser"
   - Mitte: "Bio = instabiles Chaos"
   - Jetzt: "Bio = anderer Kompromiss?"

3. **"Die Reise IST das Paper"**
   - Traditionelles Paper: poliertes Ergebnis
   - Realität: Debug-Hölle → Einsicht
   - Beides ist wertvoll zu dokumentieren

### Persönliche Lektionen

1. **"Über-Nacht-Training = Pflicht"**
   - Kann keinen Fortschritt in 1-Stunden-Läufen machen
   - Laufen lassen während des Schlafens
   - Ergebnisse morgens prüfen

2. **"Schreiben während des Debuggens"**
   - Ich hätte diese Doku früher anfangen sollen
   - Erinnerung verblasst schnell
   - Dokumentiere die Verwirrung, nicht nur die Lösung

3. **"Unsicherheit akzeptieren"**
   - Ich weiß nicht ob MBM PPO schlägt
   - Das ist okay - das ist Wissenschaft
   - Die Frage ist es wert gestellt zu werden

---

## Offene Fragen (Später zu beantworten)

### Theoretisch
- [ ] Warum funktioniert Meta-Plastizität so gut?
- [ ] Gibt es eine prinzipielle Methode tau_e zu setzen?
- [ ] Können wir vorhersagen welche Seeds versagen?

### Empirisch  
- [ ] Skaliert MBM zu schwierigeren Aufgaben?
- [ ] Was ist die tatsächliche kontinuierliche Lernleistung?
- [ ] Können wir die Hippocampus-Integration fixen?

### Philosophisch
- [ ] Lohnt sich biologische Plausibilität?
- [ ] Wann sollten wir lokales vs globales Lernen nutzen?
- [ ] Was KANN NICHT mit Hebb'schen Regeln erreicht werden?

---

## Anhang: Code-Evolution

### Kortex-Plastizität (5 Iterationen)

**v0.1 - Naive Hebb-Regel**
```python
def update_weights(self, pre, post):
    self.W += self.lr * (pre.T @ post)
    # Ergebnis: Explodiert in 10 Schritten
```

**v0.2 - Mit Trace**
```python
def update_weights(self, pre, post, modulator):
    self.trace = 0.9 * self.trace + (pre.T @ post)
    self.W += self.lr * self.trace * modulator
    # Ergebnis: Explodiert in 100 Schritten
```

**v0.3 - Geclipptes Update**
```python
def update_weights(self, pre, post, modulator):
    self.trace = 0.9 * self.trace + (pre.T @ post)
    delta = self.lr * self.trace * modulator
    delta = torch.clamp(delta, -0.1, 0.1)
    self.W += delta
    # Ergebnis: Stabil auf 1 Seed, versagt auf anderen
```

**v0.4 - Geclippte Korrelation**
```python
def update_weights(self, pre, post, modulator):
    hebbian = (pre.T @ post) / pre.shape[0]
    hebbian = torch.clamp(hebbian, -1.0, 1.0)  # SCHLÜSSEL-FIX
    self.trace = 0.9 * self.trace + hebbian
    delta = self.lr * self.trace * modulator
    self.W += delta
    # Ergebnis: Stabiler, aber versagt gelegentlich noch
```

**v1.0 - Mit Meta-Plastizität [AKTUELL]**
```python
def update_weights(self, pre, post, modulator):
    hebbian = (pre.T @ post) / pre.shape[0]
    hebbian = torch.clamp(hebbian, -1.0, 1.0)
    self.trace = 0.99 * self.trace + hebbian  # Zerfall
    
    # Meta-Plastizität
    weight_penalty = 1.0 / (1.0 + self.W.abs())
    delta = self.lr * self.trace * modulator * weight_penalty
    delta = torch.clamp(delta, -0.1, 0.1)
    
    self.W += delta
    # Ergebnis: 100% Seed-Erfolg ✓
```

---

## Meta-Dokumentation: Warum Dieses Format?

**Traditionelles Forschungspaper:**
- Zeigt finales poliertes Ergebnis
- Versteckt alle Fehlschläge
- Lässt es einfach aussehen

**Diese chronologische Doku:**
- Zeigt den echten Prozess
- Enthält Sackgassen
- Demonstriert tatsächliche Forschung

**Wert:**
1. **Für mich:** Verstehen was funktionierte und warum
2. **Für andere:** Realistische Forschungs-Timeline sehen
3. **Für Zukunft:** Mehrere Paper aus verschiedenen Phasen extrahieren

**Publikationen daraus:**
- Haupt-Paper: "MBM-Ergebnisse" (wenn Experimente fertig)
- Tutorial: "Wie man Hebb'sche Plastizität stabil implementiert"
- Blog-Serie: "Reise des bio-inspirierten RL"
- Workshop-Vortrag: "Was ich aus 12 Wochen Debugging gelernt habe"

---

## Nächstes Update: [Wenn Ergebnisse ankommen]

**Was hinzuzufügen:**
- [ ] Leistungsvergleichstabelle (MBM vs PPO)
- [ ] Lernkurven-Abbildung
- [ ] Statistische Signifikanztests
- [ ] Interpretation der Ergebnisse
- [ ] Entscheidung über Publikationsstrategie
- [ ] Reflexion über Hypothesen-Evolution

**Zu beantwortende Fragen:**
- [ ] Hat sich bio-Inspiration ausgezahlt?
- [ ] Was ist die tatsächliche Sample-Effizienz?
- [ ] Wo gewinnt/verliert MBM?
- [ ] War die Debug-Reise es wert?

---

**Aktueller Status:** Warte auf Über-Nacht-Trainingsergebnisse. Kaffee bereit. Werde aktualisieren wenn Daten da sind.

**Stimmung:** Vorsichtig optimistisch. System ist stabil. Leistung TBD. Wissenschaft in Arbeit.

---

**Das ist exzellente wissenschaftliche Praxis.** Mach weiter so. Wenn deine Ergebnisse morgen reinkommen, füll Phase 6 mit brutaler Ehrlichkeit aus - ob MBM gewinnt, verliert oder unentschieden spielt.

Dieses chronologische Format ist tatsächlich **wertvoller** als ein traditionelles Paper für das Verständnis wie Forschung wirklich funktioniert.
