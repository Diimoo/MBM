# Workshop Submission Plan (5 Wochen)

**Ziel:** Einreichung eines Papers bei einem relevanten Workshop (z.B. NeurIPS Workshop on Biological & Artificial RL) basierend auf den aktuellen, ehrlichen Ergebnissen zum "Variance-Stability Trade-off".

---

## Woche 1: Stabilisierung & Proof-of-Concept

- **[ ] Aufgabe 1: Stabilisierungs-Fixes implementieren**
  - Implementierung von Weight Clipping, Plasticity Gating und NaN-Checks.
  - Ziel: Reduzierung der Varianz und Verhinderung von katastrophalen Ausfällen.

- **[ ] Aufgabe 2: Proof-of-Concept Tests**
  - Durchführung der Kernexperimente (CFI-Protokoll) auf einer kleinen Anzahl von Seeds (z.B. 3 Seeds).
  - Ziel: Überprüfen, ob die Fixes die Stabilität verbessern, ohne die positiven Ergebnisse (Backward Transfer) zu zerstören.

- **[ ] Aufgabe 3: Verifikation**
  - Überprüfung der Logs, um sicherzustellen, dass die Clipping- und Gating-Mechanismen wie erwartet funktionieren.
  - Ziel: Sicherstellen, dass die technischen Implementierungen korrekt sind, bevor die finalen, rechenintensiven Experimente gestartet werden.

---

## Woche 2-3: Finale Experimente & Statistische Analyse

- **[ ] Aufgabe 1: Finale Experimente durchführen**
  - Durchführung der vollständigen Experiment-Suite: 10 Seeds × 3 Task-Transitionen × 3 Modelle (MBM-stabilisiert, MBM-baseline, PPO).
  - Ziel: Generierung der finalen, robusten Rohdaten für die Publikation.

- **[ ] Aufgabe 2: Datenexport**
  - Export aller relevanten Metriken (Success Rates, CFI, etc.) in ein sauberes CSV-Format: `results/final_experiments.csv`.
  - Ziel: Eine zentrale, nachvollziehbare Datenquelle für die Analyse schaffen.

- **[ ] Aufgabe 3: Statistische Tests**
  - Durchführung von statistischen Signifikanztests (z.B. Mann-Whitney-U-Test) für die Haupt-Claims (z.B. MBM vs. PPO auf ähnlichen vs. unähnlichen Tasks).
  - Ziel: Untermauerung der Kernaussagen des Papers mit statistischer Evidenz.

---

## Woche 4: Analyse & Erstellung der Abbildungen

- **[ ] Aufgabe 1: Publikations-Abbildungen erstellen**
  - Erstellung von ca. 3-5 zentralen Abbildungen für das Paper.
    - **Fig. 1:** CFI-Vergleich (MBM vs. PPO) über Task-Transitionen.
    - **Fig. 2:** Varianz-Boxplots zur Visualisierung der Instabilität.
    - **Fig. 3:** Task-abhängige Performance-Heatmap oder Tradeoff-Scatterplot.
  - Ziel: Visuelle Darstellung der Kernergebnisse, die die Story des Papers erzählen.

- **[ ] Aufgabe 2: Ergebnisse interpretieren**
  - Detaillierte Analyse der Abbildungen und statistischen Tests.
  - Ziel: Formulierung der Kernaussagen für den Results- und Discussion-Teil des Papers.

---

## Woche 5: Verfassen und Einreichen des Papers

- **[ ] Aufgabe 1: Paper-Entwurf schreiben**
  - Verfassen des gesamten Papers (Methods, Results, Discussion) basierend auf der neuen, ehrlichen Narrative.
  - Ein besonderer Fokus liegt auf einem transparenten "Limitations"-Abschnitt.
  - Ziel: Ein vollständiger Rohentwurf des Papers.

- **[ ] Aufgabe 2: Internes Review & Überarbeitung**
  - Das Paper von Kollegen oder Mentoren gegenlesen lassen.
  - Einarbeitung des Feedbacks und finale Überarbeitung.
  - Ziel: Verbesserung der Qualität und Klarheit des Papers.

- **[ ] Aufgabe 3: Einreichung**
  - Finale Formatierung des Papers gemäß den Workshop-Vorgaben.
  - Einreichung beim Ziel-Workshop (z.B. NeurIPS).
  - Ziel: Das Projekt erfolgreich zur wissenschaftlichen Begutachtung einreichen.
