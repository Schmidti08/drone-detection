# Drohnenerkennungssystem

Ein Python-Programm, das Drohnen in einem Video automatisch erkennt und verfolgt — in Echtzeit, auf einem normalen Laptop.

![Demo](demo/demo.mp4)

---

## Was macht das Programm?

Das Programm bekommt ein Video als Eingabe. Es erkennt darin automatisch alle Drohnen, zeichnet einen Rahmen um jede einzelne und vergibt ihr eine feste Nummer (#0, #1, ...). Oben links im Bild steht, wie viele Drohnen gerade gleichzeitig zu sehen sind.

---

## Wie funktioniert es?

Das Programm besteht aus drei Schritten:

**1. Erkennung**  
Ein KI-Modell (YOLOv8) schaut sich jeden Videoframe an und findet heraus, wo im Bild sich Drohnen befinden.

**2. Verfolgung**  
Ein selbst geschriebener Algorithmus merkt sich, welche Drohne in welchem Frame wo war. So bekommt jede Drohne über den gesamten Videoverlauf dieselbe Nummer — auch wenn sie kurz verdeckt war.

**3. Darstellung**  
OpenCV zeichnet die Rahmen und Nummern ins Bild und zeigt das fertige Video in einem Fenster an.

---

## Installation

**Voraussetzungen:** Python 3.11 und Git müssen installiert sein.

```bash
git clone https://github.com/Schmidti08/drone-detection.git
cd drone-detection

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Starten:**

```bash
python run_demo.py
```

Mit `q` wird das Fenster geschlossen.

> Beim ersten Start lädt das Programm das KI-Modell automatisch herunter (~140 MB). Danach funktioniert es auch offline.

---

## Verwendete Technologien

- **Python** — Programmiersprache
- **YOLOv8** — KI-Modell zur Objekterkennung
- **OpenCV** — Bibliothek zur Videoverarbeitung
- **PyTorch** — Rechenkern für das KI-Modell
- **NumPy** — Mathematikbibliothek

---

*Portfolio-Projekt — Tjorben Schmidt, 2026 · Entwickelt mit Unterstützung von Claude (Anthropic) als KI-Assistent.*
