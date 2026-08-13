# Drohnenerkennungssystem

Echtzeit-Drohnenerkennung und Multi-Objekt-Tracking mit einem spezialisierten YOLOv8-Modell und einem selbst geschriebenen IoU-Tracker — läuft auf normaler Consumer-Hardware ohne dedizierte GPU.

![Demo](demo/demo.mp4)

---

## Funktionen

- **Echtzeit-Erkennung** — YOLOv8x, auf Drohnendaten trainiert, via HuggingFace (`doguilmak/Drone-Detection-YOLOv8x`)
- **Multi-Objekt-Tracking** — Selbst geschriebener IoU-Tracker mit stabilen IDs (#0, #1, …) über alle Frames
- **Hardware-Erkennung** — Wählt automatisch Apple Silicon (MPS), CUDA oder CPU
- **Drohnenzähler** — Live-Overlay mit Anzahl gleichzeitig getrackter Objekte
- **Frame-Skipping** — Inferenz nur jeden 3. Frame für flüssige Wiedergabe auf CPU

---

## Architektur

```
Videodatei (MP4)
    │
    ▼
DroneDetector          src/detector.py
  YOLOv8x via          ├── wählt automatisch MPS / CUDA / CPU
  ultralyticsplus       └── gibt Liste von {bbox, conf} pro Frame zurück
    │
    ▼
Tracker                src/tracker.py
  IoU-basiertes        ├── ordnet Detektionen bekannten Tracks zu
  Multi-Objekt-        ├── vergibt stabile IDs über Frames hinweg
  Tracking             └── löscht Tracks nach max_age unsichtbaren Frames
    │
    ▼
Visualizer             src/visualizer.py
  OpenCV-              ├── zeichnet Bounding Boxes
  Darstellung          ├── beschriftet jeden Track mit seiner ID
                       └── zeigt Drohnenzähler als Overlay
    │
    ▼
cv2.imshow-Fenster
```

---

## Setup

**Voraussetzungen:** Python 3.11, Git

```bash
git clone https://github.com/Schmidti08/drone-detection.git
cd drone-detection

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Starten:**

```bash
python run_demo.py
```

Mit `q` wird das Fenster geschlossen.

> Das Modell (`doguilmak/Drone-Detection-YOLOv8x`) wird beim ersten Start automatisch von HuggingFace heruntergeladen (~140 MB). Einmalig ist eine Internetverbindung nötig.

---

## Projektstruktur

```
drone-detection/
├── run_demo.py          # Einstiegspunkt — OpenCV-Fenster
├── requirements.txt
├── src/
│   ├── detector.py      # YOLOv8-Wrapper mit Geräteauswahl
│   ├── tracker.py       # selbst geschriebener IoU-Tracker
│   └── visualizer.py    # Bounding Boxes + Labels + Drohnenzähler
└── demo/
    ├── drone_video2.mp4 # Testvideo
    └── create_gif.py    # Hilfsskript für Demo-GIF-Export
```

---

## Tech Stack

| Bibliothek | Zweck |
|---|---|
| `ultralytics` / `ultralyticsplus` | YOLOv8-Inferenz |
| `opencv-python` | Video-Ein-/Ausgabe und Darstellung |
| `PyTorch` | ML-Backend (MPS / CUDA / CPU) |
| `NumPy` | Numerische Berechnungen im Tracker |
| `huggingface-hub` | Modell-Download |

---

## Designentscheidungen

**Selbst geschriebener IoU-Tracker statt ByteTrack/SORT**  
Der Tracker umfasst ~50 Zeilen reines Python. Jede Zeile ist bewusst gewählt und erklärbar. Eine Drittanbieter-Bibliothek wäre eine Black Box — für ein Portfolio-Projekt ist Transparenz wichtiger als marginale Tracking-Genauigkeit.

**Keine Bedrohungsbewertung (NIEDRIG / MITTEL / HOCH)**  
Monokulare 2D-Videodaten liefern keine Tiefeninformation, keine reale Geschwindigkeit und keine räumliche Position. Eine Bewertung auf Basis von Pixeldaten wäre irreführend. In echten Systemen wäre das die Aufgabe von Radar- oder Tiefensensor-Integration.

**Keine Drohnentyp-Klassifizierung (FPV vs. Multicopter)**  
Öffentlich verfügbare Modelle unterscheiden Drohnentypen nicht zuverlässig. Eine erzwungene Klassifizierung würde die Konfidenz senken, ohne die Korrektheit zu erhöhen.

**Frame-Skipping (jeden 3. Frame)**  
YOLOv8x auf jedem Frame auszuführen ist auf CPU zu langsam für flüssige Wiedergabe. Das Überspringen von Frames hält das Fenster responsiv, während der Tracker die Kontinuität zwischen den Inferenz-Aufrufen sicherstellt.

---

*Portfolio-Projekt — Tjorben Schmidt, 2026*
