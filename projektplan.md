# Projektplan: Drohnenerkennungssystem

> Erstellt: Juni 2026  
> Ziel: Portfolio-Hauptprojekt für Bewerbung duales Studium, Start September 2026  
> Zielunternehmen: Hensoldt, Rheinmetall, Airbus Defence, Diehl Defence

---

## 1. Projektziel

Bau eines Systems das Drohnen in einem Video automatisch erkennt, verfolgt und deren Bewegungsverhalten analysiert.

**Warum dieses Projekt?**
Drohnenerkennung ist genau das Problem das Firmen wie Hensoldt, Rheinmetall und Diehl Defence gerade lösen. Mit diesem Projekt zeigst du, dass du Computer Vision verstehst, mit modernen ML-Werkzeugen umgehen kannst und ein vollständiges System von Grund auf aufbauen kannst — nicht nur ein Tutorial nachgebaut hast.

**Bewerbungsrelevanz:**
- Zeigt technisches Verständnis für einen hochrelevanten Rüstungsmarkt
- GitHub-Repo mit laufendem Code und Demo-Video ist konkreter Gesprächseinstieg im Interview
- Das Projekt ist erklärbar: du kennst jede Zeile und jede Designentscheidung

---

## 2. Was gebaut wird

### Systemübersicht

```
Eingabe: Video-Datei (mp4)
    ↓
Drohnenerkennung (YOLOv8)
    ↓
Multi-Objekt-Tracking (IoU-Tracker)
    ↓
Bewegungsanalyse (Geschwindigkeit, Richtung)
    ↓
Ausgabe: Streamlit-Dashboard (Browser) + Demo-GIF (GitHub)
```

### Funktionen im Detail

**Drohnenerkennung**
- Jede Drohne im Bild wird mit einer Bounding Box markiert
- Konfidenz-Score wird angezeigt (z.B. "drone 0.87")
- Eine Klasse: `drone` — keine Typklassifizierung (Multicopter/FPV), weil öffentlich verfügbare Pre-Trained-Modelle das nicht zuverlässig leisten

**Multi-Objekt-Tracking**
- Jede erkannte Drohne bekommt eine stabile ID über alle Frames (#0, #1, …)
- Hinter jeder Drohne wird die Flugbahn der letzten Sekunden als Linie gezeichnet

**Bewegungsanalyse**
- Pixelgeschwindigkeit jeder Drohne (px/frame)
- Richtungswechsel erkennbar aus Trajektorienverlauf
- Zähler: Wieviele Drohnen gleichzeitig im Bild

**Was bewusst NICHT gebaut wird:**
- Bedrohungsbewertung (LOW/MEDIUM/HIGH) — in monokularen 2D-Bilddaten nicht valide. Entfernung, echte Geschwindigkeit und Flugrichtung im Raum sind ohne Radar oder Tiefenkamera nicht messbar. Im Interview: "Das wäre in echten Systemen Aufgabe von Radar-Integration — ich habe das bewusst weggelassen statt etwas Falsches zu implementieren." Das zeigt Ingenieursdenken.
- Webcam-Live-Feed — auf einer Webcam sind keine echten Drohnen. Für die Demo ist ein Test-Video sinnvoller.
- Drohnentyp-Klassifizierung — braucht spezialisierte Datensätze, die kostenlos nicht in ausreichender Qualität verfügbar sind

---

## 3. Tech-Stack mit Begründungen

| Technologie | Warum |
|---|---|
| **Python 3.11** | Einzige sinnvolle Wahl — ultralytics, OpenCV, Streamlit laufen alle nur auf Python. Industriestandard für ML. |
| **ultralytics (YOLOv8n)** | YOLO ist das verbreitetste Objekterkennungsframework in der Praxis. Die `n`-Variante (nano) läuft auf CPU (~10–15 fps), braucht keine GPU. Kann später durch Fine-Tuned-Gewichte ausgetauscht werden ohne Codeänderung. |
| **OpenCV** | Standard-Bibliothek für Video-Processing. Liest Video-Dateien Frame für Frame, zeichnet Bounding Boxes und Trajektorien. |
| **NumPy** | Numerische Berechnungen für Tracker (IoU, Geschwindigkeit). Standard in jedem ML-Projekt. |
| **Streamlit** | Schnellste Weg zu einer Browser-Oberfläche ohne Frontend-Kenntnisse. Python-Code → fertige Web-App. Für Portfolio-Demos ideal. |
| **eigener IoU-Tracker (~60 Zeilen)** | Bewusst keine Tracking-Bibliothek (ByteTrack, SORT): Selbst geschriebener Tracker ist verständlich und erklärbar. Im Interview kann jede Zeile begründet werden. Bibliothek wäre Black Box. |

**Nicht verwendet und warum:**
- FastAPI/REST — überkomplex für ein Demo-Projekt
- Docker — kein Mehrwert für lokale Portfolio-Demo
- PyTorch direkt — ultralytics abstrahiert das sinnvoll

---

## 4. Datenquellen mit Begründungen

### Erkennungsmodell (Phase 1)

**Primäroption — HuggingFace:**
```python
from ultralytics import YOLO
model = YOLO("keremberke/yolov8n-drone-detection")
```
Fertiges YOLOv8n-Modell, bereits auf Drohnen trainiert, direkt ladbar. Kein eigenes Training nötig.

**Alternative — Roboflow Universe:**
Auf universe.roboflow.com nach "drone detection" suchen, Filter: YOLOv8-Format, >1000 Bilder. Gewichte herunterladen und lokal laden:
```python
model = YOLO("models/drone_detector.pt")
```

**Warum fertiges Modell in Phase 1:**
Ziel von Phase 1 ist die komplette Pipeline zum Laufen zu bringen. Eigenes Training dauert Stunden und bringt in dieser Phase keinen Mehrwert. Das Modell ist als Parameter austauschbar — wenn später Fine-Tuning gemacht wird, ändert sich nur der Modellpfad, nicht der restliche Code.

### Demo-Video

YouTube-Clips mit Drohnenaufnahmen (Suche: "drone footage 4K", "FPV drone flight"). Download mit `yt-dlp`:
```bash
yt-dlp -o demo/test_drone.mp4 "URL"
```
Alternativ: direkt als `.mp4` downloaden und nach `demo/test_drone.mp4` legen.

### Optional: Fine-Tuning (Phase 3)

Datensatz von Roboflow Universe (YOLOv8-Format, >1000 Bilder), Training über Google Colab (kostenlose T4-GPU, ~15 Minuten).

---

## 5. Phasenplan

### Phase 0 — Vorbereitung (einmalig, vor Phase 1)

**Ziel:** Entwicklungsumgebung läuft, GitHub-Account existiert.

**Schritte:**

1. Python 3.11 installieren: python.org/downloads → "Add Python to PATH" ankreuzen
   ```bash
   python3 --version   # sollte "Python 3.11.x" ausgeben
   ```

2. Cursor installieren: cursor.com → Download (Code-Editor mit KI-Hilfe)

3. GitHub-Account anlegen: github.com → "Sign up"
   - Benutzername: professionell (z.B. `tjorben-schmidt` oder `tschmidt-dev`)

4. Git prüfen:
   ```bash
   git --version   # meistens auf Mac vorinstalliert
   ```

---

### Phase 1 — Pipeline aufsetzen (Wochenende 1–2)

**Ziel:** Komplette Pipeline läuft — Detektion + Tracking + Visualisierung auf einem Test-Video.

**Demo-Moment:** Ein Fenster öffnet sich. Drohnen werden mit Bounding Boxes und stabilen IDs (#0, #1 …) verfolgt. Hinter jeder Drohne ist eine Linie die ihre Flugbahn zeigt.

**Projektordner anlegen:**

Cursor öffnen → File → Open Folder → neuen Ordner `drone-detection` anlegen.

Im Terminal (`Ctrl+ö` in Cursor):
```bash
mkdir -p data models src demo docs
touch src/__init__.py src/detector.py src/tracker.py src/visualizer.py
touch requirements.txt run_demo.py
```

**Virtuelle Umgebung:**
```bash
python3 -m venv .venv
source .venv/bin/activate   # du siehst (.venv) am Zeilenanfang
```

**Pakete installieren:**
```bash
pip install ultralytics opencv-python numpy streamlit yt-dlp
pip freeze > requirements.txt
```

**`src/detector.py`:**
```python
from ultralytics import YOLO


class DroneDetector:
    def __init__(self, model_path="keremberke/yolov8n-drone-detection", confidence=0.4):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({"bbox": (x1, y1, x2, y2), "conf": conf})
        return detections
```

**`src/tracker.py`:**
```python
import numpy as np


class SimpleTracker:
    """IoU-basiertes Multi-Objekt-Tracking.
    
    Idee: Wenn eine erkannte Box in Frame N stark mit einer Box aus Frame N-1
    überlappt (hoher IoU-Wert), ist es vermutlich dasselbe Objekt → gleiche ID.
    """

    def __init__(self, iou_threshold=0.3, max_lost=30):
        self.tracks = {}
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost  # Frames bis ein verlorener Track gelöscht wird

    def _iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0

    def update(self, detections):
        matched = set()
        for det in detections:
            best_id, best_iou = None, self.iou_threshold
            for tid, track in self.tracks.items():
                if tid in matched:
                    continue
                iou = self._iou(det["bbox"], track["bbox"])
                if iou > best_iou:
                    best_iou, best_id = iou, tid

            cx = (det["bbox"][0] + det["bbox"][2]) // 2
            cy = (det["bbox"][1] + det["bbox"][3]) // 2

            if best_id is not None:
                self.tracks[best_id]["bbox"] = det["bbox"]
                self.tracks[best_id]["lost"] = 0
                self.tracks[best_id]["history"].append((cx, cy))
                if len(self.tracks[best_id]["history"]) > 60:
                    self.tracks[best_id]["history"].pop(0)
                matched.add(best_id)
            else:
                self.tracks[self.next_id] = {
                    "bbox": det["bbox"],
                    "lost": 0,
                    "history": [(cx, cy)],
                    "conf": det["conf"],
                }
                matched.add(self.next_id)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in matched:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    del self.tracks[tid]

        return self.tracks
```

**`src/visualizer.py`:**
```python
import cv2
import numpy as np


COLOR = (0, 200, 255)


def draw_tracks(frame, tracks):
    for tid, track in tracks.items():
        x1, y1, x2, y2 = track["bbox"]
        history = track["history"]

        # Pixelgeschwindigkeit berechnen
        if len(history) >= 2:
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            speed = int(np.sqrt(dx**2 + dy**2))
            label = f"#{tid}  {speed}px/f"
        else:
            label = f"#{tid}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)

        for i in range(1, len(history)):
            cv2.line(frame, history[i-1], history[i], COLOR, 1)

    # Zähler oben links
    cv2.putText(frame, f"Drohnen: {len(tracks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR, 2)

    return frame
```

**`run_demo.py`:**
```python
import cv2
from src.detector import DroneDetector
from src.tracker import SimpleTracker
from src.visualizer import draw_tracks

detector = DroneDetector()
tracker = SimpleTracker()

cap = cv2.VideoCapture("demo/test_drone.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    frame = draw_tracks(frame, tracks)

    cv2.imshow("Drone Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Starten:
```bash
python run_demo.py
```

**Häufige Fehler:**

| Problem | Lösung |
|---|---|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` — venv aktiv? |
| Fenster öffnet sich nicht (macOS) | `pip uninstall opencv-python-headless && pip install opencv-python` |
| `demo/test_drone.mp4 not found` | Datei liegt nicht im richtigen Ordner |
| Modell lädt nicht (HuggingFace) | Internetverbindung prüfen, einmalig ~6MB Download |

---

### Phase 2 — Dashboard + GitHub-Finish (Wochenende 3–4)

**Ziel:** Interaktives Streamlit-Dashboard im Browser, Demo-GIF, sauberes GitHub-Repo.

**Demo-Moment:** Browser öffnet sich auf localhost:8501. Video hochladen, System läuft live im Browser. GitHub-Repo ist öffentlich mit Demo-GIF im README.

**`dashboard.py`:**
```python
import streamlit as st
import cv2
import tempfile
from src.detector import DroneDetector
from src.tracker import SimpleTracker
from src.visualizer import draw_tracks

st.set_page_config(page_title="Drone Detection System", layout="wide")
st.title("Drone Detection & Tracking")
st.markdown("Real-time drone tracking with YOLOv8 and IoU-based multi-object tracking.")

col1, col2 = st.columns([3, 1])

with col2:
    st.header("Settings")
    confidence = st.slider("Detection Confidence", 0.1, 0.9, 0.4)
    uploaded = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

with col1:
    if uploaded:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(uploaded.read())
            tmp_path = f.name

        detector = DroneDetector(confidence=confidence)
        tracker = SimpleTracker()
        cap = cv2.VideoCapture(tmp_path)
        stframe = st.empty()
        max_simultaneous = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections)
            max_simultaneous = max(max_simultaneous, len(tracks))
            frame = draw_tracks(frame, tracks)
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_column_width=True)

        cap.release()
        st.success(f"Fertig. Max. gleichzeitige Drohnen: {max_simultaneous}")
    else:
        st.info("Video hochladen um die Erkennung zu starten.")
```

Starten:
```bash
streamlit run dashboard.py
```

**Demo-GIF erstellen:**
```bash
pip install imageio[ffmpeg]
```

`demo/create_gif.py`:
```python
import cv2
import imageio
from src.detector import DroneDetector
from src.tracker import SimpleTracker
from src.visualizer import draw_tracks

detector = DroneDetector()
tracker = SimpleTracker()
cap = cv2.VideoCapture("demo/test_drone.mp4")
frames = []

while cap.isOpened() and len(frames) < 150:  # erste 5 Sekunden bei 30fps
    ret, frame = cap.read()
    if not ret:
        break
    detections = detector.detect(frame)
    tracks = tracker.update(detections)
    frame = draw_tracks(frame, tracks)
    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

cap.release()
imageio.mimsave("demo/demo.gif", frames, fps=15)
print("demo/demo.gif erstellt.")
```

**GitHub-Repo:**
```bash
git init
git add .
git commit -m "feat: drone detection system with YOLOv8 and IoU tracking"
```

Dann auf github.com: New Repository → Name `drone-detection` → **kein** "Initialize with README" → Anweisungen folgen:
```bash
git remote add origin https://github.com/DEIN_USERNAME/drone-detection.git
git branch -M main
git push -u origin main
```

**`README.md` (Englisch — wichtig für internationale Firmen wie Airbus):**

```markdown
# Drone Detection & Tracking System

Real-time drone detection and multi-object tracking using YOLOv8 and a custom IoU-based tracker.

![Demo](demo/demo.gif)

## Features
- **Real-time detection** — YOLOv8n fine-tuned on drone datasets
- **Multi-object tracking** — Custom IoU tracker with trajectory visualization
- **Movement analysis** — Per-drone speed (px/frame) and trajectory history
- **Interactive dashboard** — Streamlit web UI with video upload

## Setup
```bash
git clone https://github.com/DEIN_USERNAME/drone-detection.git
cd drone-detection
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py          # OpenCV window
streamlit run dashboard.py  # Browser dashboard
```

## Architecture
```
Video → DroneDetector (YOLOv8n) → SimpleTracker (IoU) → Visualizer → Streamlit UI
```

## Design Decisions
- **No threat assessment**: Monocular 2D video does not provide depth or real-world velocity — threat assessment would require radar integration.
- **Custom tracker over ByteTrack**: Keeps the codebase transparent and explainable.

## Tech Stack
Python 3.11 · ultralytics · OpenCV · NumPy · Streamlit
```

**Commit-Strategie (wichtig für GitHub-Profil):**
Nicht alles auf einmal committen. Kleine logische Commits zeigen echte Entwicklung:
- `feat: add drone detector wrapper`
- `feat: add IoU multi-object tracker`
- `feat: add trajectory visualization`
- `feat: add streamlit dashboard`
- `docs: add README with demo gif`

---

### Phase 3 — Fine-Tuning (optional, nur wenn Zeit vorhanden)

**Wann sinnvoll:** Phase 1 + 2 komplett fertig, GitHub-Repo sauber, noch Zeit vor Bewerbungsstart.

**Mehrwert:** mAP@0.5-Wert im README nennen können. Zeigt vollständigen ML-Workflow.

**Voraussetzung:** Google-Account für Colab (kostenlose T4-GPU).

**Schritte:**
1. Roboflow-Account anlegen: roboflow.com → universe.roboflow.com → "drone detection" suchen → Datensatz mit >1000 Bildern + YOLOv8-Format wählen → Download-Code kopieren
2. Google Colab öffnen: colab.research.google.com → Neue Notebook → GPU aktivieren (Laufzeit → Laufzeittyp ändern → T4 GPU)
3. Training (~15 Minuten auf Colab-GPU):
   ```python
   !pip install ultralytics roboflow
   from roboflow import Roboflow
   rf = Roboflow(api_key="DEIN_KEY")
   dataset = rf.workspace("WS").project("PROJEKT").version(1).download("yolov8")
   
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   results = model.train(data="dataset/data.yaml", epochs=50, imgsz=640, batch=16)
   ```
4. `best.pt` herunterladen → nach `models/drone_fine_tuned.pt`
5. In `detector.py`: `model_path="models/drone_fine_tuned.pt"` als Default setzen
6. mAP@0.5-Wert aus Ausgabe ins README eintragen

---

## 6. Projektstruktur (GitHub-Repo)

```
drone-detection/
├── README.md              ← Englisch, mit Demo-GIF
├── requirements.txt
├── run_demo.py            ← Schnellstart ohne Dashboard
├── dashboard.py           ← Streamlit-UI
├── src/
│   ├── __init__.py
│   ├── detector.py        ← YOLOv8-Wrapper
│   ├── tracker.py         ← IoU-Tracker
│   └── visualizer.py      ← Zeichnen + Bewegungsanalyse
├── demo/
│   ├── test_drone.mp4     ← Test-Video (nicht ins Repo pushen wenn >50MB)
│   ├── demo.gif           ← Ins Repo pushen, für README
│   └── create_gif.py
├── models/                ← .pt-Dateien (nicht pushen — .gitignore)
└── docs/                  ← Optional: technische Notizen
```

`.gitignore`:
```
.venv/
models/*.pt
demo/test_drone.mp4
__pycache__/
*.pyc
```

---

## 7. Zeitplan

| Zeitraum | Was |
|---|---|
| **Juni 2026** | Phase 0 + Phase 1 (Pipeline läuft) |
| **Juli 2026 (Sommerferien)** | Phase 2 (Dashboard + GitHub fertig) |
| **August 2026** | Optional Phase 3 (Fine-Tuning) + README polieren |
| **September 2026** | Bewerbungen rausschicken mit GitHub-Link |

---

## 8. Was du im Vorstellungsgespräch sagst

**Zur Frage "Was haben Sie in Ihrer Freizeit gemacht?":**

> "Ich habe ein Echtzeit-Drohnenerkennungssystem mit Python und YOLOv8 gebaut. Das System verfolgt mehrere Drohnen gleichzeitig mit einem selbst geschriebenen Tracking-Algorithmus und visualisiert Flugbahnen in Echtzeit. Ich habe dabei bewusst auf Bedrohungsbewertung verzichtet — aus monokularen 2D-Bilddaten ist das nicht valide, das wäre Aufgabe von Radar-Integration. Das Projekt ist auf GitHub unter [link] einsehbar, inklusive Demo-Video."

**Mögliche Folgefragen:**

*"Warum YOLOv8?"*
> "YOLOv8 ist state-of-the-art für Echtzeit-Objekterkennung und läuft auf normaler Hardware ohne GPU. Die nano-Variante schafft ~15 fps auf CPU — ausreichend für die Demo."

*"Wie funktioniert der Tracker?"*
> "IoU-basiert: Wenn eine erkannte Box im aktuellen Frame stark mit einer Box aus dem vorherigen Frame überlappt, ist es vermutlich dasselbe Objekt und bekommt dieselbe ID. Ich habe das selbst implementiert statt eine Bibliothek zu nehmen, damit ich es vollständig verstehe und erklären kann."

*"Was würden Sie als nächstes verbessern?"*
> "Radar-Integration für echte Entfernungsmessung und Bedrohungsbewertung. Oder ein eigenes Fine-Tuning auf militärisch relevante Drohnentypen — FPV-Drohnen sehen anders aus als kommerzielle DJI-Drohnen."
