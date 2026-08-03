import numpy as np
import torch
from ultralyticsplus import YOLO


def _resolve_device() -> str:      #Methode um zu überprüfen 
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU via Metal
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DroneDetector:
    def __init__(self, confidence=0.5, imgsz=480):
        self.model = YOLO("doguilmak/Drone-Detection-YOLOv8x")
        self.model.fuse()
        self.device = _resolve_device()
        self.confidence = confidence
        self.imgsz = imgsz
        self._warmup()

    def _warmup(self):
        print("Warming up YOLO...") # Info-Ausgabe
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8) #erstellt ein schwarzes Bild 
        self.model(             #schickt schwarzes bild durch das YOLO-Netz
            dummy,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

    def detect(self, frame):
        results = self.model(
            frame,
            conf=self.confidence,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append({"bbox": (x1, y1, x2, y2), "conf": conf})
        return detections
