import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import imageio
from tqdm import tqdm
from src.detector import DroneDetector
from src.tracker import Tracker
from src.visualizer import draw_tracks

detector = DroneDetector()
tracker = Tracker()
cap = cv2.VideoCapture(str(ROOT / "demo/drone_video2.mp4"))
fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int((0 * 60 + 53) * fps)  
end_frame = int((1 * 60 + 3) * fps)  
total_frames = end_frame - start_frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

frames = []
frame_count = 0
tracks = {}

with tqdm(total=total_frames, unit="frame", desc="Rendering") as progress:
    while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 3 == 0:
            detections = detector.detect(frame)
            tracks = tracker.update(detections)

        frame = draw_tracks(frame, tracks)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_count += 1
        progress.update(1)

cap.release()
output = ROOT / "demo/demo2.mp4"
print("Saving video...")
imageio.mimsave(output, frames, fps=fps)

print(f"{output} erstellt.")
