import cv2
from src.detector import DroneDetector
from src.tracker import Tracker
from src.visualizer import draw_tracks

detector = DroneDetector()
tracker = Tracker()

cap = cv2.VideoCapture("demo/drone_video2_fixed.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"FPS{fps}")
cap.set(cv2.CAP_PROP_POS_FRAMES, int((0 * 60 + 53) * fps)) 
end_frame = int((1 * 60 + 3) * fps) 

frame_count = 0
tracks = {}

while cap.isOpened() and cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 3 == 0:
        detections = detector.detect(frame)
        tracks = tracker.update(detections)

    frame = draw_tracks(frame, tracks)
    frame_count += 1

    cv2.imshow("Drone Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
