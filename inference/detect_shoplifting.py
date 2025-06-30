# inference/detect_shoplifting.py
import json
import os
import sys
from pathlib import Path

# ------------------------------------------------------------------
# Make project root importable
# ------------------------------------------------------------------
sys.path.append(str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from detectors.yolo_detector import detect_people
from features.feature_extractor import FeatureExtractor
from model.sequence_classifier import SequenceClassifier
from tracking.sort_tracker import init_tracker

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
VIDEO_PATH = "data/raw_videos/normal/normal-2.mp4"
OUT_DIR    = Path("outputs/annotated_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

THRESH = 0.80                # NEW: probability ≥ THRESH → "shoplifting"

# ------------------------------------------------------------------
# I/O setup
# ------------------------------------------------------------------
cap = cv2.VideoCapture(str(VIDEO_PATH))
if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

width, height = int(cap.get(3)), int(cap.get(4))
out_vid = cv2.VideoWriter(
    str(OUT_DIR / "result.avi"),
    cv2.VideoWriter_fourcc(*"XVID"),
    int(cap.get(cv2.CAP_PROP_FPS) or 30),
    (width, height),
)

# ------------------------------------------------------------------
# Load models
# ------------------------------------------------------------------
extractor = FeatureExtractor()
tracker   = init_tracker()
model     = SequenceClassifier()
model.load_state_dict(torch.load("model/shoplifting_lstm.pth", map_location="cpu"))
model.eval()

track_features = {}

# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    boxes = detect_people(frame)
    detections = [[x, y, x + w, y + h, 1.0] for (x, y, w, h) in boxes]
    tracked = tracker.update(np.array(detections)) if len(detections) else []

    for x1, y1, x2, y2, tid in tracked:
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        feat = extractor.extract(crop)
        if feat is None:
            continue

        tid = int(tid)
        track_features.setdefault(tid, []).append(feat)

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tid}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    out_vid.write(frame)

cap.release()
out_vid.release()
cv2.destroyAllWindows()

# ------------------------------------------------------------------
# Score each track with the LSTM + label conversion
# ------------------------------------------------------------------
labels = {}                                   # NEW: store textual labels
for tid, feats in track_features.items():
    seq_np = np.array(feats, dtype=np.float32)          # faster than list→tensor
    seq    = torch.from_numpy(seq_np).unsqueeze(0)
    with torch.no_grad():
        p = model(seq).item()
    labels[tid] = "shoplifting" if p >= THRESH else "normal"

# ------------------------------------------------------------------
# Save / print
# ------------------------------------------------------------------
with open("outputs/scores.json", "w") as f:
    json.dump(labels, f, indent=2)

print("✅  Finished. Labels saved to outputs/scores.json")
print("Final decision per ID:", labels)
