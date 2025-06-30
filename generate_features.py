# generate_features.py
"""
Build X.npy / y.npy from every video under data/raw_videos/**.
Label rule:
    parent folder contains "shoplift"  → 1
    otherwise                          → 0
"""

import os
from pathlib import Path

import cv2
import numpy as np

from detectors.yolo_detector import detect_people  # use YOLO detector
from features.feature_extractor import FeatureExtractor
from tracking.sort_tracker import init_tracker

RAW_DIR   = Path("data/raw_videos")
OUT_DIR   = Path("data/features")
SEQ_LEN   = 10                     # frames per sequence for LSTM
OUT_DIR.mkdir(parents=True, exist_ok=True)

extractor = FeatureExtractor()
tracker   = init_tracker()

X, y = [], []

print("🔍  Scanning videos under", RAW_DIR)
for video_path in RAW_DIR.rglob("*.mp4"):                # recursive search
    label = 1 if "shoplift" in video_path.parts[-2].lower() else 0
    print(f"🎥  {video_path.relative_to(RAW_DIR)}  → label={label}")

    cap = cv2.VideoCapture(str(video_path))
    track_feats = {}

    while True:
        ret, frame = cap.read()
        if not ret: break

        boxes = detect_people(frame)
        dets  = [[x, y, x+w, y+h, 1.0] for (x, y, w, h) in boxes]
        tracked = tracker.update(np.array(dets)) if len(dets) else []

        for x1, y1, x2, y2, tid in tracked:
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            feat = extractor.extract(crop)
            if feat is None: continue
            tid = int(tid)
            track_feats.setdefault(tid, []).append(feat)

    cap.release()

    # slice every track into fixed‑length sequences
    for feats in track_feats.values():
        if len(feats) >= SEQ_LEN:
            for i in range(0, len(feats) - SEQ_LEN + 1):
                X.append(feats[i:i+SEQ_LEN])
                y.append(label)

print(f"\n✅  Collected {len(X)} sequences ({sum(y)} shoplift / {len(y)-sum(y)} normal)")
np.save(OUT_DIR / "X.npy", np.array(X, dtype=np.float32))
np.save(OUT_DIR / "y.npy", np.array(y, dtype=np.float32))
print(f"💾  Saved features to {OUT_DIR}")
