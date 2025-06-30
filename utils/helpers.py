import os

import cv2
import numpy as np
import torch

from detectors.person_detector import detect_people
from features.feature_extractor import FeatureExtractor
from tracking.sort_tracker import init_tracker


def build_feature_dataset(raw_video_dir='data/raw_videos', out_dir='data/features'):
    extractor = FeatureExtractor()
    tracker = init_tracker()

    X, y = [], []  # X = sequence of features, y = labels

    os.makedirs(out_dir, exist_ok=True)

    for video_name in os.listdir(raw_video_dir):
        if not video_name.endswith('.mp4'):
            continue

        video_path = os.path.join(raw_video_dir, video_name)
        cap = cv2.VideoCapture(video_path)
        track_features = {}

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            boxes = detect_people(frame)
            detections = [[x, y, x + w, y + h, 1.0] for (x, y, w, h) in boxes]
            tracked = tracker.update(np.array(detections))

            for x1, y1, x2, y2, obj_id in tracked:
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                feat = extractor.extract(crop)
                if int(obj_id) not in track_features:
                    track_features[int(obj_id)] = []
                track_features[int(obj_id)].append(feat)

        # Save features per person track
        for obj_id, feats in track_features.items():
            if len(feats) >= 10:  # min sequence length
                X.append(feats[:10])  # crop to fixed length
                y.append(1 if "shoplifting" in video_name.lower() else 0)

        cap.release()

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print(f"✅ Extracted {len(X)} sequences. Saving to {out_dir}")
    np.save(os.path.join(out_dir, "X.npy"), X)
    np.save(os.path.join(out_dir, "y.npy"), y)
