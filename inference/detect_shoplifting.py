import cv2
import numpy as np
import torch
from detectors.person_detector import detect_people
from sort_tracker import Sort
from analyzer.behavior_scorer import BehaviorAnalyzer
from features.feature_extractor import FeatureExtractor
from model.sequence_classifier import SequenceClassifier
import os, json

video_path = 'data/raw_videos/shoplifting_1.mp4'
out_dir = 'outputs/annotated_videos/'
os.makedirs(out_dir, exist_ok=True)
cap = cv2.VideoCapture(video_path)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(f'{out_dir}/result.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

extractor = FeatureExtractor()
tracker = Sort()
model = SequenceClassifier()
model.load_state_dict(torch.load('model/shoplifting_lstm.pth'))
model.eval()

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

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {int(obj_id)}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    out.write(frame)
    cv2.imshow('Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

final_scores = {}
for obj_id, feats in track_features.items():
    sequence = torch.tensor([feats], dtype=torch.float32)
    with torch.no_grad():
        score = model(sequence).item()
    final_scores[obj_id] = score

with open('outputs/scores.json', 'w') as f:
    json.dump(final_scores, f, indent=2)
print("Scores:", final_scores)
