from ultralytics import YOLO

_model = YOLO("yolov8n.pt")  # nano model, fast and CPU-compatible

def detect_people(frame, conf=0.3):
    results = _model(frame, verbose=False)[0]
    boxes = []
    for cls, xyxy, p in zip(results.boxes.cls, results.boxes.xyxy, results.boxes.conf):
        if int(cls) == 0 and p >= conf:  # class 0 = person
            x1, y1, x2, y2 = map(int, xyxy)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes
