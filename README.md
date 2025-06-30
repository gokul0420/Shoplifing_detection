# 🛒 Shoplifting Detection Using Deep Learning and Tracking

This project detects suspicious shoplifting behavior from surveillance videos using:
- **YOLO** for person detection
- **Kalman filter-based SORT** for tracking
- **ResNet18** for feature extraction
- **LSTM** for behavior classification

---

## 📁 Project Structure

```
shoplifting_detection/
├── data/
│   └── raw_videos/
│       ├── normal/
│       └── shoplift/
├── features/
│   ├── feature_extractor.py
│   └── generate_features.py
├── model/
│   ├── sequence_classifier.py
│   └── train_model.py
├── detectors/
│   └── yolo_detector.py
├── tracking/
│   ├── sort.py
│   └── sort_tracker.py
├── inference/
│   └── detect_shoplifting.py
└── outputs/
    ├── annotated_videos/
    └── scores.json
```

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/gokul0420/Shoplifing_detection.git
cd Shoplifing_detection
```

### 2. Create a Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Linux/Mac
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install manually:

```bash
pip install torch torchvision numpy opencv-python scikit-image filterpy matplotlib
```

---

## 📦 Step-by-Step Execution

### ✅ Step 1: Prepare Your Dataset

Place your surveillance videos in the `data/raw_videos/` directory:

```
data/raw_videos/
├── normal/        # Non-shoplifting videos
└── shoplift/      # Shoplifting behavior videos
```

### ✅ Step 2: Generate Features

```bash
python features/generate_features.py
```

This will extract features using ResNet and save them to `data/features/X.npy` and `y.npy`.

### ✅ Step 3: Train the LSTM Classifier

```bash
python model/train_model.py
```

This will create a trained LSTM and save it as `model/shoplifting_lstm.pth`.

### 🎥 Step 4: Run Detection on a New Video

Edit the path in `inference/detect_shoplifting.py`:

```python
VIDEO_PATH = "data/raw_videos/shoplifting/shoplifting-7.mp4"
```

Then run:

```bash
python -m inference.detect_shoplifting
```

### 📤 Output:
* Annotated video saved to `outputs/annotated_videos/result.avi`
* Shoplifting decision printed to terminal:

```
✅ Prediction: 🛑 Shoplifting detected!
```

---

## 🧠 Model Architecture

* **Person Detection**: YOLOv3-tiny (custom)
* **Tracking**: SORT (Simple Online Realtime Tracking)
* **Feature Extraction**: ResNet18 (without FC layer)
* **Classification**: 2-layer LSTM with sigmoid output

---

## 🤝 Contributing

Feel free to fork this repo and improve:
* Model accuracy
* Real-time inference
* Web-based dashboard integration

---

## 🛑 License

This project is open-sourced under the MIT License.

---

## ✨ Credits

Developed by **Gokul Krishna R**  
Department of AI & DS, Chennai Institute of Technology
