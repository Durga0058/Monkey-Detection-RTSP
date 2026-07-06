# 🐒 Monkey Detection Pipeline via RTSP stream and YOLOv8

This repository contains a production-ready computer vision pipeline designed to detect monkeys and draw automated bounding boxes over video streams using Ultralytics YOLOv8. It supports both custom local video testing and deployment over real-time RTSP camera feeds.

---

## 🛠️ Project Overview & Architecture
As documented in the initial project specifications, this pipeline tackles the problem of monitoring monkey activity to mitigate human-wildlife conflicts. 
* **Core Framework:** Python, OpenCV (Stream Handling), Ultralytics YOLOv8 (Object Detection).
* **Dataset Focus:** Customized formatting optimized for wildlife tracking, filtering classes specifically for primate detection.

---

## 📋 System Prerequisites

Before running or reproducing this project, ensure your environment meets the dependencies.
cd models
pip install -r requirements.txt
cd ..
Step 1: Environment Setup & Prerequisites

Before running the project, you need to install all the required Python libraries.

```bash
1. Data Preprocessing & Annotation
Before training, data is curated and annotations are validated to match the YOLO format (`[class_id, x_center, y_center, width, height]`).

```python
import os
import cv2

def verify_and_resize_dataset(image_dir, target_size=(640, 640)):
    """
    Ensures all input images are properly formatted and resized before training.
    """
    print(f"Preprocessing images in {image_dir}...")
    # Core preprocessing and augmentation logic goes here
2. Model Training (Ultralytics YOLOv8)
The model is trained on the custom primate dataset using Ultralytics YOLOv8. You can initiate training via the Python API or the Command Line Interface (CLI).

Option A: Python Script (train.py)

Python
from ultralytics import YOLO

# 1. Load a pre-trained YOLOv8 nano model
model = YOLO('yolov8n.pt') 

# 2. Train the model on custom data
results = model.train(
    data='data/custom_dataset.yaml',  # Path to dataset configuration file
    epochs=50,                        # Total training epochs
    imgsz=200,                        # Input image resolution
    batch=16,                         # Batch size
    device=0                          # GPU ID (use 'cpu' if no GPU is available)
)

# Run validation
metrics = model.val()
print(f"Validation mAP50-95: {metrics.box.map}")
print(f"Validation mAP50:    {metrics.box.map50}")
4. Real-time Inference & RTSP Deployment (monkey_detection.py)
This production script handles inference over local video files or live network camera streams via RTSP.

Python
import cv2
from ultralytics import YOLO

# Load the custom-trained weights
model = YOLO('runs/detect/train/weights/best.pt')

# Configure stream source: Local file path or network RTSP URL
# rtsp_url = "rtsp://username:password@ip_address:port/stream"
video_path = "Copy of video_new_7.mp4" 

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
        
    # Stream=True utilizes a generator for memory efficiency during live playback
    results = model(frame, stream=True)
    
    for r in results:
        # Render the bounding boxes onto the frame
        annotated_frame = r.plot()
        
    # Display the processed frame
    cv2.imshow("Monkey Detection - Production Pipeline", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
