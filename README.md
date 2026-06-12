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

