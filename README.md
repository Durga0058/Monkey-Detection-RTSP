# Monkey-Detection

## Dataset
* Various video and image datasets are collected.
* The focus is specifically on tracking monkey movements.
* All collected data is systematically labeled.

## Splitting the Data
* The dataset is split into train and validation sets.
* The division follows standard machine learning optimization ratios.
* The split configuration is managed within the dataset directory.

## Cloning of YOLOv8 and installing requirements
* The project repository is cloned locally.
* Python virtual environment is set up.
* All the requirements for YOLOv8 are installed using:
  ```bash
  pip install -r requirements.txt
Data training and Validation
Mark the given dataset config files as required.

The training of the model is done using pre-trained weights.

Training is executed via the command line interface:

Bash
* To start the model training phase, run this command in your terminal:
  ```bash
  yolo task=detect mode=train model=yolov8n.pt data=data/custom_dataset.yaml epochs=50 imgsz=640 batch=16 device=0
Best validation weights are extracted upon successful run completion.

Inference
Multiple inference video streams are uploaded to the local input directory.

Automated batch inference with the custom weights is executed by running:

Bash

* To execute automated batch inference on your raw videos, run this command in your terminal:
  ```bash
  python detect_multiple.py
Display
The detected monkeys are tracked with automated bounding boxes.

The finalized output results are generated and displayed seamlessly.
