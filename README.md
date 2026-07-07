## 🛠 Prerequisites & Environment Setup

To replicate this project exactly and avoid dependency conflicts, ensure your system uses the following core versions:

* **Python Version:** `3.10.x` 
* **Pip Version:** `23.x.x` 

### Installation Setup

```bash

# Install required dependencies
pip install -r requirements.txt
Data Pipeline & Collection
1. Data Collection Instructions
Source: The dataset consists of monkey images and video frames collected from custom field recordings or public data sources.

Format: All data must be annotated in standard YOLO format (where each .jpg image has a corresponding .txt file containing class indices and normalized bounding box coordinates).

2. Preprocessing Instructions
Before initiating training, prepare your raw dataset using the following workflow:

Resizing: Standardize all dataset images to a uniform resolution of 640x640 pixels.

Dataset Splitting: Divide the preprocessed images and labels into training and validation sets (Recommended distribution: 80% Train, 20% Val).

Directory Hierarchy: Ensure files are structured correctly into images/ and labels/ subdirectories to prevent training script crashes.

Dataset Configuration
The training script requires a configuration file to reference the dataset paths and class variables.

File Location: data/custom_dataset.yaml

Structure Template: Ensure your YAML file follows this structural format:

YAML
path: ../data        # Dataset root directory path
train: images/train  # Training images subfolder (relative to path)
val: images/val      # Validation images subfolder (relative to path)

nc: 1                # Number of classes
names: ['monkey']    # Array of class names
Training & Model Checkpoints
Running the Training Script
To trigger the custom training workflow using the defined environment and dataset setup, run the following terminal command:

Bash
python train.py --data data/custom_dataset.yaml --weights yolov8n.pt --epochs 100 --img 640
How to Get the Best Model Checkpoint
Once the training process finishes execution, the framework automatically evaluates performance metrics and exports the model weights.

Directory Path: Checkpoints are stored automatically under: runs/detect/train/weights/

best.pt (Recommended): Use this specific checkpoint for production deployment, inference testing, or RTSP streaming. It represents the weights that achieved the highest precision/lowest loss on the validation split.

last.pt: This file represents the weights captured at the very last training epoch.

