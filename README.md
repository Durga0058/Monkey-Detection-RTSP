Monkey Detection RTSP

This project is a comprehensive guide to building a custom object detection
system. It uses YOLOv8 to identify monkeys in various environments, specifically
designed to work with real-time RTSP camera streams.

 1. Environment Requirements

To ensure the code runs without errors, you must use these specific versions:

  - Python: 3.10.x
  - Pip: 23.x.x
  - Operating System: Windows, Linux, or macOS.

 2. Installation

First, bring the project to your local machine and set up the environment:

1.  Clone the Repo:
    git clone https://github.com/Durga0058/Monkey-Detection-RTSP.git
    cd Monkey-Detection-RTSP
2.  Install Dependencies:
    pip install -r requirements.txt

 3. Data Collection & Preprocessing

Before the model can "see" monkeys, we must prepare the visual data. Follow
these steps:

A. Data Collection

  - Gather Images: Collect at least 200–500 images of monkeys. Ensure variety:
    different species, lighting (day/night), and distances from the camera.
  - Annotation: Use a tool like CVAT or Roboflow to draw boxes around the
    monkeys.
  - Format: Export your annotations in YOLO format (this creates one .txt file
    for every .jpg image).

B. Preprocessing

To make training faster and more accurate:

1.  Resizing: All images should be resized to 640x640 pixels.
2.  Splitting: Organize your data into two main groups:
      - Train (80%): Used to teach the model.
      - Val (20%): Used to test the model's accuracy during training.
3.  Folder Mapping: Ensure your files are placed in the data/ directory as shown
    in the Directory Structure section below.

 4. The Dataset Configuration (custom_dataset.yaml)

If you cannot find the data/custom_dataset.yaml file, you need to create it
manually inside the data/ folder. This file acts as a map for the computer to
find your images.

Create a file named custom_dataset.yaml and paste this inside:

path: ./data          # The root directory of your data
train: images/train   # Location of training images
val: images/val       # Location of validation images

nc: 1                 # Number of classes (just 1 for monkey)
names: ['monkey']     # The name of the class

5. Training the Model

Once your data and .yaml file are ready, start the training process with this
command:

python train.py --data data/custom_dataset.yaml --weights yolov8n.pt --epochs 100 --img 640

  - --weights yolov8n.pt: Starts with a pre-trained "Nano" model (fastest).
  - --epochs 100: The model will look at the dataset 100 times to learn.

 6. Finding your Best Model Checkpoint

After training finishes, the system automatically creates a folder named runs/.
This is where your results live.

Where to find the weights? Navigate to: runs/detect/train/weights/

You will see two files:

1.  best.pt: USE THIS ONE. It is the version of the model that had the highest
    accuracy during the training process.
2.  last.pt: This is simply the model at the very last epoch (100). It might not
    be as accurate as the "best" version.

💻 7. Running Inference

To test your "best" model on a local video file, run:

python monkey_detection.py "input_videos/your_video.mp4"

 Directory Structure

Your folders should look exactly like this for the scripts to work:

.
├── data/
│   ├── custom_dataset.yaml   <-- You create this
│   ├── images/
│   │   ├── train/            <-- Training .jpg files
│   │   └── val/              <-- Validation .jpg files
│   └── labels/
│       ├── train/            <-- Training .txt files
│       └── val/              <-- Validation .txt files
├── input_videos/
├── runs/
│   └── detect/
│       └── train/
│           └── weights/
│               └── best.pt   <-- Your final trained model
├── train.py
├── monkey_detection.py
└── requirements.txt

Note: This pipeline is optimized for RTSP streaming. Ensure your camera URL is
correctly configured in monkey_detection.py for live monitoring.

