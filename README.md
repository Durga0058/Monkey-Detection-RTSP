# Monkey Detection RTSP

Real-time monkey detection system for live camera streams and offline video files. It wires a YOLOv8 object detection model into a video pipeline, supporting both RTSP camera feeds and local video files as input. The active entry point is `monkey_detection.py`, using `config.yaml` for source and model selection.

Two config sources:

- **`config.yaml`** — runtime behavior (video source mode, RTSP URL / local video path, model weights path). See [Configuration](#configuration) below.
- **`data/custom_dataset.yaml`** — dataset mapping used only during training (image/label paths, class names).

## Requirements

- Python: `3.10.x`
- Pip: `23.x.x`

```bash
conda create -n monkeydet python=3.10
conda activate monkeydet        # or activate your virtualenv
```

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/Durga0058/Monkey-Detection-RTSP.git
cd Monkey-Detection-RTSP
pip install -r requirements.txt
python3 -m venv .venv
```

## Configuration

Create `config.yaml` in the project root:

```yaml
video_sources:
  active_mode: "FILE"              # "RTSP" or "FILE"
  rtsp_url: ""                     # camera stream URL, used when active_mode is "RTSP"
  offline_video_path: "input_videos/your_video.mp4"   # used when active_mode is "FILE"

models:
  model_path: "runs/detect/train/weights/best.pt"     # trained model weights
```

- Set `active_mode: "RTSP"` and fill `rtsp_url` for live camera monitoring.
- Set `active_mode: "FILE"` and fill `offline_video_path` to test on a local video.

## Data Collection & Preprocessing

Before training, prepare the visual dataset:

**A. Data Collection**
- Gather at least 200–500 images of monkeys, with variety in species, lighting (day/night), and distance from camera.
- Annotate using [Roboflow](https://roboflow.com/), drawing boxes around each monkey.
- Export annotations in YOLO format (one `.txt` label file per `.jpg` image).

**B. Preprocessing**
1. Resize all images to `640x640` pixels.
2. Split data into `train` (80%) and `val` (20%) sets.
3. Place files in the `data/` directory as shown in Directory Structure below.

## Dataset Configuration (`data/custom_dataset.yaml`)

If `data/custom_dataset.yaml` doesn't exist, create it manually:

```yaml
path: ./data
train: images/train
val: images/val
nc: 1
names: ['monkey']
```

## Training

Once data and the `.yaml` config are ready:

```bash
python train.py --data data/custom_dataset.yaml --weights yolov8n.pt --epochs 100 --img 640
```

- `--weights yolov8n.pt` — starts from a pretrained "Nano" model (fastest to train).
- `--epochs 100` — number of passes over the dataset.

Trained weights are saved to `runs/detect/train/weights/`. Use **`best.pt`** — the checkpoint with highest validation accuracy.

## Running Inference

Run detection on a local video or RTSP stream (as set in `config.yaml`):

```bash
python monkey_detection.py
```

