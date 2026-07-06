import os
import cv2
from ultralytics import YOLO

# ==========================================
# 1. Model Configuration
# ==========================================
# If you have custom trained weights, replace 'yolov8n.pt' with your path (e.g., 'models/best.pt')
model_path = 'yolov8n.pt'
model = YOLO(model_path)

# ==========================================
# 2. Folder Paths Configuration
# ==========================================
input_folder = "input_videos" 
output_folder = "output_results"

# Create output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Supported video extensions
video_extensions = ('.mp4', '.avi', '.mkv', '.mov')

if not os.path.exists(input_folder):
    print(f"❌ Error: '{input_folder}' folder not found. Please create it and add your videos.")
    exit()

# Filter and collect all valid videos from the input directory
videos = [f for f in os.listdir(input_folder) if f.lower().endswith(video_extensions)]

print(f"🔍 Found {len(videos)} videos to process in '{input_folder}' folder.\n")

# ==========================================
# 3. Processing Pipeline Loop
# ==========================================
for idx, video_name in enumerate(videos, 1):
    input_path = os.path.join(input_folder, video_name)
    output_path = os.path.join(output_folder, f"detected_{video_name}")
    
    print(f"==================================================")
    print(f"📹 [{idx}/{len(videos)}] Processing: {video_name}")
    print(f"💾 Saving output to: {output_path}")
    print(f"==================================================")
    
    cap = cv2.VideoCapture(input_path)
    
    # Read video metadata parameters for correct rendering
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Codec for MP4 format
    
    # Initialize Video Writer
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run YOLOv8 inference (Confidence threshold set to 25%)
        results = model(frame, conf=0.25, verbose=False)
        
        # Render the bounding boxes onto the frame
        annotated_frame = results[0].plot()
        
        # Write the processed frame into the output video file
        out.write(annotated_frame)
        
        # Display live visual tracking stream pane
        cv2.imshow("Batch Processing - Primate Tracking System", annotated_frame)
        
        # Press 'q' key to skip/abort current video processing smoothly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("⚠️ User interrupted current video processing.")
            break
            
    # Release resources to keep clean memory execution
    cap.release()
    out.release()
    print(f"✅ Finished: {video_name} parsed successfully ({frame_count} frames processed).\n")

# Process completion sequence
cv2.destroyAllWindows()
print("🎉 Success! All targeted inputs fully evaluated. Check the 'output_results' directory.")
