import cv2
import sys
import os
from ultralytics import YOLO

def run_detection():
    # 1. Validate command-line arguments
    if len(sys.argv) < 2:
        print("\n❌ Error: Missing input video file path.")
        print("💡 Usage: python monkey_detection.py <path_to_video.mp4>\n")
        return

    input_video_path = sys.argv[1]

    # Check if the specified file exists
    if not os.path.exists(input_video_path):
        print(f"\n❌ Error: File '{input_video_path}' not found. Please verify the path.\n")
        return

    print(f"\n🔄 Initializing detection pipeline for: {input_video_path}")

    # 2. Load pre-trained YOLOv8 model and capture video stream
    model = YOLO("models/yolov8n.pt")
    cap = cv2.VideoCapture(input_video_path)
    
    # Retrieve video structural properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Dynamically generate output file name (e.g., input_video_output.mp4)
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f"{base_name}_output.mp4"
    
    # 3. Initialize VideoWriter to save processed stream with bounding boxes
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    print("🚀 Processing video frames. Press 'q' to abort execution.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Perform object inference using YOLOv8
        results = model(frame)
        
        # Render tracking annotations and bounding boxes onto the frame
        annotated_frame = results[0].plot()
        
        # Write annotated frame to output file and display stream
        out.write(annotated_frame)
        cv2.imshow("Object Detection Pipeline - Processing Stream", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ Process interrupted by user.")
            break
            
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎯 Execution Completed. Output saved successfully at: {output_video_path}\n")

if __name__ == "__main__":
    run_detection()
    