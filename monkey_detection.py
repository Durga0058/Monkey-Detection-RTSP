import cv2
import sys
import os
from ultralytics import YOLO

def run_detection():
    # 1. Check karo ki sir ne video ka path diya hai ya nahi
    if len(sys.argv) < 2:
        print("\n❌ Error: Please provide the video file name/path!")
        print("💡 Usage Example: python monkey_detection.py path/to/your_video.mp4\n")
        return

    input_video_path = sys.argv[1]

    # Check karo ki file sach mein exist karti hai ya nahi
    if not os.path.exists(input_video_path):
        print(f"\n❌ Error: File '{input_video_path}' not found!\n")
        return

    print(f"\n🔄 Processing started for: {input_video_path}")

    # 2. Model aur Video load karo
    model = YOLO("models/yolov8n.pt")
    cap = cv2.VideoCapture(input_video_path)
    
    # Video ki properties nikalo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Output file ka naam automatically set karo (e.g., "video_output.mp4")
    base_name = os.path.splitext(os.path.basename(input_video_path))[0]
    output_video_path = f"{base_name}_output.mp4"
    
    # 3. VideoWriter setup (Sir ke desktop/folder mein video save karne ke liye)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # YOLO Prediction (Monkey detection)
        results = model(frame)
        
        # Bounding boxes draw karo
        annotated_frame = results[0].plot()
        
        # Frame save karo aur display karo
        out.write(annotated_frame)
        cv2.imshow("Monkey Detection - Processing...", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n🎯 Done! Bounding box video saved as: {output_video_path}\n")

if __name__ == "__main__":
    run_detection()
    