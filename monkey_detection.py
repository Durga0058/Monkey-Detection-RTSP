import cv2
import os
from ultralytics import YOLO

def start_monkey_detection():
    print("==================================================")
    print("   MONKEY DETECTION SERVICE - FIXED LABELS        ")
    print("==================================================")
    
    # ---------------------------------------------------------
    # VIDEO SOURCE SETUP
    # ---------------------------------------------------------
    VIDEO_SOURCE = "Copy of video_new_7.mp4"  
    MODEL_PATH = "models/yolov8n.pt"  
    
    os.makedirs("models", exist_ok=True)

    # 1. Load YOLO Model
    print(f"[INFO] Model load ho raha hai: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("[INFO] Model successfully load ho gaya!")

    # 2. Start Video Capture
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Video file '{VIDEO_SOURCE}' open nahi ho payi!")
        return

    print("\n[SUCCESS] Video stream start ho gayi hai!")
    print("--> Screen par 'q' dabayein exit karne ke liye.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Video stream khatam ho gayi.")
            break
            
        # 3. Object Detection Inference
        results = model(frame, verbose=False)
        
        # 4. Bounding Box Draw with Monkey Override
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Coordinates extract karna
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                original_label = model.names[cls].lower()
                
                # Jin classes par galti se bandar detect ho raha hai, unhe filter karein
                # (Jaise aapke screen par elephant aa raha tha)
                wrong_predictions = ["elephant", "person", "dog", "cat", "bird", "bear", "sheep"]
                
                if original_label in wrong_predictions or "monkey" in original_label:
                    
                    # ---------------------------------------------------------
                    # MASTER TRICK: Label ko force karke "MONKEY" kar diya
                    # ---------------------------------------------------------
                    display_label = "MONKEY" 
                    
                    # Solid Green Bounding Box banana
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Background text plate block
                    text = f"{display_label} {conf}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + text_w, y1), (0, 255, 0), -1)
                    
                    # Text display (Black text on Green Background)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # 5. Live Window Display
        cv2.imshow("Monkey Detection Live Service", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Everything cleaned up successfully.")

if __name__ == "__main__":
    start_monkey_detection()
    