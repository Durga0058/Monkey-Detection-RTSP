import cv2
import yaml
from ultralytics import YOLO

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_monkey_detection():
    # 1. Load Configuration
    config = load_config()
    
    # 2. Select Video Source
    if config['video_sources']['active_mode'] == "RTSP":
        # Testing ke liye agar RTSP na ho toh 0 (Webcam) use karein
        source = config['video_sources']['rtsp_url']
        if source == "0" or source == 0:
            source = 0
    else:
        source = config['video_sources']['offline_video_path']
        
    # 3. Load YOLO Model (Sir ke backend ke hisab se)
    model_path = config['models']['model_path']
    model = YOLO(model_path) 
    
    # 4. Start Video Capture
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"Error: Video source '{source}' open nahi ho paa raha hai.")
        return

    print("Monkey Detection Service Start Ho Chuki Hai... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video stream khatam ho gayi ya frame nahi mil raha.")
            break
            
        # 5. Run Object Detection on Current Frame
        # COCO dataset mein 'monkey' directly nahi hota, par YOLOv8n animals ko detect karta hai.
        # Agar aapke paas custom 'monkey.pt' model hai toh wo specific detect karega.
        results = model(frame, verbose=False)
        
        # 6. Bounding Box Draw Karna
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Coordinate nikalna
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = round(float(box.conf[0]), 2)
                cls = int(box.cls[0])
                label = model.names[cls]
                
                # Sirf monkey/animals target karne ke liye Filter (YOLO COCO list ke hisab se)
                # Agar custom monkey model hai toh is filter ki zaroorat nahi padegi.
                if label in ["monkey", "animal", "person"]: # Aap apne specific requirement ke mutabik change kar sakte hain
                    
                    # Green Color ka Bounding Box banana
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Label Text add karna
                    text = f"{label} {conf}"
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 7. Live Window Display
        cv2.imshow("Live Monkey Detection - RTSP Service", frame)
        
        # 'q' daba kar exit karne ke liye
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_monkey_detection()
    