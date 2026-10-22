import os
from ultralytics import YOLO

def run_detection():
    print("Initializing YOLO AI...")
    
    # Load the official lightweight YOLO model (downloads automatically if not found)
    model = YOLO("yolo11n.pt")
    
    # Use a sample image or fallback to webcam (0)
    # Ultralytics provides a default 'bus.jpg' for quick testing out-of-the-box
    source = "https://ultralytics.com/images/bus.jpg"
    
    print(f"Running inference on source: {source}")
    
    # Run prediction (show=False for headless/CI environments, True to pop up a window locally)
    results = model(source, show=False)
    
    # Print out detected objects
    for r in results:
        boxes = r.boxes
        print(f"Detected {len(boxes)} objects!")
        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])
            print(f" - {class_name} with confidence {conf:.2f}")

    print("Detection complete! Results saved in runs/detect/")

if __name__ == "__main__":
    run_detection()
