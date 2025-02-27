# src/train.py
from ultralytics import YOLO

def train_model(data_yaml='data/data.yaml', epochs=50, imgsz=640):
    # Load a pretrained YOLOv8 model (e.g., 'yolov8n.pt' for the nano version; you can choose others)
    model = YOLO('yolov8n.pt')
    
    # Train the model on your custom dataset
    results = model.train(data=data_yaml, epochs=epochs, imgsz=imgsz)
    print("Training complete. Best weights saved in runs/detect/exp*/weights/best.pt")
    
if __name__ == '__main__':
    train_model()
