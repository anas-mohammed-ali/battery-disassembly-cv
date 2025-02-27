# src/detect.py
from ultralytics import YOLO

def run_inference(model_path='runs/detect/exp/weights/best.pt', source='data/images/test', conf=0.5):
    # Load the trained model
    model = YOLO(model_path)
    
    # Run inference on the test images folder
    results = model.predict(source=source, conf=conf, save=True)
    # Display results (in a notebook or script, you might want to process/display these images)
    results.show()
    
if __name__ == '__main__':
    run_inference()
