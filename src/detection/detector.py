import torch
import cv2
from ultralytics import YOLO
import os


class PersonDetector:
    """Simple YOLOv8 wrapper for person detection."""
    
    def __init__(self, model_name='yolov8n.pt', conf_threshold=0.5, img_size=640, min_box_area=800):
        """Initialize detector with YOLOv8."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Look for model in models/ directory
        model_path = os.path.join('models', model_name)
        if not os.path.exists(model_path):
            # If not found, let YOLO download it
            model_path = model_name
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.min_box_area = min_box_area
    
    def detect(self, frame):
        """Detect persons in frame with filtering."""
        results = self.model(
            frame, 
            classes=[0], 
            conf=self.conf_threshold, 
            verbose=False,
            imgsz=self.img_size,
            device=self.device
        )
        
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                
                # Filter by area
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_box_area:
                    continue
                
                # Filter by aspect ratio (people are vertical)
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = height / width if width > 0 else 0
                if aspect_ratio < 0.5 or aspect_ratio > 5.0:
                    continue
                
                detections.append([x1, y1, x2, y2, float(conf)])
        
        return detections