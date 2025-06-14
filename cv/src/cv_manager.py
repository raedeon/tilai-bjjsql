"""Manages the CV model."""
# docker run -p 5002:5002 --gpus all bjjsql-cv:v4.10

from typing import Any
from ultralytics import YOLO
import numpy as np
import cv2
import tempfile
import torch


class CVManager:

    def __init__(self):
        # This is where you can initialize your model and any static
        # configurations.
        self.model = YOLO("models/best.pt")
        
        
        # Use GPU with half precision if available. Speeds up inferences
        if torch.cuda.is_available():
            self.model.to("cuda").half()
        else:
            self.model.to("cpu")
            
        # Fuse Conv+BN layers for faster inference
        self.model.fuse()
        
        self.imgsz = 1280  # Or reduce to 640 for speed
        #self.conf_threshold = 0.25 # supposed to increase accuracy but it js reduced speed
        #self.iou = 0.5
        #self.use_tta = False  # Enable test-time augmentation

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Performs object detection on an image.

        Args:
            image: The image file in bytes.

        Returns:
            A list of `dict`s containing your CV model's predictions. See
            `cv/README.md` for the expected format.
        """

        # Convert bytes to OpenCV image (numpy array)
        nparr = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Run inference
        results = self.model.predict(
            source=img, 
            imgsz=self.imgsz,
            #augment=self.use_tta,
            #conf=self.conf_threshold,
            #iou = self.iou
        )[0]  # YOLOv8 returns a list of Results

        # Format output
        predictions = []
        for box in results.boxes:
            cls = int(box.cls.item())  # category_id
            conf = float(box.conf.item())  # confidence score
            xywh = box.xywh[0].tolist()  # [x_center, y_center, width, height]
            x_center, y_center, w, h = xywh

            x_min = x_center - w / 2
            y_min = y_center - h / 2

            predictions.append({
                "category_id": cls,
                "bbox": [x_min, y_min, w, h],
                "score": conf
            })

        return predictions
