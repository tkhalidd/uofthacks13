"""
Roboflow Furniture Detection Integration
Uses Roboflow's furniture detection model with 56 classes including desk, nightstand, etc.
"""

from typing import List, Dict, Optional
import numpy as np
from pathlib import Path

try:
    from roboflow import Roboflow
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    print("Warning: roboflow not installed. Install with: pip install roboflow")


class RoboflowFurnitureDetector:
    """
    Detect furniture using Roboflow's furniture detection model
    Model: furniture-detection-qiufc/20
    Classes: 56 furniture categories including bed, desk, nightstand, etc.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "furniture-detection-qiufc",
        version: int = 20
    ):
        """
        Initialize Roboflow furniture detector
        
        Args:
            api_key: Roboflow API key (get from https://roboflow.com/settings)
            model_id: Roboflow model ID
            version: Model version number
        """
        if not ROBOFLOW_AVAILABLE:
            raise ImportError(
                "roboflow package not installed. "
                "Install with: pip install roboflow"
            )
        
        # Get API key from environment if not provided
        if api_key is None:
            import os
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if not api_key:
                raise ValueError(
                    "Roboflow API key required. "
                    "Set ROBOFLOW_API_KEY environment variable or pass api_key parameter. "
                    "Get your key from: https://roboflow.com/settings"
                )
        
        print(f"Initializing Roboflow model: {model_id}/{version}...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace().project(model_id)
        self.model = project.version(version).model
        print(f"✓ Roboflow model loaded")
        
        # Furniture categories we care about (from the 56 classes)
        self.furniture_classes = {
            'bed': {'movable': True, 'priority': 'high'},
            'master bed': {'movable': True, 'priority': 'high'},
            'desk': {'movable': True, 'priority': 'high'},
            'dining table': {'movable': True, 'priority': 'medium'},
            'dining-table': {'movable': True, 'priority': 'medium'},
            'dinning-table': {'movable': True, 'priority': 'medium'},
            'table': {'movable': True, 'priority': 'medium'},
            'nightstand': {'movable': True, 'priority': 'low'},
            'chair': {'movable': True, 'priority': 'medium'},
            'arm chair': {'movable': True, 'priority': 'medium'},
            'sofa': {'movable': True, 'priority': 'high'},
            'couch': {'movable': True, 'priority': 'high'},
            'dresser': {'movable': True, 'priority': 'medium'},
            'cabinet': {'movable': True, 'priority': 'medium'},
            'closet': {'movable': True, 'priority': 'medium'},
            'transparent closet': {'movable': True, 'priority': 'medium'},
            'wardrobe': {'movable': True, 'priority': 'medium'},
            'cupboard': {'movable': True, 'priority': 'medium'},
            'sideboard': {'movable': True, 'priority': 'medium'},
            'drawer near bed': {'movable': True, 'priority': 'low'},
            'bookshelf': {'movable': True, 'priority': 'medium'},
            'shelf': {'movable': True, 'priority': 'medium'},
            'lamp': {'movable': True, 'priority': 'low'},
            'hanging lights': {'movable': True, 'priority': 'low'},
            'tv stand': {'movable': False, 'priority': 'medium'},
            'tv': {'movable': False, 'priority': 'medium'},
            'monitor': {'movable': True, 'priority': 'medium'},
            'computer': {'movable': True, 'priority': 'medium'},
            'window': {'movable': False, 'priority': 'high'},
            'windows': {'movable': False, 'priority': 'high'},
            'door': {'movable': False, 'priority': 'high'},
            'curtains': {'movable': False, 'priority': 'low'},
            'carpet': {'movable': False, 'priority': 'low'},
            'rug': {'movable': False, 'priority': 'low'},
        }
    
    def detect_furniture(
        self,
        image_path: str,
        confidence_threshold: float = 0.3
    ) -> List[Dict]:
        """
        Detect furniture in a single image
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of detected furniture with bounding boxes and metadata
        """
        # Run inference
        predictions = self.model.predict(
            image_path,
            confidence=confidence_threshold
        ).json()
        
        detections = []
        for prediction in predictions.get('predictions', []):
            class_name = prediction['class'].lower()
            confidence = prediction['confidence']
            bbox = [
                prediction['x'] - prediction['width'] / 2,  # x_min
                prediction['y'] - prediction['height'] / 2,  # y_min
                prediction['x'] + prediction['width'] / 2,  # x_max
                prediction['y'] + prediction['height'] / 2   # y_max
            ]
            
            # Check if this is furniture we care about
            if class_name in self.furniture_classes:
                detection = {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox,
                    'center': [prediction['x'], prediction['y']],
                    'area': prediction['width'] * prediction['height'],
                    'metadata': self.furniture_classes[class_name]
                }
                detections.append(detection)
        
        return detections
    
    def detect_furniture_batch(
        self,
        image_dir: str,
        confidence_threshold: float = 0.3
    ) -> Dict[str, List[Dict]]:
        """
        Detect furniture in multiple images
        
        Args:
            image_dir: Directory containing images
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            Dictionary mapping image paths to detection lists
        """
        image_path = Path(image_dir)
        if not image_path.is_dir():
            raise ValueError(f"{image_dir} is not a directory")
        
        results = {}
        for img_file in image_path.glob("*.jpg") + image_path.glob("*.png"):
            detections = self.detect_furniture(str(img_file), confidence_threshold)
            results[str(img_file)] = detections
        
        return results


if __name__ == "__main__":
    # Example usage
    import os
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("Error: Set ROBOFLOW_API_KEY environment variable")
        print("Get your key from: https://roboflow.com/settings")
    else:
        detector = RoboflowFurnitureDetector(api_key=api_key)
        detections = detector.detect_furniture("furniture_photos/fullbed.png")
        
        print(f"\nDetected {len(detections)} furniture pieces:")
        for det in detections:
            print(f"  - {det['class']} (confidence: {det['confidence']:.2f})")

