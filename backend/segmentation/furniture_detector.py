"""
Furniture Detection and Segmentation using YOLO-v8
Identifies furniture types in images and 3D scenes
"""

from ultralytics import YOLO
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import cv2


class FurnitureDetector:
    """
    Detect and segment furniture in images using YOLO-v8 or Roboflow
    """
    
    def __init__(self, model_path: str = None, use_roboflow: bool = False):
        """
        Initialize furniture detector
        
        Args:
            model_path: Path to custom YOLO model, or None for default
            use_roboflow: If True, use Roboflow furniture detection model (better for desk/nightstand)
        """
        self.use_roboflow = use_roboflow
        
        if use_roboflow:
            try:
                from backend.segmentation.roboflow_detector import RoboflowFurnitureDetector
                import os
                api_key = os.getenv("ROBOFLOW_API_KEY")
                if not api_key:
                    print("Warning: ROBOFLOW_API_KEY not set. Falling back to YOLO.")
                    print("Get your key from: https://roboflow.com/settings")
                    use_roboflow = False
                else:
                    self.roboflow_detector = RoboflowFurnitureDetector(api_key=api_key)
                    self.model = None  # Not used when using Roboflow
                    print("✓ Using Roboflow furniture detection model")
            except Exception as e:
                print(f"Warning: Could not initialize Roboflow detector: {e}")
                print("Falling back to YOLO...")
                use_roboflow = False
        
        if not use_roboflow:
            if model_path:
                self.model = YOLO(model_path)
            else:
                # Use pre-trained furniture detection model
                # In production, download from HuggingFace: MaherMohsen/furniture-yolov8
                self.model = YOLO('yolov8n.pt')  # Start with base model
            self.roboflow_detector = None
        
        # Furniture categories we care about
        # Note: YOLO base model uses COCO classes, so "desk" maps to "table"
        # and "nightstand" might be detected as "table" or not at all
        self.furniture_classes = {
            'bed': {'movable': True, 'priority': 'high'},
            'desk': {'movable': True, 'priority': 'high'},
            'chair': {'movable': True, 'priority': 'medium'},
            'sofa': {'movable': True, 'priority': 'high'},
            'couch': {'movable': True, 'priority': 'high'},  # Alternative name
            'table': {'movable': True, 'priority': 'medium'},  # Includes desk, nightstand
            'dining table': {'movable': True, 'priority': 'medium'},
            'dresser': {'movable': True, 'priority': 'medium'},
            'bookshelf': {'movable': True, 'priority': 'medium'},
            'nightstand': {'movable': True, 'priority': 'low'},
            'lamp': {'movable': True, 'priority': 'low'},
            'tv': {'movable': False, 'priority': 'medium'},
            'window': {'movable': False, 'priority': 'high'},
            'door': {'movable': False, 'priority': 'high'}
        }
    
    def detect_furniture(
        self, 
        image_path: str,
        confidence_threshold: float = 0.5
    ) -> List[Dict]:
        """
        Detect furniture in a single image
        
        Args:
            image_path: Path to image
            confidence_threshold: Minimum confidence for detection
        
        Returns:
            List of detected furniture with bounding boxes and metadata
        """
        # Use Roboflow if available
        if self.use_roboflow and self.roboflow_detector:
            return self.roboflow_detector.detect_furniture(image_path, confidence_threshold)
        
        # Otherwise use YOLO
        results = self.model(image_path, conf=confidence_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                
                # Check if this is furniture we care about
                if class_name.lower() in self.furniture_classes:
                    detection = {
                        'class': class_name.lower(),
                        'confidence': confidence,
                        'bbox': bbox.tolist(),
                        'center': self._get_bbox_center(bbox),
                        'area': self._get_bbox_area(bbox),
                        'metadata': self.furniture_classes[class_name.lower()]
                    }
                    detections.append(detection)
        
        return detections
    
    def detect_furniture_batch(
        self,
        image_dir: str,
        confidence_threshold: float = 0.5
    ) -> Dict[str, List[Dict]]:
        """
        Detect furniture in multiple images
        
        Args:
            image_dir: Directory containing images
            confidence_threshold: Minimum confidence
        
        Returns:
            Dict mapping image names to detections
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob("*.jpg")) + \
                     list(image_dir.glob("*.png")) + \
                     list(image_dir.glob("*.jpeg"))
        
        all_detections = {}
        for img_path in image_files:
            detections = self.detect_furniture(str(img_path), confidence_threshold)
            all_detections[img_path.name] = detections
        
        return all_detections
    
    def aggregate_detections(
        self,
        detections_by_image: Dict[str, List[Dict]],
        min_occurrences: int = 3
    ) -> List[Dict]:
        """
        Aggregate detections across multiple views to get room inventory
        
        Args:
            detections_by_image: Output from detect_furniture_batch
            min_occurrences: Minimum times an object must appear to be counted
        
        Returns:
            List of unique furniture items in the room
        """
        # Count occurrences of each furniture type
        furniture_counts = {}
        for image_name, detections in detections_by_image.items():
            for det in detections:
                class_name = det['class']
                if class_name not in furniture_counts:
                    furniture_counts[class_name] = {
                        'count': 0,
                        'avg_confidence': 0,
                        'metadata': det['metadata']
                    }
                furniture_counts[class_name]['count'] += 1
                furniture_counts[class_name]['avg_confidence'] += det['confidence']
        
        # Filter and normalize
        room_inventory = []
        for class_name, data in furniture_counts.items():
            if data['count'] >= min_occurrences:
                room_inventory.append({
                    'type': class_name,
                    'quantity': self._estimate_quantity(data['count'], len(detections_by_image)),
                    'confidence': data['avg_confidence'] / data['count'],
                    'movable': data['metadata']['movable'],
                    'priority': data['metadata']['priority']
                })
        
        return room_inventory
    
    def visualize_detections(
        self,
        image_path: str,
        output_path: str,
        confidence_threshold: float = 0.5
    ):
        """
        Draw bounding boxes on image and save
        
        Args:
            image_path: Input image
            output_path: Where to save annotated image
            confidence_threshold: Minimum confidence
        """
        detections = self.detect_furniture(image_path, confidence_threshold)
        
        # Load image
        img = cv2.imread(image_path)
        
        # Draw boxes
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add label
            label = f"{det['class']} {det['confidence']:.2f}"
            cv2.putText(img, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save
        cv2.imwrite(output_path, img)
        print(f"Saved annotated image to {output_path}")
    
    def _get_bbox_center(self, bbox: np.ndarray) -> Tuple[float, float]:
        """Calculate center of bounding box"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _get_bbox_area(self, bbox: np.ndarray) -> float:
        """Calculate area of bounding box"""
        x1, y1, x2, y2 = bbox
        return (x2 - x1) * (y2 - y1)
    
    def _estimate_quantity(self, detections: int, num_images: int) -> int:
        """
        Estimate actual quantity from number of detections across images
        
        Simple heuristic: if detected in >80% of images, probably 1 item
        If detected less frequently, might be multiple items or occlusion
        """
        detection_rate = detections / num_images
        
        if detection_rate > 0.8:
            return 1
        elif detection_rate > 0.4:
            return max(1, int(detections / (num_images * 0.6)))
        else:
            return max(1, int(detections / num_images))


# Example usage
if __name__ == "__main__":
    detector = FurnitureDetector()
    
    # Detect in single image
    print("Detecting furniture in image...")
    detections = detector.detect_furniture("data/test_image.jpg")
    
    for det in detections:
        print(f"Found {det['class']} (confidence: {det['confidence']:.2f})")
    
    # Visualize
    detector.visualize_detections(
        "data/test_image.jpg",
        "data/annotated_image.jpg"
    )


