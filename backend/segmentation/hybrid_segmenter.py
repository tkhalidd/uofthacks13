"""
Hybrid segmentation: Best of both worlds
- Beds: Use improved method (expanded bbox, context) - this worked best
- Small objects: Use simple method (minimal processing) - this worked best
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2

try:
    from segment_anything import sam_model_registry, SamPredictor
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class HybridSegmenter:
    """
    Hybrid approach:
    - Beds: Improved segmentation (expanded bbox, context) - BEST for beds
    - Small objects: Simple cropping (minimal processing) - BEST for nightstand/chair
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: str = "cpu"
    ):
        if not SAM_AVAILABLE:
            raise ImportError("segment-anything not installed")
        
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path(model_type)
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {checkpoint_path}")
        
        # Only load SAM for beds (small objects use simple cropping)
        print(f"Loading SAM model ({model_type}) for bed segmentation...")
        sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        sam.to(device=self.device)
        self.predictor = SamPredictor(sam)
        print(f"✓ SAM loaded on {self.device}")
    
    def _get_default_checkpoint_path(self, model_type: str) -> str:
        home = Path.home()
        checkpoints_dir = home / ".sam_checkpoints"
        checkpoint_map = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        return str(checkpoints_dir / checkpoint_map.get(model_type, "sam_vit_b_01ec64.pth"))
    
    def process_bed_improved(
        self,
        image_path: str,
        bbox: List[float],
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Process bed using IMPROVED method (this worked best for beds)
        - 30% expansion
        - 80px padding
        - Keep background for depth
        - Use SAM with multimask
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Expand bbox by 15% (reduced to avoid including too much)
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        expand_x = max(40, int(width * 0.15))
        expand_y = max(40, int(height * 0.15))
        
        expanded_bbox = [
            max(0, int(x1) - expand_x),
            max(0, int(y1) - expand_y),
            min(w, int(x2) + expand_x),
            min(h, int(y2) + expand_y)
        ]
        
        # Use SAM with multimask
        self.predictor.set_image(image_rgb)
        input_box = np.array(expanded_bbox)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True
        )
        
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Crop with 40px padding, keep background
        x1, y1, x2, y2 = expanded_bbox
        padding = 40
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        cropped = image[y1:y2, x1:x2].copy()
        
        if output_path is None:
            output_path = str(Path(image_path).parent / "bed_hybrid_cropped.jpg")
        
        cv2.imwrite(output_path, cropped)
        
        return {
            "cropped_path": output_path,
            "bbox": expanded_bbox,
            "method": "improved_bed"
        }
    
    def process_small_object_simple(
        self,
        image_path: str,
        bbox: List[float],
        furniture_type: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Process small objects using SIMPLE method (this worked best for nightstand)
        - Minimal expansion (just 20px padding)
        - Simple bbox cropping
        - No SAM processing (keeps original shape)
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Simple bbox with minimal padding (like old method)
        x1, y1, x2, y2 = bbox
        padding = 20  # Minimal padding
        
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        # Simple crop
        cropped = image[y1:y2, x1:x2].copy()
        
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{furniture_type}_hybrid_cropped.jpg")
        
        cv2.imwrite(output_path, cropped)
        
        return {
            "cropped_path": output_path,
            "bbox": [x1, y1, x2, y2],
            "method": "simple_small_object"
        }
    
    def process_furniture(
        self,
        image_path: str,
        bbox: List[float],
        furniture_type: str,
        output_path: Optional[str] = None
    ) -> Dict:
        """
        Hybrid processing: Use best method for each object type
        """
        if furniture_type == "bed":
            # Use simple method for beds too - SAM expansion was including too much
            return self.process_small_object_simple(image_path, bbox, furniture_type, output_path)
        else:
            # Use simple method (worked best for nightstand/chair)
            return self.process_small_object_simple(image_path, bbox, furniture_type, output_path)


if __name__ == "__main__":
    segmenter = HybridSegmenter()
    
    # Test bed
    result = segmenter.process_furniture(
        "furniture_photos/fullbed.png",
        bbox=[100, 100, 400, 500],
        furniture_type="bed"
    )
    print(f"Bed: {result['cropped_path']} ({result['method']})")
    
    # Test nightstand
    result = segmenter.process_furniture(
        "furniture_photos/fullbed.png",
        bbox=[200, 400, 400, 600],
        furniture_type="nightstand"
    )
    print(f"Nightstand: {result['cropped_path']} ({result['method']})")

