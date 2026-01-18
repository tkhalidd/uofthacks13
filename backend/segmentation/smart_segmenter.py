"""
Smart segmentation that:
1. Isolates objects better (excludes nearby furniture)
2. Adjusts expansion based on object size
3. Uses mask refinement for better boundaries
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


class SmartSegmenter:
    """
    Smart segmentation that adapts to object type and size
    - Beds: Moderate expansion, exclude nearby objects
    - Small objects (nightstand, chair): Less expansion, preserve shape
    - Better mask refinement
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
        
        print(f"Loading SAM model ({model_type})...")
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
    
    def expand_bbox_smart(
        self,
        bbox: List[float],
        image_shape: Tuple[int, int],
        furniture_type: str,
        bbox_area: float
    ) -> List[float]:
        """
        Smart bbox expansion based on object type and size
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width)
            furniture_type: Type of furniture
            bbox_area: Area of original bbox
        
        Returns:
            Expanded bbox
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2]
        
        width = x2 - x1
        height = y2 - y1
        
        # Determine expansion based on object type and size
        if furniture_type == "bed":
            # Beds: Moderate expansion, but not too much to avoid nearby objects
            expansion_factor = 0.15  # Reduced from 0.3
            min_expansion = 40
        elif furniture_type in ["nightstand", "chair", "table"]:
            # Small objects: Less expansion to preserve shape
            expansion_factor = 0.1
            min_expansion = 20
        else:
            # Default
            expansion_factor = 0.15
            min_expansion = 30
        
        expand_x = max(min_expansion, int(width * expansion_factor))
        expand_y = max(min_expansion, int(height * expansion_factor))
        
        # Expand
        x1 = max(0, int(x1) - expand_x)
        y1 = max(0, int(y1) - expand_y)
        x2 = min(w, int(x2) + expand_x)
        y2 = min(h, int(y2) + expand_y)
        
        return [x1, y1, x2, y2]
    
    def refine_mask(
        self,
        mask: np.ndarray,
        bbox: List[float],
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Refine mask to remove noise and isolate object better
        
        Args:
            mask: Binary mask
            bbox: Bounding box
            image_shape: Image dimensions
        
        Returns:
            Refined mask
        """
        # Convert to uint8
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        
        # Remove small holes
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        
        # Remove small noise
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        
        # Find largest connected component (main object)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        
        if num_labels > 1:
            # Get largest component (skip background = label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            
            # Create mask with only largest component
            refined_mask = (labels == largest_label).astype(np.uint8) * 255
        else:
            refined_mask = mask_uint8
        
        # Convert back to boolean
        return (refined_mask > 127).astype(bool)
    
    def segment_with_refinement(
        self,
        image_path: str,
        bbox: List[float],
        furniture_type: str,
        bbox_area: float
    ) -> Dict:
        """
        Segment with smart expansion and mask refinement
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Smart expansion
        expanded_bbox = self.expand_bbox_smart(bbox, image.shape, furniture_type, bbox_area)
        
        self.predictor.set_image(image_rgb)
        
        # Use multimask to get better results
        input_box = np.array(expanded_bbox)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True
        )
        
        # Pick the mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        
        # Refine mask to isolate object better
        refined_mask = self.refine_mask(mask, expanded_bbox, image.shape)
        
        return {
            "mask": refined_mask,
            "score": float(scores[best_idx]),
            "bbox": expanded_bbox,
            "original_mask": mask
        }
    
    def crop_with_smart_context(
        self,
        image_path: str,
        mask: np.ndarray,
        bbox: List[float],
        furniture_type: str,
        padding: int = None,
        keep_background: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """
        Crop with smart padding based on object type
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Smart padding based on object type
        if padding is None:
            if furniture_type == "bed":
                padding = 50  # Reduced from 80
            elif furniture_type in ["nightstand", "chair"]:
                padding = 30  # Less padding for small objects
            else:
                padding = 40
        
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        # Crop image
        cropped = image[y1:y2, x1:x2].copy()
        
        # For small objects, optionally apply mask to remove background
        # This helps preserve their 3D shape
        if not keep_background and furniture_type in ["nightstand", "chair"]:
            mask_cropped = mask[y1:y2, x1:x2]
            # Create white background
            cropped_masked = np.ones_like(cropped) * 255
            cropped_masked[mask_cropped > 0] = cropped[mask_cropped > 0]
            cropped = cropped_masked
        
        # Save
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{furniture_type}_smart_cropped.jpg")
        
        cv2.imwrite(output_path, cropped)
        return output_path
    
    def process_furniture(
        self,
        image_path: str,
        bbox: List[float],
        furniture_type: str,
        bbox_area: float = None,
        output_path: Optional[str] = None,
        keep_background: bool = True
    ) -> Dict:
        """
        Complete smart processing pipeline
        """
        # Calculate bbox area if not provided
        if bbox_area is None:
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Segment with refinement
        seg_result = self.segment_with_refinement(
            image_path,
            bbox,
            furniture_type,
            bbox_area
        )
        
        # Determine if we should keep background
        # Small objects benefit from isolated background
        # Large objects (beds) benefit from context
        if furniture_type in ["nightstand", "chair"]:
            keep_bg = False  # Isolate small objects
        else:
            keep_bg = keep_background  # Keep context for large objects
        
        # Crop with smart context
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{furniture_type}_smart_cropped.jpg")
        
        cropped_path = self.crop_with_smart_context(
            image_path,
            seg_result["mask"],
            seg_result["bbox"],
            furniture_type,
            keep_background=keep_bg,
            output_path=output_path
        )
        
        return {
            "cropped_path": cropped_path,
            "mask": seg_result["mask"],
            "score": seg_result["score"],
            "original_bbox": bbox,
            "expanded_bbox": seg_result["bbox"]
        }


if __name__ == "__main__":
    segmenter = SmartSegmenter()
    result = segmenter.process_furniture(
        "furniture_photos/fullbed.png",
        bbox=[100, 100, 400, 500],
        furniture_type="bed",
        bbox_area=150000
    )
    print(f"Cropped image: {result['cropped_path']}")

