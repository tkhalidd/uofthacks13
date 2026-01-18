"""
Improved segmentation for better 3D model generation
- Expands bounding boxes to include full objects
- Uses better SAM settings
- Preserves context for depth estimation
"""

import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2

try:
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    import torch
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False


class ImprovedSegmenter:
    """
    Improved segmentation that:
    1. Expands bounding boxes to capture full objects
    2. Uses better SAM settings for large objects like beds
    3. Preserves context for better 3D depth estimation
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
        
        # Also initialize automatic mask generator for better results
        self.mask_generator = SamAutomaticMaskGenerator(sam)
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
    
    def expand_bbox(
        self,
        bbox: List[float],
        image_shape: Tuple[int, int],
        expansion_factor: float = 0.2,
        min_expansion: int = 50
    ) -> List[float]:
        """
        Expand bounding box to capture more context
        
        Args:
            bbox: [x1, y1, x2, y2]
            image_shape: (height, width)
            expansion_factor: Fraction to expand (0.2 = 20% on each side)
            min_expansion: Minimum pixels to expand
        
        Returns:
            Expanded bbox [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        h, w = image_shape[:2]
        
        # Calculate expansion
        width = x2 - x1
        height = y2 - y1
        expand_x = max(min_expansion, int(width * expansion_factor))
        expand_y = max(min_expansion, int(height * expansion_factor))
        
        # Expand
        x1 = max(0, int(x1) - expand_x)
        y1 = max(0, int(y1) - expand_y)
        x2 = min(w, int(x2) + expand_x)
        y2 = min(h, int(y2) + expand_y)
        
        return [x1, y1, x2, y2]
    
    def segment_with_multimask(
        self,
        image_path: str,
        bbox: List[float],
        use_expanded_bbox: bool = True
    ) -> Dict:
        """
        Segment using multiple masks and pick the best one
        
        Args:
            image_path: Path to image
            bbox: Original bounding box
            use_expanded_bbox: Whether to expand bbox first
        
        Returns:
            Dict with best mask and metadata
        """
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Expand bbox for better context
        if use_expanded_bbox:
            bbox = self.expand_bbox(bbox, image.shape)
        
        self.predictor.set_image(image_rgb)
        
        # Use multimask to get better results
        input_box = np.array(bbox)
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True  # Get multiple masks, pick best
        )
        
        # Pick the mask with highest score
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = float(scores[best_idx])
        
        return {
            "mask": mask,
            "score": score,
            "bbox": bbox,
            "all_masks": masks,
            "all_scores": scores
        }
    
    def crop_with_context(
        self,
        image_path: str,
        mask: np.ndarray,
        bbox: List[float],
        padding: int = 50,
        keep_background: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """
        Crop image preserving context for better 3D generation
        
        Args:
            image_path: Path to original image
            mask: Segmentation mask
            bbox: Bounding box
            padding: Extra padding around object
            keep_background: If True, keep original background (better for depth)
            output_path: Where to save
        
        Returns:
            Path to cropped image
        """
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        
        # Use expanded bbox for cropping
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        # Crop image
        cropped = image[y1:y2, x1:x2].copy()
        
        # Optionally apply mask to remove background
        if not keep_background:
            mask_cropped = mask[y1:y2, x1:x2]
            cropped[mask_cropped == 0] = [255, 255, 255]  # White background
        
        # Save
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_improved_cropped.jpg")
        
        cv2.imwrite(output_path, cropped)
        return output_path
    
    def process_furniture(
        self,
        image_path: str,
        bbox: List[float],
        furniture_type: str,
        output_path: Optional[str] = None,
        keep_background: bool = True
    ) -> Dict:
        """
        Complete processing pipeline for furniture
        
        Args:
            image_path: Original image
            bbox: Detection bounding box
            furniture_type: Type of furniture (bed, desk, etc.)
            output_path: Where to save cropped image
            keep_background: Keep background for better depth estimation
        
        Returns:
            Dict with all results
        """
        # Special handling for beds (they're large and need more context)
        if furniture_type == "bed":
            expansion_factor = 0.3  # 30% expansion
            padding = 80
        else:
            expansion_factor = 0.2
            padding = 50
        
        # Segment with expanded bbox
        seg_result = self.segment_with_multimask(
            image_path,
            bbox,
            use_expanded_bbox=True
        )
        
        # Crop with context
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{furniture_type}_improved_cropped.jpg")
        
        cropped_path = self.crop_with_context(
            image_path,
            seg_result["mask"],
            seg_result["bbox"],
            padding=padding,
            keep_background=keep_background,
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
    # Test
    segmenter = ImprovedSegmenter()
    result = segmenter.process_furniture(
        "furniture_photos/fullbed.png",
        bbox=[100, 100, 400, 500],
        furniture_type="bed"
    )
    print(f"Cropped image: {result['cropped_path']}")

