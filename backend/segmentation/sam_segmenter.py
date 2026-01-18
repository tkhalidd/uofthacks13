"""
SAM (Segment Anything Model) Integration
Provides precise pixel-level segmentation for better object isolation
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
    print("Warning: segment-anything not installed. Install with: pip install git+https://github.com/facebookresearch/segment-anything.git")


class SAMSegmenter:
    """
    Segment objects using Meta's Segment Anything Model (SAM)
    Provides pixel-perfect masks for better object isolation
    """
    
    def __init__(
        self,
        model_type: str = "vit_h",  # "vit_h", "vit_l", "vit_b"
        checkpoint_path: Optional[str] = None,
        device: str = "cuda"
    ):
        """
        Initialize SAM segmenter
        
        Args:
            model_type: SAM model variant (vit_h = best quality, vit_b = fastest)
            checkpoint_path: Path to SAM checkpoint. If None, will try to download
            device: "cuda", "cpu", or "mps"
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything not installed. "
                "Install with: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        
        # Default checkpoint paths (user should download these)
        if checkpoint_path is None:
            checkpoint_path = self._get_default_checkpoint_path(model_type)
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"SAM checkpoint not found: {checkpoint_path}\n"
                f"Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints"
            )
        
        # Load SAM model
        print(f"Loading SAM model ({model_type}) from {checkpoint_path}...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        self.predictor = SamPredictor(self.sam)
        print(f"✓ SAM model loaded on {self.device}")
    
    def _get_default_checkpoint_path(self, model_type: str) -> str:
        """Get default checkpoint path based on model type"""
        home = Path.home()
        checkpoints_dir = home / ".sam_checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        checkpoint_map = {
            "vit_h": "sam_vit_h_4b8939.pth",
            "vit_l": "sam_vit_l_0b3195.pth",
            "vit_b": "sam_vit_b_01ec64.pth"
        }
        
        return str(checkpoints_dir / checkpoint_map.get(model_type, "sam_vit_h_4b8939.pth"))
    
    def segment_from_bbox(
        self,
        image_path: str,
        bbox: List[float],  # [x1, y1, x2, y2]
        output_mask_path: Optional[str] = None
    ) -> Dict:
        """
        Segment object from bounding box
        
        Args:
            image_path: Path to image
            bbox: Bounding box [x1, y1, x2, y2]
            output_mask_path: Optional path to save mask image
        
        Returns:
            Dict with mask array and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image in predictor
        self.predictor.set_image(image_rgb)
        
        # Convert bbox to format SAM expects: [x1, y1, x2, y2]
        input_box = np.array(bbox)
        
        # Predict mask
        masks, scores, logits = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False
        )
        
        mask = masks[0]  # Get best mask
        score = float(scores[0])
        
        # Save mask if requested
        if output_mask_path:
            mask_image = (mask * 255).astype(np.uint8)
            cv2.imwrite(output_mask_path, mask_image)
        
        return {
            "mask": mask,
            "score": score,
            "bbox": bbox,
            "mask_path": output_mask_path
        }
    
    def segment_from_point(
        self,
        image_path: str,
        point: Tuple[int, int],  # (x, y)
        point_label: int = 1,  # 1 = foreground, 0 = background
        output_mask_path: Optional[str] = None
    ) -> Dict:
        """
        Segment object from a single point click
        
        Args:
            image_path: Path to image
            point: Point coordinates (x, y)
            point_label: 1 for foreground, 0 for background
            output_mask_path: Optional path to save mask
        
        Returns:
            Dict with mask and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Set image
        self.predictor.set_image(image_rgb)
        
        # Predict from point
        masks, scores, logits = self.predictor.predict(
            point_coords=np.array([point]),
            point_labels=np.array([point_label]),
            multimask_output=True
        )
        
        # Get best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]
        score = float(scores[best_idx])
        
        # Save mask if requested
        if output_mask_path:
            mask_image = (mask * 255).astype(np.uint8)
            cv2.imwrite(output_mask_path, mask_image)
        
        return {
            "mask": mask,
            "score": score,
            "point": point,
            "mask_path": output_mask_path
        }
    
    def crop_with_mask(
        self,
        image_path: str,
        mask: np.ndarray,
        padding: int = 20,
        output_path: Optional[str] = None
    ) -> str:
        """
        Crop image using SAM mask with padding
        
        Args:
            image_path: Path to original image
            mask: Binary mask from SAM
            padding: Pixels to add around mask
            output_path: Where to save cropped image
        
        Returns:
            Path to cropped image
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Find bounding box of mask
        y_indices, x_indices = np.where(mask > 0)
        if len(x_indices) == 0 or len(y_indices) == 0:
            raise ValueError("Empty mask - no object found")
        
        x_min, x_max = int(x_indices.min()) - padding, int(x_indices.max()) + padding
        y_min, y_max = int(y_indices.min()) - padding, int(y_indices.max()) + padding
        
        # Clamp to image bounds
        h, w = image.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        # Crop image
        cropped = image[y_min:y_max, x_min:x_max]
        
        # Apply mask to cropped region (remove background)
        mask_cropped = mask[y_min:y_max, x_min:x_max]
        cropped_masked = cropped.copy()
        cropped_masked[mask_cropped == 0] = [255, 255, 255]  # White background
        
        # Save
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_sam_cropped.jpg")
        
        cv2.imwrite(output_path, cropped_masked)
        
        return output_path
    
    def segment_multiple_objects(
        self,
        image_path: str,
        bboxes: List[List[float]],
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Segment multiple objects from a single image
        
        Args:
            image_path: Path to image
            bboxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            output_dir: Directory to save cropped images
        
        Returns:
            List of segmentation results
        """
        results = []
        
        for i, bbox in enumerate(bboxes):
            # Segment
            seg_result = self.segment_from_bbox(image_path, bbox)
            
            # Crop with mask
            if output_dir:
                output_path = Path(output_dir) / f"object_{i}_cropped.jpg"
                cropped_path = self.crop_with_mask(
                    image_path,
                    seg_result["mask"],
                    output_path=str(output_path)
                )
                seg_result["cropped_path"] = cropped_path
            
            results.append(seg_result)
        
        return results


# Lightweight version that doesn't require SAM (uses simple bbox cropping)
class SimpleSegmenter:
    """
    Simple segmenter that just crops using bounding boxes
    Use this if SAM is not available
    """
    
    def crop_from_bbox(
        self,
        image_path: str,
        bbox: List[float],
        padding: int = 20,
        output_path: Optional[str] = None
    ) -> str:
        """Crop image using bounding box"""
        from PIL import Image
        
        img = Image.open(image_path)
        x1, y1, x2, y2 = bbox
        
        # Add padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(img.width, int(x2) + padding)
        y2 = min(img.height, int(y2) + padding)
        
        # Crop
        cropped = img.crop((x1, y1, x2, y2))
        
        # Save
        if output_path is None:
            output_path = str(Path(image_path).parent / f"{Path(image_path).stem}_cropped.jpg")
        
        cropped.save(output_path)
        return output_path


# Example usage
if __name__ == "__main__":
    # Option 1: Use SAM (better quality)
    if SAM_AVAILABLE:
        segmenter = SAMSegmenter(model_type="vit_b")  # Fastest model
        
        # Segment from bbox
        result = segmenter.segment_from_bbox(
            "test_image.jpg",
            bbox=[100, 200, 500, 600],
            output_mask_path="mask.png"
        )
        
        # Crop with mask
        cropped = segmenter.crop_with_mask(
            "test_image.jpg",
            result["mask"],
            output_path="cropped_object.jpg"
        )
        print(f"Cropped image saved to: {cropped}")
    else:
        # Option 2: Simple cropping (fallback)
        segmenter = SimpleSegmenter()
        cropped = segmenter.crop_from_bbox(
            "test_image.jpg",
            bbox=[100, 200, 500, 600]
        )
        print(f"Cropped image saved to: {cropped}")

