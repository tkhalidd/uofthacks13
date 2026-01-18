#!/usr/bin/env python3
"""
Process room photo using SAM to find ALL objects, not just YOLO detections
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.segmentation.sam_segmenter import SAMSegmenter
from backend.segmentation.furniture_detector import FurnitureDetector
import cv2
import numpy as np
from PIL import Image
import subprocess
import json

def process_room_photo(image_path, output_dir="processed_models"):
    """Process room photo: Detect + SAM segment all objects + Generate 3D"""
    
    print("🏠 Processing Room Photo with SAM")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: YOLO detection
    print("📸 Step 1: Detecting furniture with YOLO...")
    detector = FurnitureDetector()
    yolo_detections = detector.detect_furniture(image_path, confidence_threshold=0.3)
    print(f"✓ YOLO found {len(yolo_detections)} objects")
    for det in yolo_detections:
        print(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
    
    # Step 2: Use SAM to segment all detected objects
    print("\n✂️  Step 2: Segmenting with SAM...")
    try:
        segmenter = SAMSegmenter(model_type="vit_b")
        print("  ✓ SAM loaded")
    except Exception as e:
        print(f"  ⚠️  SAM not available: {e}")
        print("  Falling back to simple cropping")
        from backend.segmentation.sam_segmenter import SimpleSegmenter
        segmenter = SimpleSegmenter()
    
    # Create output directories
    output_path = Path(output_dir)
    cropped_dir = output_path / "cropped_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each detected object
    objects_to_process = []
    
    for i, det in enumerate(yolo_detections):
        furniture_id = f"{det['class']}_{i}"
        bbox = det['bbox']
        
        print(f"\n  Processing {det['class']}...")
        
        # Segment with SAM
        if hasattr(segmenter, 'segment_from_bbox'):
            try:
                seg_result = segmenter.segment_from_bbox(image_path, bbox)
                cropped_path = segmenter.crop_with_mask(
                    image_path,
                    seg_result['mask'],
                    output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
                )
                print(f"    ✓ Segmented and cropped → {cropped_path}")
            except Exception as e:
                print(f"    ⚠️  SAM segmentation failed: {e}")
                # Fallback to simple cropping
                cropped_path = segmenter.crop_from_bbox(
                    image_path,
                    bbox,
                    output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
                )
                print(f"    ✓ Simple cropped → {cropped_path}")
        else:
            cropped_path = segmenter.crop_from_bbox(
                image_path,
                bbox,
                output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
            )
            print(f"    ✓ Cropped → {cropped_path}")
        
        objects_to_process.append({
            "id": furniture_id,
            "type": det['class'],
            "cropped_path": cropped_path
        })
    
    # Step 3: Generate 3D models
    print(f"\n🎨 Step 3: Generating 3D models for {len(objects_to_process)} objects...")
    
    for obj in objects_to_process:
        print(f"\n  📦 Processing {obj['type']}...")
        cropped_path = obj['cropped_path']
        
        # Use existing script
        result = subprocess.run(
            ["./process_to_3d.sh", cropped_path],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"    ✅ {obj['type']} 3D model generated!")
        else:
            print(f"    ❌ {obj['type']} failed")
            print(f"    Error: {result.stderr[:200]}")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed {len(objects_to_process)} objects")
    print(f"📁 Check {output_dir}/ for 3D models")
    
    # If YOLO only found 1 object but user said there are 3, suggest manual cropping
    if len(yolo_detections) < 3:
        print(f"\n⚠️  Note: Only detected {len(yolo_detections)} object(s).")
        print("   If there are more objects (desk, nightstand), you can:")
        print("   1. Manually crop them using an image editor")
        print("   2. Save as separate images")
        print("   3. Run: ./process_to_3d.sh <cropped_image.jpg>")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_room_with_sam_all_objects.py <image_path> [output_dir]")
        print("\nExample:")
        print("  python process_room_with_sam_all_objects.py furniture_photos/fullbed.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models"
    
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    process_room_photo(image_path, output_dir)

