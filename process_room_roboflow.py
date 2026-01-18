#!/usr/bin/env python3
"""
Process room photo using Roboflow furniture detection (better for desk/nightstand)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.segmentation.sam_segmenter import SAMSegmenter
import subprocess
import json

def process_room_with_roboflow(image_path, output_dir="processed_models"):
    """Process room photo using Roboflow + SAM + TripoSR"""
    
    print("🏠 Processing Room Photo with Roboflow Furniture Detection")
    print("=" * 60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}\n")
    
    # Check for API key
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("⚠️  ROBOFLOW_API_KEY not set!")
        print("   Get your free API key from: https://roboflow.com/settings")
        print("   Then run: export ROBOFLOW_API_KEY='your_key_here'")
        print("\n   Falling back to YOLO (may miss desk/nightstand)...\n")
        use_roboflow = False
    else:
        use_roboflow = True
    
    # Step 1: Detect furniture
    print("📸 Step 1: Detecting furniture...")
    detector = FurnitureDetector(use_roboflow=use_roboflow)
    detections = detector.detect_furniture(image_path, confidence_threshold=0.05)
    
    print(f"✓ Found {len(detections)} furniture pieces:")
    for det in detections:
        print(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
    
    if len(detections) == 0:
        print("\n❌ No furniture detected. Try:")
        print("  1. Lower confidence threshold")
        print("  2. Check if ROBOFLOW_API_KEY is set correctly")
        return
    
    # Step 2: Segment with HYBRID approach (best of both worlds)
    print("\n✂️  Step 2: Segmenting objects with HYBRID approach...")
    print("  • Beds: Improved method (expanded bbox, context) - BEST for beds")
    print("  • Small objects: Simple method (minimal processing) - BEST for nightstand/chair")
    try:
        from backend.segmentation.hybrid_segmenter import HybridSegmenter
        segmenter = HybridSegmenter(model_type="vit_b")
        print("  ✓ Hybrid segmenter loaded")
        use_hybrid = True
    except Exception as e:
        print(f"  ⚠️  Hybrid segmenter not available: {e}")
        print("  Falling back to smart SAM...")
        try:
            from backend.segmentation.smart_segmenter import SmartSegmenter
            segmenter = SmartSegmenter(model_type="vit_b")
            print("  ✓ Smart SAM loaded")
            use_hybrid = False
        except Exception as e2:
            print(f"  ⚠️  Smart segmenter not available: {e2}")
            print("  Falling back to improved SAM...")
            try:
                from backend.segmentation.improved_segmenter import ImprovedSegmenter
                segmenter = ImprovedSegmenter(model_type="vit_b")
                print("  ✓ Improved SAM loaded")
                use_hybrid = False
            except Exception as e3:
                print(f"  ⚠️  Improved segmenter not available: {e3}")
                print("  Using simple cropping...")
                from backend.segmentation.sam_segmenter import SimpleSegmenter
                segmenter = SimpleSegmenter()
                use_hybrid = False
    
    # Create output directories
    output_path = Path(output_dir)
    cropped_dir = output_path / "cropped_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each detected object
    objects_to_process = []
    
    for i, det in enumerate(detections):
        furniture_id = f"{det['class']}_{i}"
        bbox = det['bbox']
        
        print(f"\n  Processing {det['class']}...")
        
        # Use hybrid segmenter if available (BEST approach)
        if use_hybrid:
            try:
                result = segmenter.process_furniture(
                    image_path,
                    bbox,
                    furniture_type=det['class'],
                    output_path=str(cropped_dir / f"{furniture_id}_hybrid_cropped.jpg")
                )
                cropped_path = result['cropped_path']
                method = result.get('method', 'hybrid')
                print(f"    ✓ Hybrid segmentation ({method}) → {cropped_path}")
                if 'bbox' in result:
                    print(f"    ✓ Bbox: {result['bbox']}")
            except Exception as e:
                print(f"    ⚠️  Hybrid segmentation failed: {e}")
                use_hybrid = False  # Fall back for remaining objects
        
        # Fallback chain
        if not use_hybrid:
            # Try smart segmenter
            try:
                from backend.segmentation.smart_segmenter import SmartSegmenter
                smart_segmenter = SmartSegmenter(model_type="vit_b")
                bbox_area = det.get('area', (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                result = smart_segmenter.process_furniture(
                    image_path,
                    bbox,
                    furniture_type=det['class'],
                    bbox_area=bbox_area,
                    output_path=str(cropped_dir / f"{furniture_id}_smart_cropped.jpg"),
                    keep_background=True
                )
                cropped_path = result['cropped_path']
                print(f"    ✓ Smart segmentation → {cropped_path}")
            except Exception as e:
                # Try improved segmenter
                try:
                    from backend.segmentation.improved_segmenter import ImprovedSegmenter
                    improved_segmenter = ImprovedSegmenter(model_type="vit_b")
                    result = improved_segmenter.process_furniture(
                        image_path,
                        bbox,
                        furniture_type=det['class'],
                        output_path=str(cropped_dir / f"{furniture_id}_improved_cropped.jpg"),
                        keep_background=True
                    )
                    cropped_path = result['cropped_path']
                    print(f"    ✓ Improved segmentation → {cropped_path}")
                except Exception as e2:
                    # Final fallback to standard SAM or simple cropping
                    if hasattr(segmenter, 'segment_from_bbox'):
                        try:
                            seg_result = segmenter.segment_from_bbox(image_path, bbox)
                            cropped_path = segmenter.crop_with_mask(
                                image_path,
                                seg_result['mask'],
                                output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
                            )
                            print(f"    ✓ Segmented and cropped → {cropped_path}")
                        except Exception as e3:
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
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed {len(objects_to_process)} objects")
    print(f"📁 Check {output_dir}/ for 3D models")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_room_roboflow.py <image_path> [output_dir]")
        print("\nExample:")
        print("  export ROBOFLOW_API_KEY='your_key'")
        print("  python process_room_roboflow.py furniture_photos/fullbed.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models"
    
    if not Path(image_path).exists():
        print(f"❌ Error: File not found: {image_path}")
        sys.exit(1)
    
    process_room_with_roboflow(image_path, output_dir)

