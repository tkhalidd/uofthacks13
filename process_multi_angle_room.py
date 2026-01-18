#!/usr/bin/env python3
"""
Process multiple room angles to generate better 3D furniture models
- Detects furniture in all angles
- Matches detections across views
- Uses best view for each object (or combines views)
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.segmentation.hybrid_segmenter import HybridSegmenter
import subprocess
import json
from collections import defaultdict

def process_multi_angle_room(angles_dir, output_dir="processed_models_multiangle"):
    """
    Process room from multiple angles for better 3D generation
    
    Args:
        angles_dir: Directory containing room photos from different angles
        output_dir: Where to save 3D models
    """
    angles_path = Path(angles_dir)
    if not angles_path.exists():
        print(f"❌ Error: Directory not found: {angles_dir}")
        return
    
    # Find all images
    image_files = list(angles_path.glob("*.jpg")) + \
                  list(angles_path.glob("*.png")) + \
                  list(angles_path.glob("*.jpeg"))
    
    if len(image_files) == 0:
        print(f"❌ Error: No images found in {angles_dir}")
        return
    
    print("🏠 Processing Room from Multiple Angles")
    print("=" * 60)
    print(f"Found {len(image_files)} room angles:")
    for img in image_files:
        print(f"  - {img.name}")
    print("")
    
    # Step 1: Detect furniture in all angles
    print("📸 Step 1: Detecting furniture in all angles...")
    detector = FurnitureDetector(use_roboflow=True)
    
    all_detections = {}  # {image_path: [detections]}
    furniture_views = defaultdict(list)  # {furniture_type: [(image_path, detection), ...]}
    
    for img_path in image_files:
        detections = detector.detect_furniture(str(img_path), confidence_threshold=0.3)
        all_detections[str(img_path)] = detections
        
        for det in detections:
            furniture_views[det['class']].append((str(img_path), det))
        
        print(f"  {img_path.name}: Found {len(detections)} objects")
        for det in detections:
            print(f"    - {det['class']} (confidence: {det['confidence']:.2f})")
    
    print(f"\n✓ Total unique furniture types: {len(furniture_views)}")
    print("")
    
    # Step 2: For each furniture type, pick the best view
    print("🎯 Step 2: Selecting best view for each furniture piece...")
    
    output_path = Path(output_dir)
    cropped_dir = output_path / "cropped_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize segmenter
    try:
        segmenter = HybridSegmenter(model_type="vit_b")
        print("  ✓ Hybrid segmenter loaded")
    except Exception as e:
        print(f"  ⚠️  Hybrid segmenter not available: {e}")
        from backend.segmentation.sam_segmenter import SimpleSegmenter
        segmenter = SimpleSegmenter()
        print("  Using simple segmenter")
    
    objects_to_process = []
    
    for furniture_type, views in furniture_views.items():
        print(f"\n  Processing {furniture_type}...")
        
        # Pick best view (highest confidence, or first if same)
        best_view = max(views, key=lambda x: x[1]['confidence'])
        best_image_path, best_detection = best_view
        
        print(f"    Best view: {Path(best_image_path).name} (confidence: {best_detection['confidence']:.2f})")
        
        # Segment and crop
        furniture_id = f"{furniture_type}_0"
        bbox = best_detection['bbox']
        
        try:
            if hasattr(segmenter, 'process_furniture'):
                result = segmenter.process_furniture(
                    best_image_path,
                    bbox,
                    furniture_type=furniture_type,
                    output_path=str(cropped_dir / f"{furniture_id}_multiangle_cropped.jpg")
                )
                cropped_path = result['cropped_path']
            else:
                # Fallback to simple cropping
                from PIL import Image
                img = Image.open(best_image_path)
                x1, y1, x2, y2 = bbox
                padding = 20
                cropped = img.crop((
                    max(0, int(x1) - padding),
                    max(0, int(y1) - padding),
                    min(img.width, int(x2) + padding),
                    min(img.height, int(y2) + padding)
                ))
                cropped_path = str(cropped_dir / f"{furniture_id}_multiangle_cropped.jpg")
                cropped.save(cropped_path)
            
            print(f"    ✓ Cropped → {cropped_path}")
            
            objects_to_process.append({
                "id": furniture_id,
                "type": furniture_type,
                "cropped_path": cropped_path,
                "source_image": best_image_path,
                "confidence": best_detection['confidence']
            })
        except Exception as e:
            print(f"    ⚠️  Failed to crop: {e}")
    
    # Step 3: Generate 3D models
    print(f"\n🎨 Step 3: Generating 3D models for {len(objects_to_process)} objects...")
    print("   (Using best view for each object)")
    
    for obj in objects_to_process:
        print(f"\n  📦 Processing {obj['type']}...")
        print(f"    Source: {Path(obj['source_image']).name}")
        cropped_path = obj['cropped_path']
        
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
    print(f"✅ Complete! Processed {len(objects_to_process)} objects from {len(image_files)} angles")
    print(f"📁 Check {output_dir}/ for 3D models")
    print("")
    print("💡 Tip: Objects were generated from their best view (highest confidence)")
    print("   For even better results, take dedicated photos of each object from multiple angles")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_multi_angle_room.py <angles_directory> [output_dir]")
        print("\nExample:")
        print("  python process_multi_angle_room.py angles/ processed_models_multiangle/")
        print("")
        print("The script will:")
        print("  1. Detect furniture in all room angles")
        print("  2. Pick the best view for each furniture piece")
        print("  3. Generate 3D models from best views")
        sys.exit(1)
    
    angles_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models_multiangle"
    
    # Check for API key
    if not os.getenv("ROBOFLOW_API_KEY"):
        print("⚠️  Warning: ROBOFLOW_API_KEY not set")
        print("   Set it with: export ROBOFLOW_API_KEY='your_key'")
        print("   Will fall back to YOLO (may miss desk/nightstand)")
        print("")
    
    process_multi_angle_room(angles_dir, output_dir)

