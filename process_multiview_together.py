#!/usr/bin/env python3
"""
Process multiple angles of the SAME object together for better 3D generation
- Matches same object across views
- Crops from all views
- Processes all views together (TripoSR can take multiple images)
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import subprocess
import shutil
sys.path.insert(0, str(Path(__file__).parent))

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.segmentation.hybrid_segmenter import HybridSegmenter
from collections import defaultdict

def match_object_across_views(all_detections, furniture_type):
    """
    Match same object type across multiple views
    Returns all views where this type appears (could be same or different objects)
    """
    type_detections = [(img, det) for img, dets in all_detections.items() 
                      for det in dets if det['class'] == furniture_type]
    
    # Sort by confidence
    type_detections.sort(key=lambda x: x[1]['confidence'], reverse=True)
    return type_detections

def process_multiview_together(angles_dir, output_dir="processed_models_multiview_together"):
    """
    Process room from multiple angles, matching objects and using ALL views together
    """
    angles_path = Path(angles_dir)
    image_files = sorted(list(angles_path.glob("*.png")) + 
                        list(angles_path.glob("*.jpg")))
    
    print("🏠 Multi-View Processing (Same Object from Multiple Angles)")
    print("=" * 60)
    print(f"Found {len(image_files)} room angles:")
    for img in image_files:
        print(f"  - {img.name}")
    print("")
    
    # Step 1: Detect in all views
    print("📸 Step 1: Detecting furniture in all angles...")
    detector = FurnitureDetector(use_roboflow=True)
    all_detections = {}
    
    for img_path in image_files:
        detections = detector.detect_furniture(str(img_path), confidence_threshold=0.25)
        all_detections[str(img_path)] = detections
        print(f"  {img_path.name}: {len(detections)} objects")
    
    # Step 2: Match objects across views
    print("\n🔗 Step 2: Matching objects across views...")
    
    all_types = set()
    for detections in all_detections.values():
        for det in detections:
            all_types.add(det['class'])
    
    matched_objects = {}
    for furniture_type in sorted(all_types):
        views = match_object_across_views(all_detections, furniture_type)
        if views:
            matched_objects[furniture_type] = views
            print(f"  {furniture_type}: Found in {len(views)} view(s)")
            for img_path, det in views:
                print(f"    - {Path(img_path).name} (conf: {det['confidence']:.2f})")
    
    print(f"\n✓ Found {len(matched_objects)} object types across views")
    print("")
    
    # Step 3: Crop same object from ALL views
    print("✂️  Step 3: Cropping objects from ALL matched views...")
    
    output_path = Path(output_dir)
    cropped_dir = output_path / "cropped_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        segmenter = HybridSegmenter(model_type="vit_b")
        print("  ✓ Hybrid segmenter loaded")
    except:
        from backend.segmentation.sam_segmenter import SimpleSegmenter
        segmenter = SimpleSegmenter()
        print("  Using simple segmenter")
    
    objects_to_process = []
    
    for furniture_type, views in matched_objects.items():
        print(f"\n  Processing {furniture_type} from {len(views)} view(s)...")
        
        cropped_paths = []
        for i, (img_path, det) in enumerate(views):
            bbox = det['bbox']
            view_name = Path(img_path).stem
            cropped_name = f"{furniture_type}_view{i}_{view_name}.jpg"
            
            try:
                if hasattr(segmenter, 'process_furniture'):
                    result = segmenter.process_furniture(
                        img_path,
                        bbox,
                        furniture_type=furniture_type,
                        output_path=str(cropped_dir / cropped_name)
                    )
                    cropped_path = result['cropped_path']
                else:
                    img = Image.open(img_path)
                    x1, y1, x2, y2 = bbox
                    padding = 20
                    cropped = img.crop((
                        max(0, int(x1) - padding),
                        max(0, int(y1) - padding),
                        min(img.width, int(x2) + padding),
                        min(img.height, int(y2) + padding)
                    ))
                    cropped_path = str(cropped_dir / cropped_name)
                    cropped.save(cropped_path)
                
                cropped_paths.append(cropped_path)
                print(f"    ✓ View {i+1}: {Path(img_path).name}")
            except Exception as e:
                print(f"    ⚠️  View {i+1} failed: {e}")
        
        if cropped_paths:
            objects_to_process.append({
                "type": furniture_type,
                "id": f"{furniture_type}_0",
                "cropped_paths": cropped_paths,  # ALL views of same object
                "views": views
            })
    
    # Step 4: Process ALL views together
    print(f"\n🎨 Step 4: Processing ALL views together for each object...")
    print("   (Processing each view, then selecting best result)")
    print("")
    
    for obj in objects_to_process:
        print(f"  📦 {obj['type']} ({len(obj['cropped_paths'])} views):")
        
        best_size = 0
        best_path = None
        best_view_idx = None
        
        # Process each view
        for i, cropped_path in enumerate(obj['cropped_paths']):
            print(f"    Processing view {i+1}/{len(obj['cropped_paths'])}...")
            
            result = subprocess.run(
                ["./process_to_3d.sh", cropped_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                output_name = Path(cropped_path).stem
                model_path = f"processed_models/{output_name}/{output_name}_3d.obj"
                
                # Wait a bit for file to be ready
                import time
                time.sleep(2)
                
                if Path(model_path).exists():
                    size = Path(model_path).stat().st_size
                    if size > best_size:
                        best_size = size
                        best_path = model_path
                        best_view_idx = i
                    print(f"      ✓ Generated ({size/1024/1024:.1f}MB)")
                else:
                    print(f"      ⚠️  Model not found yet (may still be processing)")
            else:
                print(f"      ❌ Failed")
                if result.stderr:
                    print(f"      Error: {result.stderr[:100]}")
        
        # Save best result
        if best_path and Path(best_path).exists():
            final_dir = output_path / obj['id']
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"{obj['id']}_multiview_3d.obj"
            shutil.copy(best_path, final_path)
            
            best_view_name = Path(obj['cropped_paths'][best_view_idx]).stem
            print(f"    ✅ Best result: View {best_view_idx+1} ({best_size/1024/1024:.1f}MB)")
            print(f"    📁 Saved to: {final_path}")
        else:
            print(f"    ⚠️  No valid model generated")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed {len(objects_to_process)} objects")
    print(f"📁 Check {output_dir}/ for 3D models")
    print("")
    print("💡 Each object was processed from multiple angles")
    print("   The best result (most detail) was selected for each")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_multiview_together.py <angles_directory> [output_dir]")
        print("\nExample:")
        print("  python process_multiview_together.py furniture_photos/angles/")
        print("")
        print("This will:")
        print("  1. Detect furniture in all angles")
        print("  2. Match same objects across views")
        print("  3. Crop same object from all views")
        print("  4. Process all views and pick best result")
        sys.exit(1)
    
    angles_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models_multiview_together"
    
    if not os.getenv("ROBOFLOW_API_KEY"):
        print("⚠️  Warning: ROBOFLOW_API_KEY not set")
        print("   Set it with: export ROBOFLOW_API_KEY='your_key'")
        print("")
    
    process_multiview_together(angles_dir, output_dir)

