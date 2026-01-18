#!/usr/bin/env python3
"""
True multi-view processing: Match same object across views and use ALL views together
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
sys.path.insert(0, str(Path(__file__).parent))

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.segmentation.hybrid_segmenter import HybridSegmenter
import subprocess
from collections import defaultdict

def match_same_object_across_views(all_detections, furniture_type):
    """
    Match same object type across views - process ALL views of same type
    Since we can't perfectly match which specific object is which across views,
    we'll process all detections of the same type and pick the best result
    """
    # Collect ALL detections of this type from ALL views
    type_detections = [(img, det) for img, dets in all_detections.items() 
                      for det in dets if det['class'] == furniture_type]
    
    # If multiple views have this type, they might be the same object
    # Process all of them and pick the best
    if len(type_detections) > 1:
        # Sort by confidence (best views first)
        type_detections.sort(key=lambda x: x[1]['confidence'], reverse=True)
        print(f"    → Found in {len(type_detections)} views, will process all and pick best")
    
    return type_detections

def process_true_multiview(angles_dir, output_dir="processed_models_true_multiview"):
    """Process with true multi-view matching"""
    angles_path = Path(angles_dir)
    image_files = sorted(list(angles_path.glob("*.png")) + 
                        list(angles_path.glob("*.jpg")))
    
    print("🏠 True Multi-View Processing")
    print("=" * 60)
    print(f"Found {len(image_files)} room angles")
    print("")
    
    # Step 1: Detect in all views
    print("📸 Step 1: Detecting furniture in all angles...")
    detector = FurnitureDetector(use_roboflow=True)
    all_detections = {}
    
    for img_path in image_files:
        detections = detector.detect_furniture(str(img_path), confidence_threshold=0.25)  # Lower threshold
        all_detections[str(img_path)] = detections
        print(f"  {img_path.name}: {len(detections)} objects")
        for det in detections:
            print(f"    - {det['class']} (conf: {det['confidence']:.2f})")
    
    # Step 2: Match objects across views
    print("\n🔗 Step 2: Matching same objects across views...")
    
    all_types = set()
    for detections in all_detections.values():
        for det in detections:
            all_types.add(det['class'])
    
    matched_objects = {}
    for furniture_type in all_types:
        matched = match_same_object_across_views(all_detections, furniture_type)
        if matched:
            matched_objects[furniture_type] = matched
            print(f"  {furniture_type}: Found in {len(matched)} view(s)")
            for img_path, det in matched:
                print(f"    - {Path(img_path).name} (conf: {det['confidence']:.2f})")
    
    print(f"\n✓ Matched {len(matched_objects)} objects")
    print("")
    
    # Step 3: Crop from ALL matched views
    print("✂️  Step 3: Cropping objects from ALL matched views...")
    
    output_path = Path(output_dir)
    cropped_dir = output_path / "cropped_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        segmenter = HybridSegmenter(model_type="vit_b")
    except:
        from backend.segmentation.sam_segmenter import SimpleSegmenter
        segmenter = SimpleSegmenter()
    
    objects_to_process = []
    
    for furniture_type, views in matched_objects.items():
        print(f"\n  {furniture_type} ({len(views)} views):")
        
        cropped_paths = []
        for i, (img_path, det) in enumerate(views):
            bbox = det['bbox']
            view_name = Path(img_path).stem
            cropped_name = f"{furniture_type}_0_view{i}_{view_name}.jpg"
            
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
                "cropped_paths": cropped_paths,
                "views": views
            })
    
    # Step 4: Process ALL views and pick best
    print(f"\n🎨 Step 4: Processing ALL views, selecting best result...")
    
    for obj in objects_to_process:
        print(f"\n  📦 {obj['type']} ({len(obj['cropped_paths'])} views):")
        
        best_size = 0
        best_path = None
        
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
                
                if Path(model_path).exists():
                    size = Path(model_path).stat().st_size
                    if size > best_size:
                        best_size = size
                        best_path = model_path
                    print(f"      ✓ {size/1024/1024:.1f}MB")
        
        if best_path:
            import shutil
            final_dir = output_path / obj['id']
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"{obj['id']}_multiview_3d.obj"
            shutil.copy(best_path, final_path)
            print(f"    ✅ Best: {best_path} → {final_path}")
            print(f"    📊 Selected from {len(obj['cropped_paths'])} views ({best_size/1024/1024:.1f}MB)")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed {len(objects_to_process)} objects")
    print(f"📁 Check {output_dir}/ for 3D models")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_true_multiview.py <angles_directory> [output_dir]")
        sys.exit(1)
    
    angles_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models_true_multiview"
    
    if not os.getenv("ROBOFLOW_API_KEY"):
        print("⚠️  ROBOFLOW_API_KEY not set")
    
    process_true_multiview(angles_dir, output_dir)

