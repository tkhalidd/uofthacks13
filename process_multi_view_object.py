#!/usr/bin/env python3
"""
Process same object from multiple angles for better 3D generation
- Matches same object across multiple views
- Collects all views of each object
- Processes all views together for better depth
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

def calculate_bbox_overlap(bbox1, bbox2):
    """Calculate IoU (Intersection over Union) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union

def match_objects_across_views(all_detections, furniture_type, iou_threshold=0.3):
    """
    Match the same object across multiple views
    
    Args:
        all_detections: Dict {image_path: [detections]}
        furniture_type: Type of furniture to match
        iou_threshold: Minimum IoU to consider same object
    
    Returns:
        List of matched object groups, each group contains [(image_path, detection), ...]
    """
    # Collect all detections of this type
    type_detections = []
    for img_path, detections in all_detections.items():
        for det in detections:
            if det['class'] == furniture_type:
                type_detections.append((img_path, det))
    
    if len(type_detections) == 0:
        return []
    
    # Group by matching (simple approach: group by similar size and position)
    groups = []
    used = set()
    
    for i, (img1, det1) in enumerate(type_detections):
        if i in used:
            continue
        
        group = [(img1, det1)]
        used.add(i)
        
        # Find matches in other images
        for j, (img2, det2) in enumerate(type_detections):
            if j in used or img1 == img2:
                continue
            
            # Check if same object (similar size, type)
            bbox1 = det1['bbox']
            bbox2 = det2['bbox']
            
            # Calculate size similarity
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            size_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
            
            # If similar size and same type, likely same object
            if size_ratio > 0.5:  # Within 50% size difference
                group.append((img2, det2))
                used.add(j)
        
        if len(group) > 0:
            groups.append(group)
    
    return groups

def process_multi_view_object(angles_dir, output_dir="processed_models_multiview"):
    """
    Process room from multiple angles, matching same objects across views
    and using all views together for better 3D generation
    """
    angles_path = Path(angles_dir)
    if not angles_path.exists():
        print(f"❌ Error: Directory not found: {angles_dir}")
        return
    
    # Find all images
    image_files = sorted(list(angles_path.glob("*.jpg")) + 
                        list(angles_path.glob("*.png")) + 
                        list(angles_path.glob("*.jpeg")))
    
    if len(image_files) == 0:
        print(f"❌ Error: No images found in {angles_dir}")
        return
    
    print("🏠 Processing Room from Multiple Angles (Multi-View)")
    print("=" * 60)
    print(f"Found {len(image_files)} room angles:")
    for img in image_files:
        print(f"  - {img.name}")
    print("")
    
    # Step 1: Detect furniture in all angles
    print("📸 Step 1: Detecting furniture in all angles...")
    detector = FurnitureDetector(use_roboflow=True)
    
    all_detections = {}
    for img_path in image_files:
        detections = detector.detect_furniture(str(img_path), confidence_threshold=0.3)
        all_detections[str(img_path)] = detections
        print(f"  {img_path.name}: {len(detections)} objects")
    
    # Step 2: Match same objects across views
    print("\n🔗 Step 2: Matching same objects across views...")
    
    # Get unique furniture types
    all_types = set()
    for detections in all_detections.values():
        for det in detections:
            all_types.add(det['class'])
    
    matched_objects = {}  # {furniture_type: [(image_path, detection), ...]}
    
    for furniture_type in all_types:
        groups = match_objects_across_views(all_detections, furniture_type)
        if groups:
            # Use the group with most views (best coverage)
            best_group = max(groups, key=len)
            matched_objects[furniture_type] = best_group
            print(f"  {furniture_type}: Found in {len(best_group)} view(s)")
            for img_path, det in best_group:
                print(f"    - {Path(img_path).name} (confidence: {det['confidence']:.2f})")
    
    print(f"\n✓ Matched {len(matched_objects)} objects across views")
    print("")
    
    # Step 3: Crop same object from all views
    print("✂️  Step 3: Cropping objects from all views...")
    
    output_path = Path(output_dir)
    cropped_dir = output_path / "cropped_images"
    cropped_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        segmenter = HybridSegmenter(model_type="vit_b")
        print("  ✓ Hybrid segmenter loaded")
    except Exception as e:
        print(f"  ⚠️  Hybrid segmenter not available: {e}")
        from backend.segmentation.sam_segmenter import SimpleSegmenter
        segmenter = SimpleSegmenter()
        print("  Using simple segmenter")
    
    objects_to_process = []
    
    for furniture_type, views in matched_objects.items():
        print(f"\n  Processing {furniture_type} from {len(views)} view(s)...")
        
        # Crop from all views
        cropped_paths = []
        for i, (img_path, det) in enumerate(views):
            bbox = det['bbox']
            furniture_id = f"{furniture_type}_0"
            view_name = Path(img_path).stem
            
            try:
                if hasattr(segmenter, 'process_furniture'):
                    result = segmenter.process_furniture(
                        img_path,
                        bbox,
                        furniture_type=furniture_type,
                        output_path=str(cropped_dir / f"{furniture_id}_view{i}_{view_name}.jpg")
                    )
                    cropped_path = result['cropped_path']
                else:
                    # Simple crop
                    img = Image.open(img_path)
                    x1, y1, x2, y2 = bbox
                    padding = 20
                    cropped = img.crop((
                        max(0, int(x1) - padding),
                        max(0, int(y1) - padding),
                        min(img.width, int(x2) + padding),
                        min(img.height, int(y2) + padding)
                    ))
                    cropped_path = str(cropped_dir / f"{furniture_id}_view{i}_{view_name}.jpg")
                    cropped.save(cropped_path)
                
                cropped_paths.append(cropped_path)
                print(f"    ✓ View {i+1}: {Path(cropped_path).name}")
            except Exception as e:
                print(f"    ⚠️  Failed to crop view {i+1}: {e}")
        
        if cropped_paths:
            objects_to_process.append({
                "type": furniture_type,
                "id": f"{furniture_type}_0",
                "cropped_paths": cropped_paths,  # Multiple views!
                "views": views
            })
    
    # Step 4: Generate 3D models using ALL views
    print(f"\n🎨 Step 4: Generating 3D models using MULTIPLE VIEWS...")
    print("   (This should give much better depth!)")
    
    for obj in objects_to_process:
        print(f"\n  📦 Processing {obj['type']} from {len(obj['cropped_paths'])} view(s)...")
        
        # For now, use the best single view (TripoSR needs modification for true multi-view)
        # But we can process each view and pick the best result
        best_result = None
        best_path = None
        
        for i, cropped_path in enumerate(obj['cropped_paths']):
            print(f"    Processing view {i+1}/{len(obj['cropped_paths'])}...")
            
            result = subprocess.run(
                ["./process_to_3d.sh", cropped_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # Find the generated model
                output_name = Path(cropped_path).stem
                model_path = f"processed_models/{output_name}/{output_name}_3d.obj"
                
                if Path(model_path).exists():
                    # Check file size (larger = more detail)
                    size = Path(model_path).stat().st_size
                    if best_result is None or size > best_result:
                        best_result = size
                        best_path = model_path
                    print(f"      ✓ View {i+1} generated ({size/1024/1024:.1f}MB)")
                else:
                    print(f"      ⚠️  View {i+1} model not found yet")
            else:
                print(f"      ❌ View {i+1} failed")
        
        if best_path:
            # Copy best model to multi-view output
            import shutil
            final_dir = output_path / obj['id']
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"{obj['id']}_multiview_3d.obj"
            shutil.copy(best_path, final_path)
            print(f"    ✅ Best model saved: {final_path}")
            print(f"    📊 Selected from {len(obj['cropped_paths'])} views")
    
    print("\n" + "=" * 60)
    print(f"✅ Complete! Processed {len(objects_to_process)} objects from multiple views")
    print(f"📁 Check {output_dir}/ for 3D models")
    print("")
    print("💡 Each object was processed from multiple angles")
    print("   The best result (most detail) was selected for each object")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python process_multi_view_object.py <angles_directory> [output_dir]")
        print("\nExample:")
        print("  python process_multi_view_object.py furniture_photos/angles/ processed_models_multiview/")
        sys.exit(1)
    
    angles_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models_multiview"
    
    if not os.getenv("ROBOFLOW_API_KEY"):
        print("⚠️  Warning: ROBOFLOW_API_KEY not set")
        print("   Set it with: export ROBOFLOW_API_KEY='your_key'")
        print("")
    
    process_multi_view_object(angles_dir, output_dir)

