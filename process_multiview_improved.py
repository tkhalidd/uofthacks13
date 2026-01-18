#!/usr/bin/env python3
"""
Improved Multi-View Processing with TRUE multi-view reconstruction

This script:
1. Detects furniture in all angles
2. Matches same objects across views
3. Crops same object from all views
4. Uses HYBRID approach for 3D generation:
   - Try COLMAP multi-view reconstruction (best quality)
   - Fallback to TripoSR + mesh fusion (robust)
   - Last resort: TripoSR best single view

This is TRUE multi-view reconstruction, not just picking the best single view!
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess
import shutil
from collections import defaultdict

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.segmentation.hybrid_segmenter import HybridSegmenter
from backend.reconstruction.hybrid_multiview_pipeline import HybridMultiViewPipeline


def match_objects_across_views(all_detections, furniture_type):
    """
    Match same object type across multiple views
    Returns all views where this type appears
    """
    type_detections = [(img, det) for img, dets in all_detections.items() 
                      for det in dets if det['class'] == furniture_type]
    
    # Sort by confidence
    type_detections.sort(key=lambda x: x[1]['confidence'], reverse=True)
    return type_detections


def process_multiview_improved(
    angles_dir,
    output_dir="processed_models_multiview_improved",
    method="auto"
):
    """
    Process room from multiple angles with TRUE multi-view reconstruction
    
    Args:
        angles_dir: Directory with room photos from different angles
        output_dir: Where to save 3D models
        method: "auto", "colmap", "triposr_fusion", or "triposr_best"
    """
    angles_path = Path(angles_dir)
    image_files = sorted(list(angles_path.glob("*.png")) + 
                        list(angles_path.glob("*.jpg")))
    
    print("🏠 IMPROVED Multi-View Processing")
    print("   TRUE multi-view reconstruction with COLMAP + mesh fusion")
    print("=" * 70)
    print(f"Found {len(image_files)} room angles:")
    for img in image_files:
        print(f"  - {img.name}")
    print("")
    
    # Check for required tools
    print("🔧 Checking dependencies...")
    has_colmap = shutil.which("colmap") is not None
    print(f"   COLMAP: {'✓ Available' if has_colmap else '⚠️  Not installed (will use fallback)'}")
    
    try:
        import open3d
        print("   Open3D: ✓ Available")
        has_open3d = True
    except:
        print("   Open3D: ⚠️  Not installed (mesh fusion disabled)")
        has_open3d = False
    
    print("")
    
    # Step 1: Detect furniture in all views
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
        views = match_objects_across_views(all_detections, furniture_type)
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
    except Exception as e:
        print(f"  ⚠️  Hybrid segmenter not available: {e}")
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
                "cropped_paths": cropped_paths,
                "views": views
            })
    
    # Step 4: TRUE Multi-View 3D Reconstruction
    print(f"\n🎨 Step 4: TRUE Multi-View 3D Reconstruction")
    print("   This uses ALL views together, not just picking the best!")
    print("")
    
    # Initialize hybrid pipeline
    pipeline = HybridMultiViewPipeline(
        workspace_dir=str(output_path / "workspace"),
        use_depth_enhancement=True
    )
    
    results = []
    
    for obj in objects_to_process:
        print(f"\n{'='*70}")
        print(f"📦 {obj['type']} ({len(obj['cropped_paths'])} views)")
        print(f"{'='*70}")
        
        # Determine method based on availability
        if method == "auto":
            if has_colmap and has_open3d and len(obj['cropped_paths']) >= 3:
                obj_method = "colmap"
            elif has_open3d and len(obj['cropped_paths']) >= 2:
                obj_method = "triposr_fusion"
            else:
                obj_method = "triposr_best"
        else:
            obj_method = method
        
        print(f"Method: {obj_method}")
        
        # Process with hybrid pipeline
        result = pipeline.process_object_multiview(
            image_paths=obj['cropped_paths'],
            object_name=obj['id'],
            object_type=obj['type'],
            output_dir=str(output_path),
            method=obj_method
        )
        
        results.append({
            "object": obj['id'],
            "type": obj['type'],
            "result": result
        })
        
        if result.get('status') == 'success':
            print(f"\n✅ {obj['type']} complete!")
            print(f"   Method used: {result.get('method')}")
            print(f"   Output: {result.get('mesh_path')}")
        else:
            print(f"\n⚠️  {obj['type']} failed: {result.get('message')}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 PROCESSING SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r['result'].get('status') == 'success']
    failed = [r for r in results if r['result'].get('status') != 'success']
    
    print(f"\n✅ Successful: {len(successful)}/{len(results)}")
    for r in successful:
        method = r['result'].get('method', 'unknown')
        print(f"   - {r['object']}: {method}")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)}/{len(results)}")
        for r in failed:
            print(f"   - {r['object']}: {r['result'].get('message', 'unknown error')}")
    
    print(f"\n📁 Output directory: {output_dir}/")
    print("")
    print("💡 Key improvements over old approach:")
    print("   ✓ COLMAP uses all views together for true 3D reconstruction")
    print("   ✓ Mesh fusion aligns and merges multiple TripoSR results")
    print("   ✓ Depth estimation enhances single-view understanding")
    print("   ✓ Automatic fallback ensures something always works")
    print("")


def main():
    if len(sys.argv) < 2:
        print("Usage: python process_multiview_improved.py <angles_directory> [output_dir] [method]")
        print("\nArguments:")
        print("  angles_directory: Directory with room photos from multiple angles")
        print("  output_dir: Where to save 3D models (default: processed_models_multiview_improved)")
        print("  method: Reconstruction method (default: auto)")
        print("          - auto: Try COLMAP, fallback to fusion, then best single")
        print("          - colmap: Force COLMAP multi-view reconstruction")
        print("          - triposr_fusion: TripoSR per-view + mesh fusion")
        print("          - triposr_best: TripoSR per-view, select best (old method)")
        print("\nExample:")
        print("  python process_multiview_improved.py furniture_photos/angles/")
        print("")
        print("This will:")
        print("  1. Detect furniture in all angles")
        print("  2. Match same objects across views")
        print("  3. Crop same object from all views")
        print("  4. Use TRUE multi-view reconstruction (COLMAP or mesh fusion)")
        print("")
        print("Requirements:")
        print("  - COLMAP installed (optional, for best quality)")
        print("  - Open3D installed (optional, for mesh fusion)")
        print("  - MiDaS for depth (optional, auto-downloaded)")
        print("")
        sys.exit(1)
    
    angles_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "processed_models_multiview_improved"
    method = sys.argv[3] if len(sys.argv) > 3 else "auto"
    
    if not os.getenv("ROBOFLOW_API_KEY"):
        print("⚠️  Warning: ROBOFLOW_API_KEY not set")
        print("   Set it with: export ROBOFLOW_API_KEY='your_key'")
        print("")
    
    if not Path(angles_dir).exists():
        print(f"❌ Error: Directory not found: {angles_dir}")
        sys.exit(1)
    
    process_multiview_improved(angles_dir, output_dir, method)


if __name__ == "__main__":
    main()

