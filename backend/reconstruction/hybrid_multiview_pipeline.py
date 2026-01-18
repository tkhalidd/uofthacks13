"""
Hybrid Multi-View Pipeline
Combines the best of both approaches:
1. Try COLMAP multi-view reconstruction (best quality)
2. Fallback to TripoSR per-view + mesh fusion (more robust)
3. Use depth estimation to improve single-view results
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional
import subprocess
import shutil

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.reconstruction.multiview_reconstructor import (
    MultiViewReconstructor,
    MeshFusion,
    DepthEstimator
)


class HybridMultiViewPipeline:
    """
    Intelligent multi-view pipeline that chooses the best method
    """
    
    def __init__(
        self,
        workspace_dir: str = "multiview_workspace",
        use_depth_enhancement: bool = True
    ):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True, parents=True)
        
        self.multiview = MultiViewReconstructor(str(self.workspace / "colmap"))
        self.fusion = MeshFusion()
        
        self.depth_estimator = None
        if use_depth_enhancement:
            try:
                self.depth_estimator = DepthEstimator()
                print("✓ Depth estimation enabled")
            except Exception as e:
                print(f"⚠️  Depth estimation disabled: {e}")
    
    def process_object_multiview(
        self,
        image_paths: List[str],
        object_name: str,
        object_type: str,
        output_dir: str,
        method: str = "auto"
    ) -> Dict:
        """
        Process object from multiple views using best available method
        
        Args:
            image_paths: List of cropped images (different views of same object)
            object_name: Unique name for this object (e.g., "bed_0")
            object_type: Type of furniture (e.g., "bed")
            output_dir: Where to save final model
            method: "auto", "colmap", "triposr_fusion", or "triposr_best"
            
        Returns:
            Dict with processing results
        """
        print(f"\n{'='*60}")
        print(f"🎨 Processing: {object_name} ({object_type})")
        print(f"   Views: {len(image_paths)}")
        print(f"   Method: {method}")
        print(f"{'='*60}")
        
        output_path = Path(output_dir) / object_name / f"{object_name}_3d.obj"
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Auto-select method based on number of views
        if method == "auto":
            if len(image_paths) >= 3:
                method = "colmap"  # Try COLMAP first for 3+ views
            else:
                method = "triposr_best"  # Not enough views for COLMAP
        
        result = None
        
        # Method 1: COLMAP Multi-View Stereo (best quality)
        if method == "colmap" and len(image_paths) >= 3:
            print("\n🔬 Attempting COLMAP multi-view reconstruction...")
            try:
                result = self.multiview.reconstruct_from_images(
                    image_paths=image_paths,
                    object_name=object_name,
                    output_path=str(output_path)
                )
                
                if result.get("status") == "success":
                    print(f"✅ COLMAP reconstruction successful!")
                    return result
                else:
                    print("⚠️  COLMAP failed, trying fallback...")
            except Exception as e:
                print(f"⚠️  COLMAP error: {e}")
        
        # Method 2: TripoSR + Mesh Fusion (robust fallback)
        if method in ["auto", "colmap", "triposr_fusion"]:
            print("\n🔧 Using TripoSR + Mesh Fusion...")
            result = self._triposr_with_fusion(
                image_paths=image_paths,
                object_name=object_name,
                object_type=object_type,
                output_path=str(output_path)
            )
            
            if result.get("status") == "success":
                return result
        
        # Method 3: TripoSR Best Single View (last resort)
        print("\n📸 Using best single view from TripoSR...")
        result = self._triposr_best_view(
            image_paths=image_paths,
            object_name=object_name,
            output_path=str(output_path)
        )
        
        return result
    
    def _triposr_with_fusion(
        self,
        image_paths: List[str],
        object_name: str,
        object_type: str,
        output_path: str
    ) -> Dict:
        """
        Process each view with TripoSR, then align and merge meshes
        """
        print(f"   Processing {len(image_paths)} views with TripoSR...")
        
        # Enhance images with depth if available
        enhanced_paths = []
        if self.depth_estimator:
            print("   📊 Enhancing images with depth estimation...")
            enhanced_dir = self.workspace / "enhanced" / object_name
            enhanced_dir.mkdir(exist_ok=True, parents=True)
            
            for i, img_path in enumerate(image_paths):
                try:
                    enhanced_path = enhanced_dir / f"view_{i}_enhanced.jpg"
                    self.depth_estimator.enhance_image_with_depth(
                        img_path,
                        str(enhanced_path),
                        depth_weight=0.2  # Subtle enhancement
                    )
                    enhanced_paths.append(str(enhanced_path))
                    print(f"      ✓ View {i+1} enhanced")
                except Exception as e:
                    print(f"      ⚠️  View {i+1} enhancement failed: {e}")
                    enhanced_paths.append(img_path)  # Use original
        else:
            enhanced_paths = image_paths
        
        # Process each view with TripoSR
        mesh_paths = []
        temp_models_dir = self.workspace / "temp_models" / object_name
        temp_models_dir.mkdir(exist_ok=True, parents=True)
        
        for i, img_path in enumerate(enhanced_paths):
            print(f"\n   View {i+1}/{len(enhanced_paths)}:")
            
            # Use process_to_3d.sh to send to RunPod
            result = subprocess.run(
                ["./process_to_3d.sh", img_path],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            
            if result.returncode == 0:
                # Find generated model
                img_name = Path(img_path).stem
                model_path = Path(f"processed_models/{img_name}/{img_name}_3d.obj")
                
                # Wait for file
                import time
                time.sleep(2)
                
                if model_path.exists():
                    # Copy to temp location
                    temp_path = temp_models_dir / f"view_{i}.obj"
                    shutil.copy(model_path, temp_path)
                    mesh_paths.append(str(temp_path))
                    
                    size = model_path.stat().st_size / 1024 / 1024
                    print(f"      ✅ Generated ({size:.1f}MB)")
                else:
                    print(f"      ⚠️  Model file not found")
            else:
                print(f"      ❌ Processing failed")
        
        if len(mesh_paths) == 0:
            return {
                "status": "error",
                "message": "No meshes generated from any view"
            }
        
        # Align and merge meshes
        print(f"\n   🔗 Aligning and merging {len(mesh_paths)} meshes...")
        result = self.fusion.align_and_merge(
            mesh_paths=mesh_paths,
            output_path=output_path
        )
        
        if result.get("status") == "success":
            print(f"   ✅ Multi-view fusion complete!")
            result["method"] = "triposr_fusion"
            result["num_views"] = len(image_paths)
            result["num_meshes_merged"] = len(mesh_paths)
        
        return result
    
    def _triposr_best_view(
        self,
        image_paths: List[str],
        object_name: str,
        output_path: str
    ) -> Dict:
        """
        Process all views and select the best result (fallback method)
        """
        print(f"   Processing {len(image_paths)} views, will select best...")
        
        best_size = 0
        best_path = None
        best_view_idx = None
        
        for i, img_path in enumerate(image_paths):
            print(f"\n   View {i+1}/{len(image_paths)}:")
            
            result = subprocess.run(
                ["./process_to_3d.sh", img_path],
                capture_output=True,
                text=True,
                cwd=str(Path(__file__).parent.parent.parent)
            )
            
            if result.returncode == 0:
                img_name = Path(img_path).stem
                model_path = Path(f"processed_models/{img_name}/{img_name}_3d.obj")
                
                import time
                time.sleep(2)
                
                if model_path.exists():
                    size = model_path.stat().st_size
                    size_mb = size / 1024 / 1024
                    print(f"      ✅ Generated ({size_mb:.1f}MB)")
                    
                    if size > best_size:
                        best_size = size
                        best_path = model_path
                        best_view_idx = i
                else:
                    print(f"      ⚠️  Model not found")
            else:
                print(f"      ❌ Failed")
        
        if best_path and best_path.exists():
            # Copy best result to output
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(best_path, output_path)
            
            print(f"\n   ✅ Best result: View {best_view_idx+1} ({best_size/1024/1024:.1f}MB)")
            
            return {
                "status": "success",
                "method": "triposr_best_single",
                "mesh_path": output_path,
                "best_view": best_view_idx,
                "num_views_tried": len(image_paths)
            }
        else:
            return {
                "status": "error",
                "message": "No valid models generated from any view"
            }


def test_pipeline():
    """Test the hybrid pipeline"""
    pipeline = HybridMultiViewPipeline()
    
    # Test with multiple views of a bed
    image_paths = [
        "furniture_photos/angles/a1.png",
        "furniture_photos/angles/a2.png",
        "furniture_photos/angles/a3.png"
    ]
    
    result = pipeline.process_object_multiview(
        image_paths=image_paths,
        object_name="bed_test",
        object_type="bed",
        output_dir="processed_models_hybrid",
        method="auto"
    )
    
    print(f"\n{'='*60}")
    print("Final Result:")
    print(f"  Status: {result.get('status')}")
    print(f"  Method: {result.get('method')}")
    if result.get('mesh_path'):
        print(f"  Output: {result['mesh_path']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    test_pipeline()

