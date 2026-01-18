"""
True Multi-View 3D Reconstruction
Combines multiple camera angles to generate accurate 3D models with proper depth
"""

import subprocess
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import shutil
import tempfile
import cv2
from PIL import Image


class MultiViewReconstructor:
    """
    Multi-view 3D reconstruction using COLMAP + MVS
    Handles camera pose estimation and dense reconstruction
    """
    
    def __init__(self, workspace_dir: str = "multiview_workspace"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(exist_ok=True, parents=True)
    
    def reconstruct_from_images(
        self,
        image_paths: List[str],
        object_name: str,
        output_path: str,
        camera_model: str = "SIMPLE_RADIAL"
    ) -> Dict:
        """
        Reconstruct 3D model from multiple images using COLMAP
        
        Args:
            image_paths: List of image paths (different views of same object)
            object_name: Name for this object
            output_path: Where to save final mesh
            camera_model: COLMAP camera model
            
        Returns:
            Dict with reconstruction results
        """
        print(f"\n🔬 Multi-View Reconstruction: {object_name}")
        print(f"   Using {len(image_paths)} views")
        
        # Create workspace for this object
        obj_workspace = self.workspace / object_name
        obj_workspace.mkdir(exist_ok=True, parents=True)
        
        # Setup directories
        images_dir = obj_workspace / "images"
        sparse_dir = obj_workspace / "sparse"
        dense_dir = obj_workspace / "dense"
        
        images_dir.mkdir(exist_ok=True)
        sparse_dir.mkdir(exist_ok=True)
        dense_dir.mkdir(exist_ok=True)
        
        # Copy images to workspace
        print("   📸 Preparing images...")
        for i, img_path in enumerate(image_paths):
            img = Image.open(img_path)
            # Resize if too large (COLMAP works better with moderate resolution)
            max_size = 1024
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            dest = images_dir / f"view_{i:03d}.jpg"
            img.save(dest, quality=95)
        
        # Step 1: Feature extraction
        print("   🔍 Extracting features...")
        db_path = obj_workspace / "database.db"
        
        result = subprocess.run([
            "colmap", "feature_extractor",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", camera_model,
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.use_gpu", "0"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return self._fallback_reconstruction(image_paths, object_name, output_path)
        
        # Step 2: Feature matching
        print("   🔗 Matching features...")
        result = subprocess.run([
            "colmap", "exhaustive_matcher",
            "--database_path", str(db_path),
            "--SiftMatching.use_gpu", "0"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return self._fallback_reconstruction(image_paths, object_name, output_path)
        
        # Step 3: Sparse reconstruction
        print("   🏗️  Building sparse model...")
        result = subprocess.run([
            "colmap", "mapper",
            "--database_path", str(db_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir)
        ], capture_output=True, text=True)
        
        if result.returncode != 0 or not list(sparse_dir.glob("*/cameras.bin")):
            return self._fallback_reconstruction(image_paths, object_name, output_path)
        
        # Find the reconstruction folder (usually "0")
        recon_dirs = [d for d in sparse_dir.iterdir() if d.is_dir()]
        if not recon_dirs:
            return self._fallback_reconstruction(image_paths, object_name, output_path)
        
        sparse_model = recon_dirs[0]
        
        # Step 4: Dense reconstruction
        print("   🎨 Dense reconstruction...")
        
        # Image undistortion
        result = subprocess.run([
            "colmap", "image_undistorter",
            "--image_path", str(images_dir),
            "--input_path", str(sparse_model),
            "--output_path", str(dense_dir),
            "--output_type", "COLMAP"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            return self._fallback_reconstruction(image_paths, object_name, output_path)
        
        # Patch match stereo
        result = subprocess.run([
            "colmap", "patch_match_stereo",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true"
        ], capture_output=True, text=True)
        
        # Stereo fusion
        print("   🔧 Fusing depth maps...")
        result = subprocess.run([
            "colmap", "stereo_fusion",
            "--workspace_path", str(dense_dir),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(dense_dir / "fused.ply")
        ], capture_output=True, text=True)
        
        fused_ply = dense_dir / "fused.ply"
        
        if fused_ply.exists():
            # Convert to OBJ
            print("   📦 Converting to mesh...")
            output_obj = Path(output_path)
            output_obj.parent.mkdir(exist_ok=True, parents=True)
            
            self._ply_to_obj(str(fused_ply), str(output_obj))
            
            return {
                "status": "success",
                "method": "colmap_mvs",
                "mesh_path": str(output_obj),
                "num_views": len(image_paths),
                "workspace": str(obj_workspace)
            }
        else:
            return self._fallback_reconstruction(image_paths, object_name, output_path)
    
    def _fallback_reconstruction(
        self,
        image_paths: List[str],
        object_name: str,
        output_path: str
    ) -> Dict:
        """
        Fallback: Process each view separately and merge
        """
        print("   ⚠️  COLMAP failed, using fallback: process + merge")
        
        from .mesh_fusion import MeshFusion
        fusion = MeshFusion()
        
        # This will be handled by the hybrid pipeline
        return {
            "status": "fallback",
            "method": "process_and_merge",
            "message": "COLMAP reconstruction failed, will use mesh fusion fallback"
        }
    
    def _ply_to_obj(self, ply_path: str, obj_path: str):
        """Convert PLY point cloud to OBJ mesh"""
        try:
            import open3d as o3d
            
            # Load point cloud
            pcd = o3d.io.read_point_cloud(ply_path)
            
            # Estimate normals
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )
            
            # Poisson surface reconstruction
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=9
            )
            
            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)
            
            # Save
            o3d.io.write_triangle_mesh(obj_path, mesh)
            
        except Exception as e:
            print(f"   ⚠️  Mesh conversion failed: {e}")
            # Just copy the PLY as fallback
            shutil.copy(ply_path, obj_path.replace('.obj', '.ply'))


class MeshFusion:
    """
    Align and merge multiple meshes from different views
    """
    
    def __init__(self):
        pass
    
    def align_and_merge(
        self,
        mesh_paths: List[str],
        output_path: str,
        method: str = "icp"
    ) -> Dict:
        """
        Align multiple meshes and merge them
        
        Args:
            mesh_paths: List of OBJ/PLY mesh files
            output_path: Where to save merged mesh
            method: Alignment method (icp, feature)
            
        Returns:
            Dict with merge results
        """
        print(f"\n🔗 Mesh Fusion: Aligning {len(mesh_paths)} meshes")
        
        try:
            import open3d as o3d
            
            # Load all meshes
            meshes = []
            for path in mesh_paths:
                mesh = o3d.io.read_triangle_mesh(path)
                if not mesh.has_vertices():
                    print(f"   ⚠️  Skipping empty mesh: {path}")
                    continue
                meshes.append(mesh)
            
            if len(meshes) == 0:
                return {"status": "error", "message": "No valid meshes to merge"}
            
            if len(meshes) == 1:
                # Only one mesh, just copy it
                o3d.io.write_triangle_mesh(output_path, meshes[0])
                return {
                    "status": "success",
                    "method": "single_mesh",
                    "mesh_path": output_path
                }
            
            # Use first mesh as reference
            reference = meshes[0]
            aligned_meshes = [reference]
            
            # Align each mesh to reference
            for i, mesh in enumerate(meshes[1:], 1):
                print(f"   Aligning mesh {i+1}/{len(meshes)}...")
                
                # Convert to point clouds for ICP
                ref_pcd = reference.sample_points_uniformly(number_of_points=10000)
                target_pcd = mesh.sample_points_uniformly(number_of_points=10000)
                
                # ICP alignment
                threshold = 0.02  # 2cm threshold
                trans_init = np.eye(4)
                
                reg = o3d.pipelines.registration.registration_icp(
                    target_pcd, ref_pcd, threshold, trans_init,
                    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
                )
                
                # Apply transformation
                mesh.transform(reg.transformation)
                aligned_meshes.append(mesh)
                
                print(f"      Fitness: {reg.fitness:.3f}, RMSE: {reg.inlier_rmse:.4f}")
            
            # Merge all aligned meshes
            print("   🔧 Merging meshes...")
            merged = aligned_meshes[0]
            for mesh in aligned_meshes[1:]:
                merged += mesh
            
            # Clean up
            merged.remove_duplicated_vertices()
            merged.remove_duplicated_triangles()
            merged.remove_degenerate_triangles()
            
            # Save
            Path(output_path).parent.mkdir(exist_ok=True, parents=True)
            o3d.io.write_triangle_mesh(output_path, merged)
            
            print(f"   ✅ Merged mesh saved: {output_path}")
            
            return {
                "status": "success",
                "method": "icp_fusion",
                "mesh_path": output_path,
                "num_meshes": len(aligned_meshes),
                "vertices": len(merged.vertices),
                "triangles": len(merged.triangles)
            }
            
        except Exception as e:
            print(f"   ❌ Mesh fusion failed: {e}")
            # Fallback: just use the largest mesh
            return self._select_best_mesh(mesh_paths, output_path)
    
    def _select_best_mesh(self, mesh_paths: List[str], output_path: str) -> Dict:
        """Fallback: Select the best single mesh by size"""
        print("   Using fallback: selecting best single mesh")
        
        best_path = None
        best_size = 0
        
        for path in mesh_paths:
            size = Path(path).stat().st_size
            if size > best_size:
                best_size = size
                best_path = path
        
        if best_path:
            shutil.copy(best_path, output_path)
            return {
                "status": "success",
                "method": "best_single",
                "mesh_path": output_path,
                "source": best_path
            }
        
        return {"status": "error", "message": "No valid meshes found"}


class DepthEstimator:
    """
    Estimate depth maps from images to improve 3D reconstruction
    """
    
    def __init__(self, model_type: str = "DPT_Large"):
        """
        Args:
            model_type: MiDaS model type (DPT_Large, DPT_Hybrid, MiDaS_small)
        """
        self.model_type = model_type
        self.model = None
        self.transform = None
        self.device = None
    
    def load_model(self):
        """Load MiDaS depth estimation model"""
        try:
            import torch
            
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Load MiDaS
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.model.to(self.device)
            self.model.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            
            if self.model_type in ["DPT_Large", "DPT_Hybrid"]:
                self.transform = midas_transforms.dpt_transform
            else:
                self.transform = midas_transforms.small_transform
            
            print(f"✓ MiDaS {self.model_type} loaded on {self.device}")
            
        except Exception as e:
            print(f"⚠️  Could not load MiDaS: {e}")
            self.model = None
    
    def estimate_depth(self, image_path: str, output_path: Optional[str] = None) -> np.ndarray:
        """
        Estimate depth map from image
        
        Args:
            image_path: Input image
            output_path: Optional path to save depth map visualization
            
        Returns:
            Depth map as numpy array
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            raise RuntimeError("Depth estimation model not available")
        
        import torch
        
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Transform
        input_batch = self.transform(img).to(self.device)
        
        # Predict
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # Save visualization if requested
        if output_path:
            depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
            depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
            cv2.imwrite(output_path, depth_colored)
        
        return depth_map
    
    def enhance_image_with_depth(
        self,
        image_path: str,
        output_path: str,
        depth_weight: float = 0.3
    ) -> str:
        """
        Create enhanced image with depth information for better 3D generation
        
        Args:
            image_path: Input image
            output_path: Output enhanced image
            depth_weight: How much to blend depth (0-1)
            
        Returns:
            Path to enhanced image
        """
        depth_map = self.estimate_depth(image_path)
        
        # Load original image
        img = cv2.imread(image_path)
        
        # Normalize depth to 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_BONE)
        
        # Blend with original
        enhanced = cv2.addWeighted(img, 1 - depth_weight, depth_colored, depth_weight, 0)
        
        # Save
        Path(output_path).parent.mkdir(exist_ok=True, parents=True)
        cv2.imwrite(output_path, enhanced)
        
        return output_path


if __name__ == "__main__":
    # Test multi-view reconstruction
    reconstructor = MultiViewReconstructor()
    
    # Example: reconstruct bed from multiple angles
    image_paths = [
        "furniture_photos/angles/a1.png",
        "furniture_photos/angles/a2.png",
        "furniture_photos/angles/a3.png"
    ]
    
    result = reconstructor.reconstruct_from_images(
        image_paths=image_paths,
        object_name="bed_multiview",
        output_path="processed_models/bed_multiview/bed_3d.obj"
    )
    
    print(f"\nResult: {result}")

