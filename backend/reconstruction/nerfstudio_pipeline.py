"""
Nerfstudio Pipeline Wrapper
Handles video processing, training, and export for room reconstruction
"""

import subprocess
import os
from pathlib import Path
from typing import Optional, Dict
import json


class NerfstudioPipeline:
    """
    Wrapper for Nerfstudio commands to reconstruct rooms from video/images
    """
    
    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def process_video(
        self, 
        video_path: str, 
        output_name: str,
        camera_type: str = "perspective",
        matching_method: str = "exhaustive"
    ) -> Dict:
        """
        Process video using ns-process-data (includes COLMAP)
        
        Args:
            video_path: Path to input video
            output_name: Name for processed dataset
            camera_type: Camera type (perspective, fisheye, equirectangular)
            matching_method: COLMAP matching method (exhaustive, sequential, vocab_tree)
        
        Returns:
            Dict with processed data path and metadata
        """
        output_path = self.data_dir / output_name
        
        cmd = [
            "ns-process-data", "video",
            "--data", video_path,
            "--output-dir", str(output_path),
            "--camera-type", camera_type,
            "--matching-method", matching_method,
            "--num-downscales", "3"  # Create multiple resolution versions
        ]
        
        print(f"Processing video with COLMAP...")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Video processing failed: {result.stderr}")
        
        return {
            "data_path": str(output_path),
            "status": "success",
            "transforms_path": str(output_path / "transforms.json")
        }
    
    def process_images(
        self,
        images_dir: str,
        output_name: str,
        camera_type: str = "perspective"
    ) -> Dict:
        """
        Process directory of images using ns-process-data
        
        Args:
            images_dir: Directory containing images
            output_name: Name for processed dataset
            camera_type: Camera type
        
        Returns:
            Dict with processed data path and metadata
        """
        output_path = self.data_dir / output_name
        
        cmd = [
            "ns-process-data", "images",
            "--data", images_dir,
            "--output-dir", str(output_path),
            "--camera-type", camera_type
        ]
        
        print(f"Processing images with COLMAP...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Image processing failed: {result.stderr}")
        
        return {
            "data_path": str(output_path),
            "status": "success"
        }
    
    def train_model(
        self,
        data_path: str,
        model_name: str = "splatfacto",  # Gaussian Splatting - fast & high quality
        experiment_name: Optional[str] = None,
        max_num_iterations: int = 30000,
        viewer_enabled: bool = True
    ) -> Dict:
        """
        Train a NeRF/Gaussian Splatting model
        
        Args:
            data_path: Path to processed data
            model_name: Model to use (splatfacto, nerfacto, instant-ngp)
            experiment_name: Name for this training run
            max_num_iterations: Training iterations
            viewer_enabled: Enable web viewer during training
        
        Returns:
            Dict with model path and config
        """
        cmd = [
            "ns-train", model_name,
            "--data", data_path,
            "--max-num-iterations", str(max_num_iterations)
        ]
        
        if experiment_name:
            cmd.extend(["--experiment-name", experiment_name])
        
        if not viewer_enabled:
            cmd.append("--vis", "tensorboard")
        
        print(f"Training {model_name} model...")
        print(f"Command: {' '.join(cmd)}")
        print(f"Training will take several minutes. Viewer available at http://localhost:7007")
        
        # Note: This will block until training completes
        # In production, you'd want to run this async or in a separate process
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed: {result.stderr}")
        
        # Parse output to find config path
        # Format: "Saving config to: outputs/.../config.yml"
        config_path = self._extract_config_path(result.stdout)
        
        return {
            "status": "success",
            "config_path": config_path,
            "model_name": model_name
        }
    
    def export_mesh(
        self,
        config_path: str,
        output_path: str,
        method: str = "poisson",  # or "marching-cubes"
        texture_resolution: int = 2048
    ) -> Dict:
        """
        Export trained model as a mesh
        
        Args:
            config_path: Path to model config.yml
            output_path: Where to save the mesh
            method: Export method (poisson, marching-cubes)
            texture_resolution: Resolution of texture maps
        
        Returns:
            Dict with mesh path
        """
        cmd = [
            "ns-export", method,
            "--load-config", config_path,
            "--output-dir", output_path,
            "--texture-resolution", str(texture_resolution),
            "--num-faces", "100000"  # Target face count
        ]
        
        print(f"Exporting mesh using {method}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Mesh export failed: {result.stderr}")
        
        # Find exported mesh file
        mesh_files = list(Path(output_path).glob("*.ply")) + \
                    list(Path(output_path).glob("*.obj"))
        
        if not mesh_files:
            raise RuntimeError("No mesh file found after export")
        
        return {
            "status": "success",
            "mesh_path": str(mesh_files[0]),
            "format": mesh_files[0].suffix
        }
    
    def export_pointcloud(
        self,
        config_path: str,
        output_path: str,
        num_points: int = 1000000
    ) -> Dict:
        """
        Export trained model as a point cloud
        
        Args:
            config_path: Path to model config.yml
            output_path: Where to save point cloud
            num_points: Number of points to generate
        
        Returns:
            Dict with point cloud path
        """
        cmd = [
            "ns-export", "pointcloud",
            "--load-config", config_path,
            "--output-dir", output_path,
            "--num-points", str(num_points),
            "--remove-outliers", "True"
        ]
        
        print(f"Exporting point cloud...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"Point cloud export failed: {result.stderr}")
        
        ply_files = list(Path(output_path).glob("*.ply"))
        
        if not ply_files:
            raise RuntimeError("No point cloud file found after export")
        
        return {
            "status": "success",
            "pointcloud_path": str(ply_files[0])
        }
    
    def _extract_config_path(self, stdout: str) -> str:
        """Extract config path from training output"""
        for line in stdout.split('\n'):
            if 'config' in line.lower() and '.yml' in line:
                # Parse the path
                parts = line.split()
                for part in parts:
                    if part.endswith('.yml'):
                        return part
        raise RuntimeError("Could not find config path in training output")


# Example usage
if __name__ == "__main__":
    pipeline = NerfstudioPipeline()
    
    # Example workflow
    print("=== Room Reconstruction Pipeline ===\n")
    
    # Step 1: Process video
    print("Step 1: Processing video...")
    processed = pipeline.process_video(
        video_path="data/uploads/room_video.mp4",
        output_name="room_scan_001"
    )
    print(f"✓ Data processed: {processed['data_path']}\n")
    
    # Step 2: Train model
    print("Step 2: Training Gaussian Splatting model...")
    trained = pipeline.train_model(
        data_path=processed['data_path'],
        model_name="splatfacto",
        experiment_name="room_001",
        max_num_iterations=30000
    )
    print(f"✓ Model trained: {trained['config_path']}\n")
    
    # Step 3: Export mesh
    print("Step 3: Exporting mesh...")
    mesh = pipeline.export_mesh(
        config_path=trained['config_path'],
        output_path="data/reconstructions/room_001"
    )
    print(f"✓ Mesh exported: {mesh['mesh_path']}\n")
    
    print("=== Reconstruction Complete ===")
    print("Next steps:")
    print("1. Run furniture segmentation on the mesh")
    print("2. Apply identity-based optimization")
    print("3. Generate before/after visualization")


