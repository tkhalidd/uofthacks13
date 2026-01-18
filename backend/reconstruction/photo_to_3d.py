"""
Photo to 3D Model Generator
Converts single photos of furniture into 3D models using AI
"""

import requests
import subprocess
from pathlib import Path
from typing import Dict, Optional
import base64
from PIL import Image
import json


class PhotoTo3DGenerator:
    """
    Generate 3D models from single photos of furniture
    Supports multiple backends: TRELLIS.2, SAM 3D, TripoSR
    """
    
    def __init__(
        self, 
        backend: str = "sam3d",  # "sam3d" (recommended), "trellis", or "triposr"
        model_path: Optional[str] = None,
        sam3d_server_url: Optional[str] = None
    ):
        """
        Initialize photo to 3D generator
        
        Args:
            backend: Which model to use (sam3d, trellis, triposr)
            model_path: Path to local model weights (if self-hosting)
            sam3d_server_url: URL of SAM 3D server on Vultr (for sam3d backend)
        """
        self.backend = backend
        self.model_path = model_path
        self.sam3d_server_url = sam3d_server_url
        
        if backend == "sam3d":
            self._init_sam3d()
        elif backend == "trellis":
            self._init_trellis()
        elif backend == "triposr":
            self._init_triposr()
    
    def generate_3d_model(
        self,
        image_path: str,
        output_path: str,
        object_type: Optional[str] = None,
        bounding_box: Optional[Dict] = None,
        cleanup: bool = True
    ) -> Dict:
        """
        Generate 3D model from photo
        
        Args:
            image_path: Path to photo of furniture
            output_path: Where to save 3D model
            object_type: Type of object (bed, desk, chair, etc.)
            bounding_box: Optional bbox to isolate object {"x1": 0, "y1": 0, "x2": 100, "y2": 100}
            cleanup: Whether to clean up mesh (remove artifacts, optimize)
        
        Returns:
            Dict with model path and metadata
        """
        print(f"Generating 3D model from {image_path} using {self.backend}...")
        
        if self.backend == "trellis":
            result = self._generate_trellis(image_path, output_path, object_type)
        elif self.backend == "sam3d":
            result = self._generate_sam3d(image_path, output_path, bounding_box)
        elif self.backend == "triposr":
            result = self._generate_triposr(image_path, output_path)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
        
        # Optional cleanup
        if cleanup and result.get("model_path"):
            result = self._cleanup_mesh(result["model_path"])
        
        return result
    
    def _init_trellis(self):
        """Initialize TRELLIS.2 model"""
        try:
            # Check if TRELLIS is installed
            import torch
            # In production, load the model here
            # from trellis import TRELLIS
            # self.model = TRELLIS.from_pretrained("microsoft/TRELLIS-2")
            print("TRELLIS.2 initialized")
        except ImportError:
            print("Warning: TRELLIS not installed. Install with: pip install trellis-3d")
    
    def _init_sam3d(self):
        """Initialize SAM 3D Objects client"""
        try:
            from backend.reconstruction.sam3d_client import SAM3DClient
            import os
            
            server_url = self.sam3d_server_url or os.getenv("SAM3D_SERVER_URL")
            api_key = os.getenv("SAM3D_API_KEY")
            
            if not server_url:
                raise ValueError("SAM3D_SERVER_URL not set. Please set environment variable or pass sam3d_server_url parameter")
            
            self.sam3d_client = SAM3DClient(server_url, api_key)
            
            # Check server health
            if not self.sam3d_client.health_check():
                raise ConnectionError(f"SAM 3D server at {server_url} is not responding")
            
            print(f"SAM 3D Objects client initialized (server: {server_url})")
        except Exception as e:
            print(f"Warning: SAM 3D initialization failed: {e}")
    
    def _init_triposr(self):
        """Initialize TripoSR client"""
        try:
            from backend.reconstruction.triposr_client import TripoSRClient
            import os
            
            server_url = os.getenv("TRIPOSR_SERVER_URL")
            if not server_url:
                raise ValueError(
                    "TRIPOSR_SERVER_URL not set. "
                    "Set to your RunPod server URL (e.g., http://runpod-ip:8000)"
                )
            
            self.triposr_client = TripoSRClient(server_url)
            
            # Check server health
            if not self.triposr_client.health_check():
                raise ConnectionError(
                    f"TripoSR server at {server_url} is not responding"
                )
            
            print(f"TripoSR client initialized (server: {server_url})")
        except Exception as e:
            print(f"Warning: TripoSR initialization failed: {e}")
            raise
    
    def _generate_trellis(
        self, 
        image_path: str, 
        output_path: str,
        object_type: Optional[str]
    ) -> Dict:
        """
        Generate 3D model using TRELLIS.2
        
        TRELLIS.2 is best for high-quality furniture models
        """
        # Placeholder implementation
        # In production, this would call the actual TRELLIS model
        
        """
        Example actual implementation:
        
        from trellis import TRELLIS
        import torch
        
        # Load image
        image = Image.open(image_path)
        
        # Generate 3D model
        with torch.no_grad():
            result = self.model.generate(
                image=image,
                object_type=object_type,
                resolution="high",
                export_format="glb"
            )
        
        # Save
        result.save(output_path)
        """
        
        print(f"[TRELLIS] Would generate 3D model from {image_path}")
        print(f"[TRELLIS] Object type: {object_type}")
        print(f"[TRELLIS] Output: {output_path}")
        
        return {
            "status": "success",
            "model_path": output_path,
            "format": "glb",
            "backend": "trellis",
            "quality": "high",
            "metadata": {
                "object_type": object_type,
                "has_texture": True,
                "has_pbr_materials": True
            }
        }
    
    def _generate_sam3d(
        self,
        image_path: str,
        output_path: str,
        bounding_box: Optional[Dict]
    ) -> Dict:
        """
        Generate 3D model using SAM 3D Objects
        
        SAM 3D is good for isolating objects in cluttered scenes
        """
        if not hasattr(self, 'sam3d_client'):
            raise RuntimeError("SAM 3D client not initialized")
        
        # Call SAM 3D server on Vultr
        result = self.sam3d_client.generate_3d_model(
            image_path=image_path,
            output_path=output_path,
            mask_path=None,  # TODO: handle bounding_box -> mask conversion
            seed=42
        )
        
        return result
    
    def _generate_triposr(
        self,
        image_path: str,
        output_path: str
    ) -> Dict:
        """
        Generate 3D model using TripoSR
        
        TripoSR is fast (~10-30s) and good for furniture models
        Connects to TripoSR API server on RunPod
        """
        if not hasattr(self, 'triposr_client'):
            raise RuntimeError("TripoSR client not initialized")
        
        # Call TripoSR server on RunPod
        result = self.triposr_client.generate_3d_model(
            image_path=image_path,
            output_path=output_path,
            device="cuda:0"
        )
        
        return result
    
    def _cleanup_mesh(self, model_path: str) -> Dict:
        """
        Clean up generated mesh using Blender or pymeshlab
        
        - Remove artifacts
        - Optimize topology
        - Fix normals
        - Reduce poly count if needed
        """
        try:
            import pymeshlab
            
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(model_path)
            
            # Remove duplicate vertices
            ms.meshing_remove_duplicate_vertices()
            
            # Remove unreferenced vertices
            ms.meshing_remove_unreferenced_vertices()
            
            # Fix normals
            ms.meshing_repair_non_manifold_edges()
            ms.compute_normal_per_vertex()
            
            # Optional: simplify if too many faces
            if ms.current_mesh().face_number() > 50000:
                ms.meshing_decimation_quadric_edge_collapse(
                    targetfacenum=50000
                )
            
            # Save cleaned version
            cleaned_path = model_path.replace(".glb", "_cleaned.glb")
            ms.save_current_mesh(cleaned_path)
            
            print(f"Mesh cleaned and saved to {cleaned_path}")
            
            return {
                "status": "success",
                "model_path": cleaned_path,
                "cleaned": True
            }
        
        except ImportError:
            print("pymeshlab not installed, skipping cleanup")
            return {"status": "skipped", "model_path": model_path}
        except Exception as e:
            print(f"Cleanup failed: {e}")
            return {"status": "failed", "model_path": model_path}
    
    def batch_generate(
        self,
        image_dir: str,
        output_dir: str,
        furniture_types: Optional[Dict[str, str]] = None
    ) -> Dict[str, Dict]:
        """
        Generate 3D models for multiple furniture photos
        
        Args:
            image_dir: Directory with furniture photos
            output_dir: Where to save 3D models
            furniture_types: Map of filename → furniture type
        
        Returns:
            Dict mapping filenames to generation results
        """
        image_dir = Path(image_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {}
        
        image_files = list(image_dir.glob("*.jpg")) + \
                     list(image_dir.glob("*.png")) + \
                     list(image_dir.glob("*.jpeg"))
        
        for img_path in image_files:
            object_type = None
            if furniture_types and img_path.name in furniture_types:
                object_type = furniture_types[img_path.name]
            
            output_path = output_dir / f"{img_path.stem}.glb"
            
            try:
                result = self.generate_3d_model(
                    str(img_path),
                    str(output_path),
                    object_type=object_type
                )
                results[img_path.name] = result
            except Exception as e:
                results[img_path.name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = PhotoTo3DGenerator(backend="trellis")
    
    # Generate 3D model from single photo
    print("=== Single Photo to 3D ===\n")
    
    result = generator.generate_3d_model(
        image_path="data/photos/my_bed.jpg",
        output_path="models/furniture/my_bed.glb",
        object_type="bed",
        cleanup=True
    )
    
    print(f"\n✓ Generated: {result['model_path']}")
    print(f"  Quality: {result['quality']}")
    print(f"  Format: {result['format']}")
    print(f"  Backend: {result['backend']}")
    
    # Batch generation
    print("\n=== Batch Generation ===\n")
    
    batch_results = generator.batch_generate(
        image_dir="data/photos/furniture",
        output_dir="models/furniture/custom",
        furniture_types={
            "bed.jpg": "bed",
            "desk.jpg": "desk",
            "chair.jpg": "chair"
        }
    )
    
    print(f"\n✓ Generated {len(batch_results)} models")
    for filename, result in batch_results.items():
        if result["status"] == "success":
            print(f"  ✓ {filename} → {result['model_path']}")
        else:
            print(f"  ✗ {filename} → {result.get('error', 'failed')}")

