"""
TripoSR Client
Connects to TripoSR API server running on RunPod
"""

import requests
import time
from pathlib import Path
from typing import Dict, Optional
import os


class TripoSRClient:
    """
    Client for TripoSR 3D generation API on RunPod
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize TripoSR client
        
        Args:
            server_url: URL of TripoSR API server (e.g., "http://runpod-ip:8000")
            timeout: Request timeout in seconds
        """
        self.server_url = server_url or os.getenv("TRIPOSR_SERVER_URL")
        self.timeout = timeout
        
        if not self.server_url:
            raise ValueError(
                "TRIPOSR_SERVER_URL not set. "
                "Set environment variable or pass server_url parameter"
            )
        
        # Remove trailing slash
        self.server_url = self.server_url.rstrip('/')
        
        print(f"TripoSR client initialized (server: {self.server_url})")
    
    def health_check(self) -> bool:
        """
        Check if TripoSR server is running
        
        Returns:
            True if server is healthy
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False
    
    def generate_3d_model(
        self,
        image_path: str,
        output_path: str,
        device: str = "cuda:0",
        timeout: Optional[int] = None
    ) -> Dict:
        """
        Generate 3D model from single photo
        
        Args:
            image_path: Path to furniture photo
            output_path: Where to save generated 3D model (.obj file)
            device: Device to use ("cuda:0" or "mps" or "cpu")
            timeout: Request timeout (defaults to self.timeout)
        
        Returns:
            Dict with model path and metadata
        """
        print(f"Sending {image_path} to TripoSR server...")
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Prepare output directory
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare files
        with open(image_path, 'rb') as f:
            files = {
                'image': (Path(image_path).name, f, 'image/jpeg')
            }
        
        # Prepare form data
        data = {
            'device': device,
            'output_dir': str(output_path.parent)
        }
        
        # Make request
        start_time = time.time()
        timeout = timeout or self.timeout
        
        try:
            response = requests.post(
                f"{self.server_url}/generate",
                files=files,
                data=data,
                timeout=timeout
            )
            response.raise_for_status()
            
            result = response.json()
            
            # TripoSR saves to output_dir/0/mesh.obj
            # Move to our desired output_path
            generated_path = Path(result.get('mesh_path', ''))
            if generated_path.exists():
                import shutil
                shutil.move(str(generated_path), str(output_path))
                print(f"✓ Moved model to {output_path}")
            
            elapsed = time.time() - start_time
            
            return {
                "status": "success",
                "model_path": str(output_path),
                "format": "obj",
                "backend": "triposr",
                "generation_time": elapsed,
                "metadata": {
                    "device": device,
                    "has_texture": False,
                    "quality": "fast"
                }
            }
        
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"TripoSR request timed out after {timeout}s. "
                "Try increasing timeout or check server status."
            )
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"TripoSR request failed: {e}")
    
    def batch_generate(
        self,
        image_paths: list[str],
        output_dir: str,
        device: str = "cuda:0"
    ) -> Dict[str, Dict]:
        """
        Generate 3D models for multiple images
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save all models
            device: Device to use
        
        Returns:
            Dict mapping image paths to generation results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        for img_path in image_paths:
            img_path = Path(img_path)
            output_path = output_dir / f"{img_path.stem}.obj"
            
            try:
                result = self.generate_3d_model(
                    str(img_path),
                    str(output_path),
                    device=device
                )
                results[str(img_path)] = result
            except Exception as e:
                results[str(img_path)] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = TripoSRClient(server_url="http://localhost:8000")
    
    # Check health
    if client.health_check():
        print("✓ TripoSR server is healthy")
    else:
        print("✗ TripoSR server is not responding")
        exit(1)
    
    # Generate 3D model
    result = client.generate_3d_model(
        image_path="test_chair.png",
        output_path="output/chair.obj",
        device="cuda:0"
    )
    
    print(f"\n✓ Generated: {result['model_path']}")
    print(f"  Time: {result['generation_time']:.2f}s")

