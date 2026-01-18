"""
SAM 3D Client
Connects to SAM 3D Objects API running on Vultr GPU server
"""

import requests
from typing import Dict, Optional
from pathlib import Path
import time


class SAM3DClient:
    """
    Client for SAM 3D Objects API
    Sends photos to Vultr server and receives 3D models
    """
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """
        Initialize SAM 3D client
        
        Args:
            server_url: URL of Vultr SAM 3D server (e.g., "http://123.45.67.89:5001")
            api_key: Optional API key for authentication
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def health_check(self) -> bool:
        """
        Check if SAM 3D server is healthy
        
        Returns:
            True if server is healthy, False otherwise
        """
        try:
            response = requests.get(
                f"{self.server_url}/health",
                headers=self.headers,
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
        mask_path: Optional[str] = None,
        seed: int = 42,
        timeout: int = 300
    ) -> Dict:
        """
        Generate 3D model from photo
        
        Args:
            image_path: Path to furniture photo
            output_path: Where to save generated 3D model
            mask_path: Optional mask to isolate object
            seed: Random seed for reproducibility
            timeout: Request timeout in seconds
        
        Returns:
            Dict with model path and metadata
        """
        print(f"Sending {image_path} to SAM 3D server...")
        
        # Prepare files
        files = {}
        with open(image_path, 'rb') as f:
            files['image'] = f.read()
        
        if mask_path:
            with open(mask_path, 'rb') as f:
                files['mask'] = f.read()
        
        # Prepare form data
        data = {'seed': seed}
        
        # Make request
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/generate-3d",
                files={'image': files['image'], 'mask': files.get('mask')},
                data=data,
                headers=self.headers,
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                # Save the 3D model
                output_path = Path(output_path)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ 3D model generated in {elapsed_time:.1f}s")
                print(f"  Saved to: {output_path}")
                
                return {
                    "status": "success",
                    "model_path": str(output_path),
                    "format": "ply",
                    "generation_time": elapsed_time,
                    "backend": "sam3d-objects"
                }
            else:
                error_msg = response.json().get('error', 'Unknown error')
                raise Exception(f"Server returned error: {error_msg}")
        
        except requests.Timeout:
            raise Exception(f"Request timed out after {timeout}s")
        except Exception as e:
            raise Exception(f"Failed to generate 3D model: {e}")
    
    def batch_generate(
        self,
        image_paths: Dict[str, str],
        output_dir: str,
        seed: int = 42
    ) -> Dict[str, Dict]:
        """
        Generate 3D models for multiple photos
        
        Args:
            image_paths: Dict of {name: image_path}
            output_dir: Directory to save models
            seed: Random seed
        
        Returns:
            Dict of {name: result}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {}
        
        for name, image_path in image_paths.items():
            output_path = output_dir / f"{name}.ply"
            
            try:
                result = self.generate_3d_model(
                    image_path=image_path,
                    output_path=str(output_path),
                    seed=seed
                )
                results[name] = result
            except Exception as e:
                print(f"✗ Failed to generate {name}: {e}")
                results[name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return results
    
    def estimate_cost(
        self,
        num_images: int,
        gpu_cost_per_hour: float = 1.50
    ) -> Dict:
        """
        Estimate cost for generating 3D models
        
        Args:
            num_images: Number of images to process
            gpu_cost_per_hour: Cost of GPU per hour
        
        Returns:
            Dict with cost estimates
        """
        # Estimates based on A40/A6000
        time_per_image = 1.0  # minutes (after model loaded)
        model_load_time = 3.0  # minutes (one-time)
        
        total_time_minutes = model_load_time + (num_images * time_per_image)
        total_time_hours = total_time_minutes / 60
        
        total_cost = total_time_hours * gpu_cost_per_hour
        cost_per_image = total_cost / num_images if num_images > 0 else 0
        
        return {
            "num_images": num_images,
            "estimated_time_minutes": total_time_minutes,
            "estimated_cost_usd": round(total_cost, 2),
            "cost_per_image_usd": round(cost_per_image, 2),
            "gpu_cost_per_hour": gpu_cost_per_hour
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize client
    client = SAM3DClient(
        server_url=os.getenv("SAM3D_SERVER_URL", "http://localhost:5001"),
        api_key=os.getenv("SAM3D_API_KEY")
    )
    
    # Check server health
    print("Checking SAM 3D server...")
    if not client.health_check():
        print("✗ Server is not available")
        print("Make sure SAM 3D is running on your Vultr server")
        exit(1)
    
    print("✓ Server is healthy\n")
    
    # Generate 3D model from photo
    print("=== Single Photo to 3D ===\n")
    
    result = client.generate_3d_model(
        image_path="data/photos/my_bed.jpg",
        output_path="models/furniture/custom/my_bed.ply",
        seed=42
    )
    
    print(f"\n✓ Generated: {result['model_path']}")
    print(f"  Time: {result['generation_time']:.1f}s")
    print(f"  Format: {result['format']}")
    
    # Batch generation
    print("\n=== Batch Generation ===\n")
    
    batch_results = client.batch_generate(
        image_paths={
            "my_bed": "data/photos/my_bed.jpg",
            "my_desk": "data/photos/my_desk.jpg",
            "my_chair": "data/photos/my_chair.jpg"
        },
        output_dir="models/furniture/custom"
    )
    
    successful = sum(1 for r in batch_results.values() if r["status"] == "success")
    print(f"\n✓ Generated {successful}/{len(batch_results)} models")
    
    # Cost estimate
    print("\n=== Cost Estimate ===\n")
    
    estimate = client.estimate_cost(num_images=10, gpu_cost_per_hour=1.50)
    print(f"Processing 10 images:")
    print(f"  Time: ~{estimate['estimated_time_minutes']:.1f} minutes")
    print(f"  Cost: ${estimate['estimated_cost_usd']}")
    print(f"  Per image: ${estimate['cost_per_image_usd']}")


