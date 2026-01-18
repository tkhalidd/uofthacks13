"""
SAM 3D Room Client
Handles room reconstruction using SAM 3D Objects
"""

import requests
from typing import Dict, List, Optional
from pathlib import Path
import time
import subprocess


class SAM3DRoomClient:
    """
    Client for SAM 3D room reconstruction
    Extends SAM3DClient with room-specific methods
    """
    
    def __init__(self, server_url: str, api_key: Optional[str] = None):
        """
        Initialize SAM 3D room client
        
        Args:
            server_url: URL of Vultr SAM 3D server
            api_key: Optional API key for authentication
        """
        self.server_url = server_url.rstrip('/')
        self.api_key = api_key
        self.headers = {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    def generate_room_from_photos(
        self,
        photos: List[str],
        output_path: str,
        merge_method: str = "auto",
        timeout: int = 600
    ) -> Dict:
        """
        Generate 3D room model from multiple photos
        
        Args:
            photos: List of photo paths (5-10 photos recommended)
            output_path: Where to save room model
            merge_method: How to merge multiple reconstructions ("auto", "manual")
            timeout: Request timeout in seconds
        
        Returns:
            Dict with model path and metadata
        """
        print(f"Generating room from {len(photos)} photos...")
        
        # Prepare files
        files = []
        for i, photo_path in enumerate(photos):
            with open(photo_path, 'rb') as f:
                files.append(('photos', (f'photo_{i}.jpg', f.read(), 'image/jpeg')))
        
        # Prepare form data
        data = {
            'merge_method': merge_method,
            'output_format': 'ply'
        }
        
        # Make request
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.server_url}/generate-room",
                files=files,
                data=data,
                headers=self.headers,
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                # Save the room model
                output_path = Path(output_path)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✓ Room model generated in {elapsed_time:.1f}s")
                print(f"  Saved to: {output_path}")
                
                return {
                    "status": "success",
                    "model_path": str(output_path),
                    "format": "ply",
                    "generation_time": elapsed_time,
                    "num_photos": len(photos),
                    "backend": "sam3d-room"
                }
            else:
                error_msg = response.json().get('error', 'Unknown error')
                raise Exception(f"Server returned error: {error_msg}")
        
        except requests.Timeout:
            raise Exception(f"Request timed out after {timeout}s")
        except Exception as e:
            raise Exception(f"Failed to generate room model: {e}")
    
    def generate_room_from_video(
        self,
        video_path: str,
        output_path: str,
        frame_interval: int = 30,
        max_frames: int = 20,
        timeout: int = 600
    ) -> Dict:
        """
        Generate 3D room model from video
        
        Args:
            video_path: Path to video file
            output_path: Where to save room model
            frame_interval: Extract every Nth frame
            max_frames: Maximum frames to use
            timeout: Request timeout
        
        Returns:
            Dict with model path and metadata
        """
        print(f"Extracting frames from video...")
        
        # Extract frames from video
        frames_dir = Path("temp/video_frames")
        frames_dir.mkdir(exist_ok=True, parents=True)
        
        # Use ffmpeg to extract frames
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vf', f'select=not(mod(n\\,{frame_interval}))',
            '-vsync', 'vfr',
            '-frames:v', str(max_frames),
            str(frames_dir / 'frame_%04d.jpg')
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to extract frames: {e}")
        
        # Get extracted frames
        frame_files = sorted(frames_dir.glob('*.jpg'))
        
        if not frame_files:
            raise Exception("No frames extracted from video")
        
        print(f"✓ Extracted {len(frame_files)} frames")
        
        # Generate room from frames
        result = self.generate_room_from_photos(
            photos=[str(f) for f in frame_files],
            output_path=output_path,
            timeout=timeout
        )
        
        # Cleanup temp frames
        for frame in frame_files:
            frame.unlink()
        frames_dir.rmdir()
        
        result['source'] = 'video'
        result['frame_interval'] = frame_interval
        
        return result
    
    def generate_furniture_model(
        self,
        image_path: str,
        output_path: str,
        furniture_type: Optional[str] = None,
        timeout: int = 300
    ) -> Dict:
        """
        Generate 3D furniture model from photo
        (Convenience method - same as SAM3DClient.generate_3d_model)
        
        Args:
            image_path: Path to furniture photo
            output_path: Where to save model
            furniture_type: Type of furniture (bed, desk, chair, etc.)
            timeout: Request timeout
        
        Returns:
            Dict with model path and metadata
        """
        print(f"Generating {furniture_type or 'furniture'} model...")
        
        with open(image_path, 'rb') as f:
            files = {'image': f}
            data = {'furniture_type': furniture_type} if furniture_type else {}
            
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/generate-furniture",
                files=files,
                data=data,
                headers=self.headers,
                timeout=timeout
            )
            
            elapsed_time = time.time() - start_time
            
            if response.status_code == 200:
                output_path = Path(output_path)
                output_path.parent.mkdir(exist_ok=True, parents=True)
                
                with open(output_path, 'wb') as out:
                    out.write(response.content)
                
                print(f"✓ Furniture model generated in {elapsed_time:.1f}s")
                
                return {
                    "status": "success",
                    "model_path": str(output_path),
                    "format": "ply",
                    "generation_time": elapsed_time,
                    "furniture_type": furniture_type,
                    "backend": "sam3d-furniture"
                }
            else:
                error_msg = response.json().get('error', 'Unknown error')
                raise Exception(f"Server returned error: {error_msg}")
    
    def complete_room_reconstruction(
        self,
        room_photos: List[str],
        furniture_photos: Dict[str, str],
        output_dir: str
    ) -> Dict:
        """
        Complete room reconstruction pipeline
        
        Args:
            room_photos: List of room photo paths
            furniture_photos: Dict of {name: photo_path} for furniture
            output_dir: Directory to save all models
        
        Returns:
            Dict with all generated models
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        results = {
            "room": None,
            "furniture": {},
            "total_time": 0
        }
        
        start_time = time.time()
        
        # 1. Generate room model
        print("\n=== Generating Room Model ===\n")
        room_result = self.generate_room_from_photos(
            photos=room_photos,
            output_path=str(output_dir / "room.ply")
        )
        results["room"] = room_result
        
        # 2. Generate furniture models
        print("\n=== Generating Furniture Models ===\n")
        for name, photo_path in furniture_photos.items():
            try:
                furniture_result = self.generate_furniture_model(
                    image_path=photo_path,
                    output_path=str(output_dir / f"{name}.ply"),
                    furniture_type=name.split('_')[0]  # e.g., "my_bed" → "bed"
                )
                results["furniture"][name] = furniture_result
            except Exception as e:
                print(f"✗ Failed to generate {name}: {e}")
                results["furniture"][name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        results["total_time"] = time.time() - start_time
        
        print(f"\n✓ Complete reconstruction finished in {results['total_time']:.1f}s")
        print(f"  Room: {results['room']['model_path']}")
        print(f"  Furniture: {len([f for f in results['furniture'].values() if f['status'] == 'success'])} models")
        
        return results


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize client
    client = SAM3DRoomClient(
        server_url=os.getenv("SAM3D_SERVER_URL", "http://localhost:5001"),
        api_key=os.getenv("SAM3D_API_KEY")
    )
    
    # Complete room reconstruction
    print("=== Complete Room Reconstruction ===\n")
    
    results = client.complete_room_reconstruction(
        room_photos=[
            "data/photos/room_wall1.jpg",
            "data/photos/room_wall2.jpg",
            "data/photos/room_wall3.jpg",
            "data/photos/room_wall4.jpg",
            "data/photos/room_ceiling.jpg"
        ],
        furniture_photos={
            "my_bed": "data/photos/bed.jpg",
            "my_desk": "data/photos/desk.jpg",
            "my_chair": "data/photos/chair.jpg"
        },
        output_dir="models/reconstructions/room_001"
    )
    
    print("\n=== Results ===")
    print(f"Room model: {results['room']['model_path']}")
    print(f"Furniture models: {len(results['furniture'])}")
    print(f"Total time: {results['total_time']:.1f}s")


