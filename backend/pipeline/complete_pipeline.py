"""
Complete Pipeline: Room Photos → 3D Models → Scene Composition
Combines detection, SAM segmentation, and parallel 3D generation
"""

from pathlib import Path
from typing import Dict, List, Optional
import os

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.segmentation.sam_segmenter import SAMSegmenter, SimpleSegmenter
from backend.pipeline.parallel_processor import RoomPhotoProcessor


class CompletePipeline:
    """
    End-to-end pipeline for processing room photos into 3D furniture models
    """
    
    def __init__(
        self,
        backend: str = "triposr",
        max_workers: int = 5,
        use_sam: bool = True,
        triposr_server_url: Optional[str] = None
    ):
        """
        Initialize complete pipeline
        
        Args:
            backend: 3D generation backend ("triposr", "sam3d")
            max_workers: Number of parallel workers
            use_sam: Whether to use SAM for segmentation (better quality)
            triposr_server_url: TripoSR server URL
        """
        self.backend = backend
        self.max_workers = max_workers
        self.use_sam = use_sam
        
        # Initialize components
        self.detector = FurnitureDetector()
        
        # Try to initialize SAM, fallback to simple segmenter
        self.segmenter = None
        if use_sam:
            try:
                self.segmenter = SAMSegmenter(model_type="vit_b")  # Fastest SAM model
                print("✓ SAM segmenter initialized")
            except (ImportError, FileNotFoundError) as e:
                print(f"⚠️  SAM not available: {e}")
                print("   Falling back to simple bbox cropping")
                self.segmenter = SimpleSegmenter()
        else:
            self.segmenter = SimpleSegmenter()
        
        # Initialize processor
        self.processor = RoomPhotoProcessor(
            detector=self.detector,
            segmenter=self.segmenter,
            backend=backend,
            max_workers=max_workers
        )
        
        if triposr_server_url:
            os.environ["TRIPOSR_SERVER_URL"] = triposr_server_url
        
        print(f"CompletePipeline initialized: backend={backend}, workers={max_workers}, SAM={use_sam}")
    
    def process_room(
        self,
        room_photos_dir: str,
        output_dir: str,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Process room photos into 3D furniture models
        
        Args:
            room_photos_dir: Directory containing room photos
            output_dir: Where to save 3D models
            confidence_threshold: YOLO detection confidence
        
        Returns:
            Complete results with models and metadata
        """
        return self.processor.process_room_photos(
            room_photos_dir=room_photos_dir,
            output_dir=output_dir,
            use_sam=self.use_sam,
            confidence_threshold=confidence_threshold
        )
    
    def get_model_paths(self, results: Dict) -> Dict[str, str]:
        """
        Extract just the model paths from results
        
        Returns:
            Dict mapping furniture_id to .obj file path
        """
        return {
            fid: model.get("model_path")
            for fid, model in results["models"].items()
            if model.get("model_path")
        }


# Example usage
if __name__ == "__main__":
    import sys
    
    # Configuration
    TRIPOSR_SERVER = os.getenv("TRIPOSR_SERVER_URL", "http://your-runpod-ip:8000")
    
    # Initialize pipeline
    pipeline = CompletePipeline(
        backend="triposr",
        max_workers=5,  # Process 5 objects simultaneously
        use_sam=True,   # Use SAM for better segmentation
        triposr_server_url=TRIPOSR_SERVER
    )
    
    # Process room photos
    if len(sys.argv) > 1:
        room_photos_dir = sys.argv[1]
    else:
        room_photos_dir = "uploads/room_photos/"
    
    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = "models/furniture/"
    
    print(f"\n📁 Processing: {room_photos_dir}")
    print(f"💾 Output: {output_dir}\n")
    
    results = pipeline.process_room(
        room_photos_dir=room_photos_dir,
        output_dir=output_dir
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Total detections: {len(results['detections'])}")
    print(f"Successful models: {len(results['models'])}")
    print(f"Failed: {len(results.get('failed', {}))}")
    
    print("\n✅ Generated Models:")
    for fid, model in results["models"].items():
        print(f"  - {fid}: {model.get('model_path', 'N/A')}")
    
    if results.get('failed'):
        print("\n❌ Failed:")
        for fid, error in results['failed'].items():
            print(f"  - {fid}: {error.get('error', 'Unknown error')}")

