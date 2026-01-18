"""
Parallel Processing Pipeline
Processes multiple furniture objects simultaneously using asyncio
"""

import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from backend.reconstruction.photo_to_3d import PhotoTo3DGenerator
from backend.segmentation.furniture_detector import FurnitureDetector


@dataclass
class ProcessingJob:
    """Represents a single 3D generation job"""
    furniture_id: str
    furniture_type: str
    image_path: str
    bbox: List[float]
    output_path: str
    priority: int = 0  # Higher = process first


class ParallelProcessor:
    """
    Parallel processor for generating 3D models from furniture photos
    Uses ThreadPoolExecutor to process multiple objects simultaneously
    """
    
    def __init__(
        self,
        backend: str = "triposr",
        max_workers: int = 5,
        triposr_server_url: Optional[str] = None
    ):
        """
        Initialize parallel processor
        
        Args:
            backend: 3D generation backend ("triposr", "sam3d", etc.)
            max_workers: Maximum number of parallel workers
            triposr_server_url: TripoSR server URL (if using triposr backend)
        """
        self.backend = backend
        self.max_workers = max_workers
        
        # Initialize generator (shared across workers)
        if triposr_server_url:
            os.environ["TRIPOSR_SERVER_URL"] = triposr_server_url
        
        self.generator = PhotoTo3DGenerator(backend=backend)
        
        print(f"ParallelProcessor initialized: {max_workers} workers, backend={backend}")
    
    def process_single_object(
        self,
        job: ProcessingJob,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Process a single furniture object (used by workers)
        
        Args:
            job: Processing job
            progress_callback: Optional callback for progress updates
        
        Returns:
            Result dict with status and model path
        """
        try:
            if progress_callback:
                progress_callback(job.furniture_id, "started", 0)
            
            # Generate 3D model
            result = self.generator.generate_3d_model(
                image_path=job.image_path,
                output_path=job.output_path,
                object_type=job.furniture_type,
                bounding_box={"x1": job.bbox[0], "y1": job.bbox[1], 
                            "x2": job.bbox[2], "y2": job.bbox[3]} if job.bbox else None
            )
            
            if progress_callback:
                progress_callback(job.furniture_id, "completed", 100)
            
            return {
                "furniture_id": job.furniture_id,
                "furniture_type": job.furniture_type,
                "status": "success",
                **result
            }
        
        except Exception as e:
            if progress_callback:
                progress_callback(job.furniture_id, "failed", 0)
            
            return {
                "furniture_id": job.furniture_id,
                "furniture_type": job.furniture_type,
                "status": "error",
                "error": str(e)
            }
    
    def process_batch(
        self,
        jobs: List[ProcessingJob],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Dict]:
        """
        Process multiple jobs in parallel
        
        Args:
            jobs: List of processing jobs
            progress_callback: Optional callback(status, progress) for each job
        
        Returns:
            Dict mapping furniture_id to result
        """
        print(f"\n🚀 Processing {len(jobs)} objects in parallel ({self.max_workers} workers)...")
        
        # Sort by priority (higher first)
        jobs = sorted(jobs, key=lambda j: j.priority, reverse=True)
        
        results = {}
        start_time = time.time()
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs
            future_to_job = {
                executor.submit(self.process_single_object, job, progress_callback): job
                for job in jobs
            }
            
            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                try:
                    result = future.result()
                    results[job.furniture_id] = result
                    completed += 1
                    
                    if result["status"] == "success":
                        print(f"  ✓ [{completed}/{len(jobs)}] {job.furniture_type} → {result.get('model_path', 'N/A')}")
                    else:
                        print(f"  ✗ [{completed}/{len(jobs)}] {job.furniture_type} failed: {result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    results[job.furniture_id] = {
                        "furniture_id": job.furniture_id,
                        "status": "error",
                        "error": str(e)
                    }
                    completed += 1
                    print(f"  ✗ [{completed}/{len(jobs)}] {job.furniture_type} exception: {e}")
        
        elapsed = time.time() - start_time
        successful = sum(1 for r in results.values() if r.get("status") == "success")
        
        print(f"\n✅ Batch complete: {successful}/{len(jobs)} successful in {elapsed:.1f}s")
        print(f"   Average: {elapsed/len(jobs):.1f}s per object")
        
        return results


class RoomPhotoProcessor:
    """
    Complete pipeline: Detect → Segment → Generate 3D (parallel)
    """
    
    def __init__(
        self,
        detector: Optional[FurnitureDetector] = None,
        segmenter=None,  # SAMSegmenter or SimpleSegmenter
        processor: Optional[ParallelProcessor] = None,
        backend: str = "triposr",
        max_workers: int = 5
    ):
        """
        Initialize complete room photo processor
        
        Args:
            detector: Furniture detector (YOLO)
            segmenter: SAM segmenter or simple segmenter
            processor: Parallel processor
            backend: 3D generation backend
            max_workers: Number of parallel workers
        """
        self.detector = detector or FurnitureDetector()
        self.segmenter = segmenter
        
        if processor is None:
            processor = ParallelProcessor(backend=backend, max_workers=max_workers)
        self.processor = processor
        
        print("RoomPhotoProcessor initialized")
    
    def process_room_photos(
        self,
        room_photos_dir: str,
        output_dir: str,
        use_sam: bool = True,
        confidence_threshold: float = 0.5
    ) -> Dict:
        """
        Complete pipeline: Detect furniture → Segment → Generate 3D models
        
        Args:
            room_photos_dir: Directory with room photos
            output_dir: Where to save 3D models
            use_sam: Whether to use SAM for better segmentation
            confidence_threshold: YOLO detection confidence threshold
        
        Returns:
            Dict with all generated models and metadata
        """
        from pathlib import Path
        from backend.segmentation.sam_segmenter import SimpleSegmenter
        
        print("=" * 60)
        print("🏠 Processing Room Photos → 3D Furniture Models")
        print("=" * 60)
        
        # Step 1: Detect furniture
        print("\n📸 Step 1: Detecting furniture...")
        detections_by_image = self.detector.detect_furniture_batch(
            room_photos_dir,
            confidence_threshold=confidence_threshold
        )
        
        # Aggregate detections
        all_detections = []
        for image_name, detections in detections_by_image.items():
            image_path = Path(room_photos_dir) / image_name
            for det in detections:
                all_detections.append({
                    **det,
                    "image_path": str(image_path),
                    "image_name": image_name
                })
        
        print(f"✓ Found {len(all_detections)} furniture pieces across {len(detections_by_image)} images")
        
        # Step 2: Segment and prepare jobs
        print("\n✂️  Step 2: Segmenting objects...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Use simple segmenter if SAM not available
        if self.segmenter is None:
            self.segmenter = SimpleSegmenter()
            print("  Using simple bbox cropping (SAM not available)")
        
        jobs = []
        cropped_dir = output_dir / "cropped_images"
        cropped_dir.mkdir(exist_ok=True)
        
        for i, det in enumerate(all_detections):
            furniture_id = f"{det['class']}_{i}"
            furniture_type = det['class']
            bbox = det['bbox']
            image_path = det['image_path']
            
            # Crop object
            if use_sam and hasattr(self.segmenter, 'segment_from_bbox'):
                # Use SAM for better segmentation
                seg_result = self.segmenter.segment_from_bbox(image_path, bbox)
                cropped_path = self.segmenter.crop_with_mask(
                    image_path,
                    seg_result['mask'],
                    output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
                )
            else:
                # Simple bbox cropping
                cropped_path = self.segmenter.crop_from_bbox(
                    image_path,
                    bbox,
                    output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
                )
            
            # Create job
            job = ProcessingJob(
                furniture_id=furniture_id,
                furniture_type=furniture_type,
                image_path=cropped_path,
                bbox=bbox,
                output_path=str(output_dir / f"{furniture_id}.obj"),
                priority=1 if furniture_type in ['bed', 'desk', 'sofa'] else 0
            )
            jobs.append(job)
        
        print(f"✓ Prepared {len(jobs)} objects for 3D generation")
        
        # Step 3: Generate 3D models in parallel
        print("\n🎨 Step 3: Generating 3D models (parallel)...")
        results = self.processor.process_batch(jobs)
        
        # Step 4: Compile results
        successful_models = {
            fid: r for fid, r in results.items()
            if r.get("status") == "success"
        }
        
        print(f"\n✅ Complete! Generated {len(successful_models)}/{len(jobs)} 3D models")
        
        return {
            "detections": all_detections,
            "models": successful_models,
            "failed": {fid: r for fid, r in results.items() if r.get("status") != "success"},
            "output_dir": str(output_dir)
        }


# Example usage
if __name__ == "__main__":
    import os
    
    # Set TripoSR server URL
    os.environ["TRIPOSR_SERVER_URL"] = "http://your-runpod-ip:8000"
    
    # Initialize processor
    processor = RoomPhotoProcessor(
        backend="triposr",
        max_workers=5  # Process 5 objects at once
    )
    
    # Process room photos
    results = processor.process_room_photos(
        room_photos_dir="uploads/room_photos/",
        output_dir="models/furniture/",
        use_sam=False  # Set to True if SAM is installed
    )
    
    print(f"\n🎉 Generated {len(results['models'])} 3D models!")
    for fid, model in results['models'].items():
        print(f"  - {fid}: {model.get('model_path', 'N/A')}")

