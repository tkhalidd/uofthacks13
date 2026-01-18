"""
Example: Parallel Processing + SAM Segmentation
Shows how to process room photos with parallel 3D generation
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.pipeline.complete_pipeline import CompletePipeline


def main():
    # Configuration
    TRIPOSR_SERVER = os.getenv(
        "TRIPOSR_SERVER_URL",
        "http://69.30.85.117:8000"  # Your RunPod server
    )
    
    print("🚀 Parallel Processing + SAM Example")
    print("=" * 60)
    
    # Initialize pipeline
    print("\n1. Initializing pipeline...")
    pipeline = CompletePipeline(
        backend="triposr",
        max_workers=5,  # Process 5 objects simultaneously
        use_sam=True,   # Use SAM if available (falls back to simple if not)
        triposr_server_url=TRIPOSR_SERVER
    )
    
    # Example: Process room photos
    room_photos_dir = "furniture_photos/"
    output_dir = "processed_models/"
    
    if not Path(room_photos_dir).exists():
        print(f"\n⚠️  Directory not found: {room_photos_dir}")
        print("   Create it and add some room photos first!")
        return
    
    print(f"\n2. Processing room photos from: {room_photos_dir}")
    print(f"   Output directory: {output_dir}")
    
    # Process room
    results = pipeline.process_room(
        room_photos_dir=room_photos_dir,
        output_dir=output_dir,
        confidence_threshold=0.5
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("✅ RESULTS")
    print("=" * 60)
    print(f"Total furniture detected: {len(results['detections'])}")
    print(f"Successfully generated: {len(results['models'])}")
    print(f"Failed: {len(results.get('failed', {}))}")
    
    print("\n📦 Generated 3D Models:")
    for fid, model in results["models"].items():
        model_path = model.get("model_path", "N/A")
        gen_time = model.get("generation_time", 0)
        print(f"  ✓ {fid}: {model_path} ({gen_time:.1f}s)")
    
    if results.get('failed'):
        print("\n❌ Failed:")
        for fid, error in results['failed'].items():
            print(f"  ✗ {fid}: {error.get('error', 'Unknown error')}")
    
    # Get just the model paths for next steps
    model_paths = pipeline.get_model_paths(results)
    print(f"\n🎯 Next: Use these {len(model_paths)} .obj files to compose your 3D scene!")


if __name__ == "__main__":
    main()

