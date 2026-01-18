"""
Example: How to use TripoSR in your room redesign workflow
"""

from backend.reconstruction.photo_to_3d import PhotoTo3DGenerator
from backend.segmentation.furniture_detector import FurnitureDetector
import os

# Set your RunPod server URL
os.environ["TRIPOSR_SERVER_URL"] = "http://your-runpod-ip:8000"


def process_room_photos(room_photos_dir: str, output_dir: str):
    """
    Complete workflow: Detect furniture → Generate 3D models → Ready for scene
    
    This is what happens when a user uploads room photos
    """
    
    # Step 1: Detect furniture in room photos
    print("🔍 Step 1: Detecting furniture in room photos...")
    detector = FurnitureDetector()
    furniture_detections = detector.detect_furniture_batch(room_photos_dir)
    
    print(f"✓ Found {len(furniture_detections)} furniture pieces:")
    for item in furniture_detections:
        print(f"  - {item['type']} at {item['bbox']}")
    
    # Step 2: Generate 3D models for each furniture piece
    print("\n🎨 Step 2: Generating 3D models with TripoSR...")
    generator = PhotoTo3DGenerator(backend="triposr")
    
    furniture_models = {}
    
    for detection in furniture_detections:
        furniture_id = detection["id"]
        furniture_type = detection["type"]
        
        # Crop furniture from room photo (you'd implement this)
        cropped_image = crop_furniture_from_photo(
            detection["image_path"],
            detection["bbox"]
        )
        
        # Generate 3D model
        print(f"  Generating 3D model for {furniture_type}...")
        result = generator.generate_3d_model(
            image_path=cropped_image,
            output_path=f"{output_dir}/{furniture_type}_{furniture_id}.obj",
            object_type=furniture_type
        )
        
        furniture_models[furniture_id] = {
            "type": furniture_type,
            "model_path": result["model_path"],
            "position": detection.get("position"),  # from room photo analysis
            "rotation": detection.get("rotation", (0, 0, 0))
        }
        
        print(f"  ✓ Generated: {result['model_path']} ({result['generation_time']:.1f}s)")
    
    # Step 3: Ready for scene building!
    print("\n✅ Step 3: All 3D models ready for scene building")
    print(f"   {len(furniture_models)} furniture models generated")
    
    return furniture_models


def crop_furniture_from_photo(image_path: str, bbox: dict) -> str:
    """
    Crop furniture from room photo using bounding box
    
    Args:
        image_path: Path to room photo
        bbox: {"x1": 100, "y1": 200, "x2": 500, "y2": 600}
    
    Returns:
        Path to cropped image
    """
    from PIL import Image
    
    # Load image
    img = Image.open(image_path)
    
    # Crop
    cropped = img.crop((bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    
    # Save cropped image
    cropped_path = image_path.replace(".jpg", "_cropped.jpg")
    cropped.save(cropped_path)
    
    return cropped_path


# Example usage
if __name__ == "__main__":
    # Process room photos
    furniture_models = process_room_photos(
        room_photos_dir="uploads/user_room_photos/",
        output_dir="models/furniture/"
    )
    
    # Now you can:
    # 1. Load floorplan → Create 3D room
    # 2. Place furniture models in room
    # 3. Let Gemini optimize layout
    # 4. Show Before/After toggle
    
    print("\n🎯 Next steps:")
    print("   1. Load floorplan → Create 3D room scene")
    print("   2. Place furniture models in scene")
    print("   3. Run Gemini optimization")
    print("   4. Return Before/After 3D scenes to frontend")

