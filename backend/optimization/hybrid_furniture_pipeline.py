"""
Hybrid Furniture Pipeline
Combines:
1. Detected furniture from room scan (YOLO)
2. User-uploaded photos → 3D models (TRELLIS/SAM3D)
3. Library 3D models
4. Gemini optimization
"""

from typing import Dict, List, Optional
import json
from pathlib import Path

from backend.segmentation.furniture_detector import FurnitureDetector
from backend.reconstruction.photo_to_3d import PhotoTo3DGenerator
from backend.optimization.gemini_optimizer import GeminiLayoutOptimizer


class HybridFurniturePipeline:
    """
    Complete pipeline that handles:
    - Furniture from room scan
    - Custom furniture from user photos
    - Library furniture models
    - Gemini-powered optimization
    """
    
    def __init__(
        self,
        gemini_api_key: str,
        photo_to_3d_backend: str = "trellis"
    ):
        self.detector = FurnitureDetector()
        self.photo_generator = PhotoTo3DGenerator(backend=photo_to_3d_backend)
        self.optimizer = GeminiLayoutOptimizer(api_key=gemini_api_key)
        
        self.custom_models_dir = Path("models/furniture/custom")
        self.custom_models_dir.mkdir(exist_ok=True, parents=True)
    
    def process_room_with_custom_furniture(
        self,
        room_scan_path: str,
        room_images_dir: str,
        custom_furniture_photos: Optional[Dict[str, str]] = None,
        identity_profile: str = "studious",
        budget: Optional[int] = None
    ) -> Dict:
        """
        Complete pipeline from room scan to optimized layout
        
        Args:
            room_scan_path: Path to 3D scan from Nerfstudio
            room_images_dir: Directory with images from scan (for YOLO)
            custom_furniture_photos: Dict of {furniture_name: photo_path} for custom items
            identity_profile: User's identity profile
            budget: Optional budget for new furniture
        
        Returns:
            Complete optimization plan with custom models
        """
        print("=== Hybrid Furniture Pipeline ===\n")
        
        # Step 1: Detect furniture in room scan
        print("Step 1: Detecting furniture in room scan...")
        detected_furniture = self.detector.detect_furniture_batch(room_images_dir)
        inventory = self.detector.aggregate_detections(detected_furniture)
        print(f"✓ Detected {len(inventory)} furniture types\n")
        
        # Step 2: Generate 3D models from user photos
        custom_models = {}
        if custom_furniture_photos:
            print("Step 2: Generating 3D models from your photos...")
            for furniture_name, photo_path in custom_furniture_photos.items():
                print(f"  Generating {furniture_name}...")
                
                result = self.photo_generator.generate_3d_model(
                    image_path=photo_path,
                    output_path=str(self.custom_models_dir / f"{furniture_name}.glb"),
                    object_type=furniture_name.split('_')[0],  # e.g., "my_bed" → "bed"
                    cleanup=True
                )
                
                if result["status"] == "success":
                    custom_models[furniture_name] = result
                    print(f"  ✓ {furniture_name} → {result['model_path']}")
            
            print(f"✓ Generated {len(custom_models)} custom 3D models\n")
        
        # Step 3: Build complete room data for Gemini
        print("Step 3: Preparing room data for AI optimization...")
        room_data = self._build_room_data(
            inventory=inventory,
            custom_models=custom_models,
            room_scan_path=room_scan_path
        )
        
        # Step 4: Let Gemini optimize with knowledge of custom models
        print("Step 4: AI optimizing layout based on your identity...")
        optimization_plan = self.optimizer.optimize_layout(
            room_data=room_data,
            identity_profile=identity_profile,
            budget=budget
        )
        
        # Step 5: Inject custom models into optimization plan
        optimization_plan = self._inject_custom_models(
            optimization_plan,
            custom_models
        )
        
        print("✓ Optimization complete!\n")
        
        return {
            "detected_furniture": inventory,
            "custom_models": custom_models,
            "optimization_plan": optimization_plan,
            "room_data": room_data
        }
    
    def _build_room_data(
        self,
        inventory: List[Dict],
        custom_models: Dict,
        room_scan_path: str
    ) -> Dict:
        """
        Build comprehensive room data including custom models
        """
        # Parse room dimensions from scan metadata
        # In production, extract from Nerfstudio output
        room_data = {
            "dimensions": {"width": 4.0, "length": 5.0, "height": 2.7},
            "room_type": "bedroom",
            "furniture": [],
            "windows": [],
            "doors": [],
            "custom_furniture_available": list(custom_models.keys())
        }
        
        # Add detected furniture
        for item in inventory:
            room_data["furniture"].append({
                "type": item["type"],
                "quantity": item["quantity"],
                "movable": item["movable"],
                "priority": item["priority"]
            })
        
        # Add custom models info
        for name, model_info in custom_models.items():
            room_data["furniture"].append({
                "type": model_info["metadata"].get("object_type", "unknown"),
                "name": name,
                "custom": True,
                "model_path": model_info["model_path"],
                "quality": model_info["quality"]
            })
        
        return room_data
    
    def _inject_custom_models(
        self,
        optimization_plan: Dict,
        custom_models: Dict
    ) -> Dict:
        """
        Update optimization plan to use custom models where appropriate
        """
        # If Gemini suggests moving furniture that we have custom models for,
        # update the plan to reference the custom model
        
        if "furniture_changes" in optimization_plan:
            for change in optimization_plan["furniture_changes"]:
                item_name = change["item"]
                
                # Check if we have a custom model for this item
                for custom_name, model_info in custom_models.items():
                    if item_name in custom_name or custom_name in item_name:
                        change["custom_model"] = model_info["model_path"]
                        change["use_custom"] = True
        
        # Add metadata about custom models
        optimization_plan["custom_models_used"] = list(custom_models.keys())
        
        return optimization_plan
    
    def explain_pipeline(self) -> str:
        """
        Explain what this pipeline does
        """
        return """
🏗️ Hybrid Furniture Pipeline

This pipeline combines multiple sources of furniture:

1. DETECTED FURNITURE (from room scan)
   → YOLO-v8 detects furniture in your room
   → Knows what you currently have
   
2. CUSTOM FURNITURE (from your photos)
   → You upload a photo of your bed, desk, etc.
   → AI generates 3D model of YOUR actual furniture
   → Uses TRELLIS.2 or SAM 3D for high quality
   
3. LIBRARY FURNITURE (pre-made models)
   → Curated collection of 3D furniture
   → Used for new additions Gemini suggests
   
4. AI OPTIMIZATION (Gemini)
   → Understands your identity profile
   → Rearranges YOUR furniture (custom models)
   → Suggests new items from library
   → Explains every decision

RESULT: A 1:1 digital twin of your room with your actual furniture,
        optimized for who you want to be.
"""


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize pipeline
    pipeline = HybridFurniturePipeline(
        gemini_api_key=os.getenv("GOOGLE_AI_API_KEY"),
        photo_to_3d_backend="trellis"
    )
    
    print(pipeline.explain_pipeline())
    print("\n" + "="*60 + "\n")
    
    # Example: User uploads room scan + photos of their furniture
    result = pipeline.process_room_with_custom_furniture(
        room_scan_path="data/reconstructions/room_001/mesh.ply",
        room_images_dir="data/reconstructions/room_001/images",
        custom_furniture_photos={
            "my_bed": "data/photos/my_bed.jpg",
            "my_desk": "data/photos/my_desk.jpg",
            "my_chair": "data/photos/my_chair.jpg"
        },
        identity_profile="night_owl",
        budget=500
    )
    
    print("\n=== Results ===\n")
    print(f"Detected: {len(result['detected_furniture'])} furniture types")
    print(f"Custom models: {len(result['custom_models'])} generated")
    print(f"Optimization: {len(result['optimization_plan'].get('furniture_changes', []))} changes")
    
    print("\n" + result['optimization_plan'].get('summary', ''))


