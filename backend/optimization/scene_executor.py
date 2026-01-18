"""
Scene Executor
Takes Gemini's optimization plan and applies it to the 3D scene
"""

import json
from pathlib import Path
from typing import Dict, List
import trimesh
import numpy as np


class SceneExecutor:
    """
    Executes layout changes on 3D scene based on Gemini's optimization plan
    """
    
    def __init__(self, furniture_library_path: str = "models/furniture"):
        self.furniture_library_path = Path(furniture_library_path)
        self.furniture_library_path.mkdir(exist_ok=True, parents=True)
    
    def apply_optimization(
        self,
        original_scene_path: str,
        optimization_plan: Dict,
        output_path: str
    ) -> Dict:
        """
        Apply Gemini's optimization plan to create new scene
        
        Args:
            original_scene_path: Path to original 3D scan mesh
            optimization_plan: Output from GeminiLayoutOptimizer
            output_path: Where to save optimized scene
        
        Returns:
            Dict with paths to before/after scenes and metadata
        """
        # Load original scene
        print("Loading original scene...")
        scene = self._load_scene(original_scene_path)
        
        # Create a copy for the optimized version
        optimized_scene = {
            "room_mesh": scene["room_mesh"],
            "furniture": scene["furniture"].copy(),
            "metadata": {
                "optimization_applied": True,
                "identity_profile": optimization_plan.get("identity_profile"),
                "changes": []
            }
        }
        
        # Apply furniture movements/rotations
        if optimization_plan.get("furniture_changes"):
            print("Applying furniture changes...")
            for change in optimization_plan["furniture_changes"]:
                self._apply_furniture_change(optimized_scene, change)
        
        # Add new furniture
        if optimization_plan.get("furniture_additions"):
            print("Adding new furniture...")
            for addition in optimization_plan["furniture_additions"]:
                self._add_furniture(optimized_scene, addition)
        
        # Save both scenes
        before_path = f"{output_path}_before.glb"
        after_path = f"{output_path}_after.glb"
        
        self._save_scene(scene, before_path)
        self._save_scene(optimized_scene, after_path)
        
        return {
            "before": before_path,
            "after": after_path,
            "changes_applied": len(optimization_plan.get("furniture_changes", [])),
            "items_added": len(optimization_plan.get("furniture_additions", [])),
            "metadata": optimized_scene["metadata"]
        }
    
    def _load_scene(self, scene_path: str) -> Dict:
        """
        Load 3D scene from file
        
        In production, this would:
        1. Load the room mesh from Nerfstudio export
        2. Load segmented furniture pieces
        3. Parse metadata about positions
        """
        # Placeholder - in production, load actual mesh
        return {
            "room_mesh": None,  # trimesh.load(scene_path)
            "furniture": [],
            "metadata": {}
        }
    
    def _apply_furniture_change(self, scene: Dict, change: Dict):
        """
        Move, rotate, or remove furniture in the scene
        """
        item_name = change["item"]
        action = change["action"]
        
        # Find the furniture item in scene
        furniture_item = None
        for idx, item in enumerate(scene["furniture"]):
            if item["name"] == item_name:
                furniture_item = item
                furniture_idx = idx
                break
        
        if not furniture_item:
            print(f"Warning: Furniture '{item_name}' not found in scene")
            return
        
        if action == "move":
            # Update position
            new_pos = change["new_position"]
            furniture_item["position"] = new_pos
            
            # Apply transformation to mesh
            if furniture_item.get("mesh"):
                translation = np.array([
                    new_pos["x"],
                    new_pos["y"],
                    new_pos["z"]
                ])
                furniture_item["mesh"].apply_translation(translation)
            
            scene["metadata"]["changes"].append({
                "type": "move",
                "item": item_name,
                "reason": change["reason"]
            })
        
        elif action == "rotate":
            # Update rotation
            new_pos = change["new_position"]
            furniture_item["rotation"] = new_pos.get("rotation", 0)
            
            # Apply rotation to mesh
            if furniture_item.get("mesh"):
                angle = np.radians(new_pos.get("rotation", 0))
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    angle, [0, 0, 1]  # Rotate around Z axis
                )
                furniture_item["mesh"].apply_transform(rotation_matrix)
            
            scene["metadata"]["changes"].append({
                "type": "rotate",
                "item": item_name,
                "reason": change["reason"]
            })
        
        elif action == "remove":
            # Remove from scene
            scene["furniture"].pop(furniture_idx)
            
            scene["metadata"]["changes"].append({
                "type": "remove",
                "item": item_name,
                "reason": change["reason"]
            })
    
    def _add_furniture(self, scene: Dict, addition: Dict):
        """
        Add new furniture to the scene
        """
        model_id = addition["model_id"]
        position = addition["position"]
        
        # Load 3D model from library
        model_path = self.furniture_library_path / model_id
        
        if not model_path.exists():
            print(f"Warning: 3D model '{model_id}' not found")
            # In production, you might download from a CDN or use a placeholder
            return
        
        # Load mesh
        try:
            mesh = trimesh.load(str(model_path))
            
            # Apply position
            translation = np.array([
                position["x"],
                position["y"],
                position["z"]
            ])
            mesh.apply_translation(translation)
            
            # Apply rotation if specified
            if "rotation" in position:
                angle = np.radians(position["rotation"])
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    angle, [0, 0, 1]
                )
                mesh.apply_transform(rotation_matrix)
            
            # Add to scene
            scene["furniture"].append({
                "name": addition["item"],
                "type": addition["item"],
                "model_id": model_id,
                "mesh": mesh,
                "position": position,
                "metadata": {
                    "reason": addition["reason"],
                    "cost": addition.get("estimated_cost", 0)
                }
            })
            
            scene["metadata"]["changes"].append({
                "type": "add",
                "item": addition["item"],
                "reason": addition["reason"]
            })
        
        except Exception as e:
            print(f"Error loading furniture model: {e}")
    
    def _save_scene(self, scene: Dict, output_path: str):
        """
        Save scene to file (glTF format for web viewing)
        """
        # Combine room mesh and all furniture into one scene
        meshes = []
        
        if scene["room_mesh"]:
            meshes.append(scene["room_mesh"])
        
        for furniture in scene["furniture"]:
            if furniture.get("mesh"):
                meshes.append(furniture["mesh"])
        
        if meshes:
            # Combine all meshes
            combined = trimesh.util.concatenate(meshes)
            
            # Export as glTF (good for web)
            combined.export(output_path)
            print(f"Saved scene to {output_path}")
        
        # Also save metadata
        metadata_path = output_path.replace(".glb", "_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(scene["metadata"], f, indent=2)
    
    def generate_comparison_data(
        self,
        before_path: str,
        after_path: str,
        optimization_plan: Dict
    ) -> Dict:
        """
        Generate data for before/after comparison in frontend
        """
        return {
            "before": {
                "model_url": before_path,
                "description": "Original room layout"
            },
            "after": {
                "model_url": after_path,
                "description": optimization_plan.get("summary", "Optimized layout")
            },
            "changes": optimization_plan.get("furniture_changes", []),
            "additions": optimization_plan.get("furniture_additions", []),
            "benefits": optimization_plan.get("expected_benefits", []),
            "principles": optimization_plan.get("layout_principles_applied", [])
        }


# Example usage
if __name__ == "__main__":
    executor = SceneExecutor()
    
    # Example optimization plan from Gemini
    optimization_plan = {
        "identity_profile": "night_owl",
        "furniture_changes": [
            {
                "item": "desk",
                "action": "move",
                "current_position": {"x": 0.6, "y": 0.5, "z": 0, "rotation": 90},
                "new_position": {"x": 2.4, "y": 0.5, "z": 0, "rotation": 270},
                "reason": "Moved away from east window to reduce morning glare"
            }
        ],
        "furniture_additions": [
            {
                "item": "blackout_curtain",
                "model_id": "curtain_blackout_01.glb",
                "position": {"x": 0, "y": 2.0, "z": 1.0, "rotation": 0},
                "reason": "Block morning sunlight for better sleep",
                "estimated_cost": 60
            }
        ],
        "summary": "Optimized for night owl lifestyle"
    }
    
    # Apply optimization
    result = executor.apply_optimization(
        original_scene_path="data/reconstructions/room_001/mesh.ply",
        optimization_plan=optimization_plan,
        output_path="data/optimized/room_001"
    )
    
    print(json.dumps(result, indent=2))


