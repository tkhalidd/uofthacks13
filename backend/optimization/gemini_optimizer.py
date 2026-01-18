"""
Gemini-Powered Layout Optimizer
Uses Google's Gemini AI to understand personality and optimize room layouts
"""

import google.generativeai as genai
import json
from typing import Dict, List, Optional
from pathlib import Path
import base64
from PIL import Image


class GeminiLayoutOptimizer:
    """
    Uses Gemini to analyze rooms and generate optimized layouts based on identity
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini optimizer
        
        Args:
            api_key: Google AI API key
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        # Identity profiles with design principles
        self.identity_profiles = {
            'night_owl': {
                'description': 'Works/studies late, sleeps in, needs darkness control',
                'principles': [
                    'Minimize morning light exposure',
                    'Optimize artificial lighting for night work',
                    'Position bed away from east-facing windows',
                    'Add blackout solutions',
                    'Create cozy evening atmosphere'
                ]
            },
            'early_bird': {
                'description': 'Wakes early, productive mornings, values natural light',
                'principles': [
                    'Maximize morning sunlight',
                    'Position desk near east-facing windows',
                    'Bed positioned to wake with sunrise',
                    'Open, energizing layout',
                    'Minimize evening artificial light needs'
                ]
            },
            'studious': {
                'description': 'Focused on learning, needs concentration, minimal distractions',
                'principles': [
                    'Desk in quiet zone with minimal visual distractions',
                    'Good natural light without glare',
                    'Separate study zone from rest zone',
                    'Organized storage for materials',
                    'Ergonomic furniture placement'
                ]
            },
            'creative': {
                'description': 'Artistic, needs inspiration, flexible space',
                'principles': [
                    'Maximize natural light',
                    'Flexible furniture arrangement',
                    'Inspiration wall space',
                    'Room for movement and experimentation',
                    'Visual interest and stimulation'
                ]
            },
            'minimalist': {
                'description': 'Values simplicity, clean spaces, intentional design',
                'principles': [
                    'Reduce visual clutter',
                    'Consolidate furniture',
                    'Clear circulation paths',
                    'Multi-functional pieces',
                    'Breathing room around items'
                ]
            },
            'restful': {
                'description': 'Prioritizes sleep and relaxation',
                'principles': [
                    'Bed as focal point',
                    'Minimize electronics in sleep zone',
                    'Soft, calming layout',
                    'Good air circulation',
                    'Separate work from rest areas'
                ]
            }
        }
        
        # 3D furniture library (models you have available)
        self.furniture_library = self._load_furniture_library()
    
    def optimize_layout(
        self,
        room_data: Dict,
        identity_profile: str,
        budget: Optional[int] = None,
        room_images: Optional[List[str]] = None
    ) -> Dict:
        """
        Use Gemini to optimize room layout based on identity
        
        Args:
            room_data: Dict with room dimensions, current furniture, etc.
            identity_profile: One of the identity profiles
            budget: Optional budget for new furniture
            room_images: Optional list of image paths of the 3D scan
        
        Returns:
            Dict with layout changes, additions, and explanations
        """
        # Build prompt for Gemini
        prompt = self._build_optimization_prompt(
            room_data, 
            identity_profile, 
            budget
        )
        
        # Prepare images if provided
        image_parts = []
        if room_images:
            for img_path in room_images[:5]:  # Max 5 images
                image_parts.append(self._load_image(img_path))
        
        # Call Gemini
        if image_parts:
            response = self.model.generate_content([prompt] + image_parts)
        else:
            response = self.model.generate_content(prompt)
        
        # Parse response
        optimization_plan = self._parse_gemini_response(response.text)
        
        return optimization_plan
    
    def _build_optimization_prompt(
        self,
        room_data: Dict,
        identity_profile: str,
        budget: Optional[int]
    ) -> str:
        """
        Build detailed prompt for Gemini
        """
        profile = self.identity_profiles.get(identity_profile, {})
        
        prompt = f"""You are an expert interior designer specializing in identity-based room optimization.

ROOM INFORMATION:
- Dimensions: {room_data.get('dimensions', 'unknown')}
- Current Furniture: {json.dumps(room_data.get('furniture', []), indent=2)}
- Windows: {json.dumps(room_data.get('windows', []), indent=2)}
- Doors: {json.dumps(room_data.get('doors', []), indent=2)}
- Room Type: {room_data.get('room_type', 'bedroom')}

USER IDENTITY PROFILE: {identity_profile.upper()}
Description: {profile.get('description', '')}

Design Principles for this profile:
{chr(10).join(f"- {p}" for p in profile.get('principles', []))}

AVAILABLE 3D FURNITURE MODELS:
{json.dumps(self.furniture_library, indent=2)}

BUDGET: {"$" + str(budget) if budget else "No budget specified"}

TASK:
Analyze the current room layout and generate an optimized layout that supports the user's identity.

Provide your response in the following JSON format:

{{
  "analysis": "Brief analysis of current layout issues",
  "furniture_changes": [
    {{
      "item": "furniture_name",
      "action": "move" or "remove" or "rotate",
      "current_position": {{"x": 0, "y": 0, "z": 0, "rotation": 0}},
      "new_position": {{"x": 0, "y": 0, "z": 0, "rotation": 0}},
      "reason": "Why this change supports the identity"
    }}
  ],
  "furniture_additions": [
    {{
      "item": "furniture_type",
      "model_id": "3d_model_filename",
      "position": {{"x": 0, "y": 0, "z": 0, "rotation": 0}},
      "reason": "Why this addition helps",
      "estimated_cost": 0
    }}
  ],
  "layout_principles_applied": ["principle 1", "principle 2"],
  "expected_benefits": ["benefit 1", "benefit 2"],
  "summary": "One sentence summary of the transformation"
}}

IMPORTANT:
- Use the room's coordinate system (origin at corner)
- Ensure furniture doesn't overlap
- Maintain clear circulation paths (min 2ft)
- Consider window and door swing clearances
- Only suggest additions within budget
- Prioritize changes that most impact the identity goal
"""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict:
        """
        Parse Gemini's JSON response
        """
        try:
            # Extract JSON from response (Gemini might wrap it in markdown)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                json_text = response_text[json_start:json_end].strip()
            else:
                json_text = response_text
            
            return json.loads(json_text)
        except json.JSONDecodeError as e:
            print(f"Failed to parse Gemini response: {e}")
            print(f"Response was: {response_text}")
            return {
                "error": "Failed to parse response",
                "raw_response": response_text
            }
    
    def suggest_furniture_from_library(
        self,
        furniture_type: str,
        style_preference: str = "modern"
    ) -> List[Dict]:
        """
        Get matching furniture from 3D library
        
        Args:
            furniture_type: Type of furniture needed
            style_preference: Style preference
        
        Returns:
            List of matching 3D models
        """
        matches = []
        for item in self.furniture_library:
            if item['type'] == furniture_type:
                if style_preference in item.get('styles', []):
                    matches.append(item)
        
        return matches
    
    def _load_furniture_library(self) -> List[Dict]:
        """
        Load available 3D furniture models
        In production, this would query a database
        """
        return [
            {
                "id": "desk_modern_01",
                "type": "desk",
                "name": "Modern Minimalist Desk",
                "model_file": "models/furniture/desk_modern_01.glb",
                "dimensions": {"width": 1.2, "depth": 0.6, "height": 0.75},
                "styles": ["modern", "minimalist"],
                "cost": 200
            },
            {
                "id": "desk_wood_01",
                "type": "desk",
                "name": "Wooden Study Desk",
                "model_file": "models/furniture/desk_wood_01.glb",
                "dimensions": {"width": 1.4, "depth": 0.7, "height": 0.75},
                "styles": ["traditional", "warm"],
                "cost": 300
            },
            {
                "id": "bookshelf_tall_01",
                "type": "bookshelf",
                "name": "Tall Bookshelf",
                "model_file": "models/furniture/bookshelf_tall_01.glb",
                "dimensions": {"width": 0.8, "depth": 0.3, "height": 2.0},
                "styles": ["modern", "minimalist"],
                "cost": 150
            },
            {
                "id": "lamp_desk_01",
                "type": "lamp",
                "name": "LED Desk Lamp",
                "model_file": "models/furniture/lamp_desk_01.glb",
                "dimensions": {"width": 0.2, "depth": 0.2, "height": 0.4},
                "styles": ["modern", "minimalist"],
                "cost": 40
            },
            {
                "id": "plant_small_01",
                "type": "plant",
                "name": "Small Potted Plant",
                "model_file": "models/furniture/plant_small_01.glb",
                "dimensions": {"width": 0.2, "depth": 0.2, "height": 0.3},
                "styles": ["natural", "modern"],
                "cost": 25
            },
            {
                "id": "curtain_blackout_01",
                "type": "curtain",
                "name": "Blackout Curtain",
                "model_file": "models/furniture/curtain_blackout_01.glb",
                "dimensions": {"width": 1.5, "depth": 0.1, "height": 2.5},
                "styles": ["modern", "minimalist"],
                "cost": 60
            },
            {
                "id": "chair_ergonomic_01",
                "type": "chair",
                "name": "Ergonomic Office Chair",
                "model_file": "models/furniture/chair_ergonomic_01.glb",
                "dimensions": {"width": 0.6, "depth": 0.6, "height": 1.2},
                "styles": ["modern", "ergonomic"],
                "cost": 250
            }
        ]
    
    def _load_image(self, image_path: str):
        """Load image for Gemini"""
        img = Image.open(image_path)
        return img
    
    def explain_changes(self, optimization_plan: Dict) -> str:
        """
        Generate human-readable explanation of changes
        """
        explanation = f"**Layout Optimization Summary**\n\n"
        explanation += f"{optimization_plan.get('analysis', '')}\n\n"
        
        if optimization_plan.get('furniture_changes'):
            explanation += "**Furniture Rearrangements:**\n"
            for change in optimization_plan['furniture_changes']:
                explanation += f"- **{change['item']}**: {change['reason']}\n"
            explanation += "\n"
        
        if optimization_plan.get('furniture_additions'):
            explanation += "**Recommended Additions:**\n"
            for addition in optimization_plan['furniture_additions']:
                cost = addition.get('estimated_cost', 0)
                explanation += f"- **{addition['item']}** (${cost}): {addition['reason']}\n"
            explanation += "\n"
        
        if optimization_plan.get('expected_benefits'):
            explanation += "**Expected Benefits:**\n"
            for benefit in optimization_plan['expected_benefits']:
                explanation += f"- {benefit}\n"
        
        return explanation


# Example usage
if __name__ == "__main__":
    import os
    
    # Initialize with API key
    optimizer = GeminiLayoutOptimizer(
        api_key=os.getenv("GOOGLE_AI_API_KEY")
    )
    
    # Example room data
    room_data = {
        "dimensions": {"width": 3.6, "length": 4.2, "height": 2.7},  # meters
        "room_type": "bedroom",
        "furniture": [
            {
                "type": "bed",
                "position": {"x": 1.8, "y": 2.1, "z": 0, "rotation": 0},
                "dimensions": {"width": 1.4, "length": 2.0, "height": 0.5}
            },
            {
                "type": "desk",
                "position": {"x": 0.6, "y": 0.5, "z": 0, "rotation": 90},
                "dimensions": {"width": 1.2, "length": 0.6, "height": 0.75}
            },
            {
                "type": "chair",
                "position": {"x": 0.6, "y": 1.0, "z": 0, "rotation": 270},
                "dimensions": {"width": 0.5, "length": 0.5, "height": 1.0}
            }
        ],
        "windows": [
            {
                "position": {"x": 0, "y": 2.0, "z": 1.0},
                "dimensions": {"width": 1.2, "height": 1.5},
                "orientation": "east"
            }
        ],
        "doors": [
            {
                "position": {"x": 3.6, "y": 1.0, "z": 0},
                "dimensions": {"width": 0.9, "height": 2.1},
                "swing": "inward"
            }
        ]
    }
    
    # Optimize for night owl identity
    print("Optimizing room for 'night owl' identity...\n")
    
    result = optimizer.optimize_layout(
        room_data=room_data,
        identity_profile="night_owl",
        budget=500
    )
    
    print(json.dumps(result, indent=2))
    print("\n" + "="*50 + "\n")
    print(optimizer.explain_changes(result))


