#!/bin/bash
# Process a room photo: Detect furniture → Segment with SAM → Generate 3D models

ROOM_PHOTO="$1"
OUTPUT_DIR="${2:-processed_models}"

if [ -z "$ROOM_PHOTO" ]; then
    echo "Usage: ./process_room_photo.sh <room_photo_path> [output_dir]"
    echo ""
    echo "Example:"
    echo "  ./process_room_photo.sh furniture_photos/fullbed.png"
    exit 1
fi

if [ ! -f "$ROOM_PHOTO" ]; then
    echo "❌ Error: File not found: $ROOM_PHOTO"
    exit 1
fi

echo "🏠 Processing Room Photo: $ROOM_PHOTO"
echo "=" | awk '{for(i=0;i<60;i++)printf "=";print ""}'
echo ""

# Step 1: Detect furniture
echo "📸 Step 1: Detecting furniture..."
python3 << EOF
import sys
sys.path.insert(0, '.')
from backend.segmentation.furniture_detector import FurnitureDetector
from pathlib import Path
import json

detector = FurnitureDetector()
# Lower confidence to catch more objects (desk/nightstand might be detected as "table")
detections = detector.detect_furniture("$ROOM_PHOTO", confidence_threshold=0.3)

print(f"✓ Found {len(detections)} furniture pieces:")
for i, det in enumerate(detections):
    print(f"  {i+1}. {det['class']} (confidence: {det['confidence']:.2f})")

# Save detections (convert numpy types to Python types)
Path("$OUTPUT_DIR").mkdir(parents=True, exist_ok=True)
detections_serializable = []
for det in detections:
    det_clean = {
        'class': str(det['class']),
        'confidence': float(det['confidence']),
        'bbox': [float(x) for x in det['bbox']],
        'center': [float(x) for x in det['center']],
        'area': float(det['area'])
    }
    detections_serializable.append(det_clean)

with open("$OUTPUT_DIR/detections.json", "w") as f:
    json.dump(detections_serializable, f, indent=2)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Detection failed"
    exit 1
fi

echo ""

# Step 2: Segment and crop with SAM
echo "✂️  Step 2: Segmenting objects with SAM..."
python3 << EOF
import sys
sys.path.insert(0, '.')
from backend.segmentation.sam_segmenter import SAMSegmenter, SimpleSegmenter
from backend.segmentation.furniture_detector import FurnitureDetector
from pathlib import Path
import json

# Load detections
with open("$OUTPUT_DIR/detections.json", "r") as f:
    detections = json.load(f)

# Try SAM, fallback to simple
try:
    segmenter = SAMSegmenter(model_type="vit_b")
    print("  Using SAM for segmentation")
except:
    segmenter = SimpleSegmenter()
    print("  Using simple bbox cropping")

# Crop each object
cropped_dir = Path("$OUTPUT_DIR/cropped_images")
cropped_dir.mkdir(exist_ok=True)

cropped_files = []
for i, det in enumerate(detections):
    furniture_id = f"{det['class']}_{i}"
    bbox = det['bbox']
    
    if hasattr(segmenter, 'segment_from_bbox'):
        # Use SAM
        seg_result = segmenter.segment_from_bbox("$ROOM_PHOTO", bbox)
        cropped_path = segmenter.crop_with_mask(
            "$ROOM_PHOTO",
            seg_result['mask'],
            output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
        )
    else:
        # Simple cropping
        cropped_path = segmenter.crop_from_bbox(
            "$ROOM_PHOTO",
            bbox,
            output_path=str(cropped_dir / f"{furniture_id}_cropped.jpg")
        )
    
    cropped_files.append({
        "id": furniture_id,
        "type": det['class'],
        "cropped_path": cropped_path,
        "bbox": bbox
    })
    print(f"  ✓ Cropped {det['class']} → {cropped_path}")

# Save cropped files list
with open("$OUTPUT_DIR/cropped_files.json", "w") as f:
    json.dump(cropped_files, f, indent=2)
EOF

if [ $? -ne 0 ]; then
    echo "❌ Segmentation failed"
    exit 1
fi

echo ""

# Step 3: Generate 3D models using existing script
echo "🎨 Step 3: Generating 3D models (using RunPod)..."
python3 << EOF
import sys
sys.path.insert(0, '.')
from pathlib import Path
import json
import subprocess

# Load cropped files
with open("$OUTPUT_DIR/cropped_files.json", "r") as f:
    cropped_files = json.load(f)

print(f"  Processing {len(cropped_files)} objects...")
print("")

# Process each with existing script
for item in cropped_files:
    cropped_path = item['cropped_path']
    furniture_type = item['type']
    
    print(f"  📦 Processing {furniture_type}...")
    result = subprocess.run(
        ["./process_to_3d.sh", cropped_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"    ✓ {furniture_type} done")
    else:
        print(f"    ✗ {furniture_type} failed")
    print("")
EOF

echo ""
echo "=" | awk '{for(i=0;i<60;i++)printf "=";print ""}'
echo "✅ Complete! Check $OUTPUT_DIR/ for all 3D models"
echo ""

