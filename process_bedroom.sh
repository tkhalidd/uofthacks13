#!/bin/bash
# Quick script to process bedroom.jpg → 3D bed model

echo "🛏️  Processing Bedroom Photo → 3D Bed Model"
echo ""

# Find bedroom.jpg
BEDROOM_PHOTO=""
if [ -f "bedroom.jpg" ]; then
    BEDROOM_PHOTO="bedroom.jpg"
elif [ -f "furniture_photos/bedroom.jpg" ]; then
    BEDROOM_PHOTO="furniture_photos/bedroom.jpg"
elif [ -f ~/Desktop/bedroom.jpg ]; then
    BEDROOM_PHOTO=~/Desktop/bedroom.jpg
elif [ -f ~/Downloads/bedroom.jpg ]; then
    BEDROOM_PHOTO=~/Downloads/bedroom.jpg
else
    echo "❌ Could not find bedroom.jpg"
    echo ""
    echo "Please either:"
    echo "  1. Put bedroom.jpg in this directory, OR"
    echo "  2. Put it in furniture_photos/ folder, OR"
    echo "  3. Tell me the full path to the file"
    exit 1
fi

echo "✅ Found: $BEDROOM_PHOTO"
echo ""

# Check if we need to crop
echo "⚠️  IMPORTANT: TripoSR works best with a single object."
echo "   Your bedroom photo has multiple items (bed, nightstands, etc.)"
echo ""
echo "Do you want to:"
echo "  1. Crop just the bed first (recommended)"
echo "  2. Process the whole photo (may not work well)"
echo ""
read -p "Enter choice (1 or 2): " choice

if [ "$choice" = "1" ]; then
    echo ""
    echo "📐 Let's crop the bed..."
    echo ""
    echo "Option A: Use image editor"
    echo "  - Open $BEDROOM_PHOTO in any image editor"
    echo "  - Crop around just the bed"
    echo "  - Save as furniture_photos/bed_cropped.jpg"
    echo ""
    echo "Option B: Use Python script"
    echo "  - Run: python3 scripts/crop_furniture.py $BEDROOM_PHOTO"
    echo ""
    read -p "Press Enter after you've cropped the bed, or type 'skip' to process whole photo: " response
    
    if [ "$response" != "skip" ]; then
        if [ -f "furniture_photos/bed_cropped.jpg" ]; then
            echo "✅ Found bed_cropped.jpg, processing..."
            ./process_to_3d.sh furniture_photos/bed_cropped.jpg
        else
            echo "⚠️  bed_cropped.jpg not found. Processing original photo instead..."
            ./process_to_3d.sh "$BEDROOM_PHOTO"
        fi
    else
        echo "Processing whole photo..."
        ./process_to_3d.sh "$BEDROOM_PHOTO"
    fi
else
    echo "Processing whole photo (may not work well with multiple objects)..."
    ./process_to_3d.sh "$BEDROOM_PHOTO"
fi

echo ""
echo "✅ Done! Check processed_models/ folder for your 3D model"

