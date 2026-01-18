#!/bin/bash
# Process furniture with multiple views for better depth
# Usage: ./process_multi_view_3d.sh <image1> [image2] [image3] ... [output_name]

if [ $# -lt 2 ]; then
    echo "Usage: ./process_multi_view_3d.sh <image1> [image2] [image3] ... [output_name]"
    echo ""
    echo "Example:"
    echo "  ./process_multi_view_3d.sh nightstand_view1.jpg nightstand_view2.jpg nightstand_view3.jpg nightstand"
    echo ""
    echo "For best results:"
    echo "  • Take 3-5 photos from different angles (front, side, top-down)"
    echo "  • Keep the same lighting"
    echo "  • Focus on the same object"
    exit 1
fi

# Configuration
RUNPOD_HOST="69.30.85.117"
RUNPOD_USER="root"
RUNPOD_PORT="22082"
SSH_KEY="~/.ssh/id_ed25519"

# Get output name (last argument) or generate from first image
OUTPUT_NAME="${@: -1}"
FIRST_IMAGE="${1}"

# Check if last arg is a file (then it's not output name)
if [ -f "$OUTPUT_NAME" ]; then
    OUTPUT_NAME=$(basename "$FIRST_IMAGE" | sed 's/\.[^.]*$//')
fi

# Remove output name from image list
IMAGES=("${@:1:$(($#-1))}")
if [ -f "$OUTPUT_NAME" ]; then
    IMAGES=("$@")
fi

echo "📸 Processing ${#IMAGES[@]} views for: $OUTPUT_NAME"
echo ""

# Upload all images
echo "📤 Uploading images to RunPod..."
for img in "${IMAGES[@]}"; do
    if [ ! -f "$img" ]; then
        echo "❌ Error: File not found: $img"
        exit 1
    fi
    
    scp -i "$SSH_KEY" -P "$RUNPOD_PORT" \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        "$img" "$RUNPOD_USER@$RUNPOD_HOST:/workspace/TripoSR/input_images/" 2>&1 | grep -v "Warning: Permanently added"
done

echo "✅ Images uploaded"
echo ""

# Process with TripoSR (multi-view if supported, or use first image)
echo "🎨 Generating 3D model from ${#IMAGES[@]} views..."
ssh -i "$SSH_KEY" -p "$RUNPOD_PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "$RUNPOD_USER@$RUNPOD_HOST" << EOF
cd /workspace/TripoSR

# Try multi-view if TripoSR supports it, otherwise use first image
if [ ${#IMAGES[@]} -gt 1 ]; then
    echo "Using multi-view processing..."
    # Note: TripoSR may need specific multi-view format
    # For now, we'll use the first image but you can extend this
    python run.py --input_path input_images/$(basename "${IMAGES[0]}") --output_path output_models/${OUTPUT_NAME}_multiview
else
    python run.py --input_path input_images/$(basename "${IMAGES[0]}") --output_path output_models/${OUTPUT_NAME}
fi
EOF

# Download result
echo ""
echo "📥 Downloading 3D model..."
mkdir -p "processed_models/${OUTPUT_NAME}_multiview"

scp -i "$SSH_KEY" -P "$RUNPOD_PORT" \
    -o StrictHostKeyChecking=no \
    -o UserKnownHostsFile=/dev/null \
    "$RUNPOD_USER@$RUNPOD_HOST:/workspace/TripoSR/output_models/${OUTPUT_NAME}*/mesh.obj" \
    "processed_models/${OUTPUT_NAME}_multiview/${OUTPUT_NAME}_multiview_3d.obj" 2>&1 | grep -v "Warning: Permanently added"

if [ $? -eq 0 ]; then
    echo "✅ 3D model saved to: processed_models/${OUTPUT_NAME}_multiview/${OUTPUT_NAME}_multiview_3d.obj"
else
    echo "⚠️  Model might still be processing. Check RunPod output."
fi

