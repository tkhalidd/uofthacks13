#!/bin/bash
# Simple script to process furniture photos with TripoSR on RunPod

# Check if SSH key is in ssh-agent (to avoid passphrase prompts)
if ! ssh-add -l 2>/dev/null | grep -q "id_ed25519\|id_rsa"; then
    echo "⚠️  SSH key not in ssh-agent. You'll be asked for your passphrase."
    echo "   Run './setup_ssh.sh' first to avoid this, or enter your passphrase when prompted."
    echo ""
fi

# Configuration - Get these from RunPod dashboard → Connect → SSH
# Option 1: Using SSH over exposed TCP (recommended for SCP)
RUNPOD_HOST="69.30.85.117"  # Direct IP from dashboard
RUNPOD_USER="root"
RUNPOD_PORT="22082"  # Port from dashboard (just the number)
# SSH key path - will auto-detect or set manually
SSH_KEY=""  # Leave empty to auto-detect, or set path like "/Users/yourname/.ssh/id_ed25519"

# Option 2: Using SSH proxy (if Option 1 doesn't work)
# RUNPOD_HOST="3kwu0r88n5bvqc-644113db@ssh.runpod.io"
# RUNPOD_USER=""
# RUNPOD_PORT=""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  🛏️  Furniture Photo → 3D Model Processor${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if image provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: ./process_to_3d.sh <image.jpg>${NC}"
    echo ""
    echo "Example:"
    echo "  ./process_to_3d.sh furniture_photos/bed.jpg"
    echo ""
    echo "Or process all images in furniture_photos/:"
    echo "  ./process_to_3d.sh furniture_photos/*.jpg"
    exit 1
fi

# Process each image
for IMAGE_PATH in "$@"; do
    if [ ! -f "$IMAGE_PATH" ]; then
        echo -e "${YELLOW}⚠️  File not found: $IMAGE_PATH${NC}"
        continue
    fi
    
    # Get filename without extension
    FURNITURE_NAME=$(basename "$IMAGE_PATH" | sed 's/\.[^.]*$//')
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}📸 Processing: $FURNITURE_NAME${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Step 1: Upload
    echo -e "${BLUE}📤 Step 1/3: Uploading to RunPod...${NC}"
    echo "   Connecting to: ${RUNPOD_USER}@${RUNPOD_HOST}"
    
    # Auto-detect SSH key if not set
    if [ -z "$SSH_KEY" ]; then
        if [ -f ~/.ssh/id_ed25519 ]; then
            SSH_KEY=~/.ssh/id_ed25519
        elif [ -f ~/.ssh/id_rsa ]; then
            SSH_KEY=~/.ssh/id_rsa
        fi
    fi
    
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    
    if [ -n "$RUNPOD_PORT" ]; then
        scp -P "$RUNPOD_PORT" $SSH_OPTS "$IMAGE_PATH" ${RUNPOD_USER}@${RUNPOD_HOST}:~/TripoSR/${FURNITURE_NAME}.jpg
    else
        scp $SSH_OPTS "$IMAGE_PATH" ${RUNPOD_USER}@${RUNPOD_HOST}:~/TripoSR/${FURNITURE_NAME}.jpg
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}❌ Upload failed!${NC}"
        echo ""
        echo -e "${YELLOW}Possible issues:${NC}"
        echo "   1. RunPod instance is stopped → Go to dashboard and start it"
        echo "   2. Wrong hostname → Update RUNPOD_HOST in this script"
        echo "   3. Need different port → Update RUNPOD_PORT in this script"
        echo ""
        echo -e "${YELLOW}To fix:${NC}"
        echo "   1. Go to https://www.runpod.io/ → Your Pod → Connect"
        echo "   2. Copy the SSH connection string"
        echo "   3. Edit this script and update RUNPOD_HOST (and RUNPOD_PORT if needed)"
        echo ""
        echo -e "${YELLOW}Test connection manually:${NC}"
        if [ -n "$RUNPOD_PORT" ]; then
            echo "   ssh -p $RUNPOD_PORT ${RUNPOD_USER}@${RUNPOD_HOST}"
        else
            echo "   ssh ${RUNPOD_USER}@${RUNPOD_HOST}"
        fi
        continue
    fi
    echo -e "${GREEN}✅ Upload complete${NC}"
    echo ""
    
    # Step 2: Process
    echo -e "${BLUE}🎨 Step 2/3: Generating 3D model on RunPod...${NC}"
    echo "   (This takes ~20-30 seconds)"
    # Auto-detect SSH key if not set
    if [ -z "$SSH_KEY" ]; then
        if [ -f ~/.ssh/id_ed25519 ]; then
            SSH_KEY=~/.ssh/id_ed25519
        elif [ -f ~/.ssh/id_rsa ]; then
            SSH_KEY=~/.ssh/id_rsa
        fi
    fi
    
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    
    if [ -n "$RUNPOD_PORT" ]; then
        ssh -p "$RUNPOD_PORT" $SSH_OPTS ${RUNPOD_USER}@${RUNPOD_HOST} << EOF
cd ~/TripoSR
source venv/bin/activate
python run.py ${FURNITURE_NAME}.jpg --output-dir output/${FURNITURE_NAME}/ --device cuda --mc-resolution 256
EOF
    else
        ssh $SSH_OPTS ${RUNPOD_USER}@${RUNPOD_HOST} << EOF
cd ~/TripoSR
source venv/bin/activate
python run.py ${FURNITURE_NAME}.jpg --output-dir output/${FURNITURE_NAME}/ --device cuda --mc-resolution 256
EOF
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}❌ Processing failed${NC}"
        continue
    fi
    echo -e "${GREEN}✅ 3D model generated${NC}"
    echo ""
    
    # Step 3: Download
    echo -e "${BLUE}📥 Step 3/3: Downloading 3D model...${NC}"
    OUTPUT_DIR="processed_models/${FURNITURE_NAME}"
    mkdir -p "$OUTPUT_DIR"
    
    # Auto-detect SSH key if not set
    if [ -z "$SSH_KEY" ]; then
        if [ -f ~/.ssh/id_ed25519 ]; then
            SSH_KEY=~/.ssh/id_ed25519
        elif [ -f ~/.ssh/id_rsa ]; then
            SSH_KEY=~/.ssh/id_rsa
        fi
    fi
    
    SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
    if [ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ]; then
        SSH_OPTS="$SSH_OPTS -i $SSH_KEY"
    fi
    
    if [ -n "$RUNPOD_PORT" ]; then
        scp -P "$RUNPOD_PORT" $SSH_OPTS ${RUNPOD_USER}@${RUNPOD_HOST}:~/TripoSR/output/${FURNITURE_NAME}/0/mesh.obj "$OUTPUT_DIR/${FURNITURE_NAME}_3d.obj"
    else
        scp $SSH_OPTS ${RUNPOD_USER}@${RUNPOD_HOST}:~/TripoSR/output/${FURNITURE_NAME}/0/mesh.obj "$OUTPUT_DIR/${FURNITURE_NAME}_3d.obj"
    fi
    
    if [ $? -ne 0 ]; then
        echo -e "${YELLOW}❌ Download failed${NC}"
        continue
    fi
    
    echo -e "${GREEN}✅ Complete!${NC}"
    echo ""
    echo -e "${GREEN}📁 3D model saved to: $OUTPUT_DIR/${FURNITURE_NAME}_3d.obj${NC}"
    echo -e "${BLUE}🌐 View it online at: https://3dviewer.net/${NC}"
    echo ""
done

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✨ All done! Check processed_models/ folder${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

