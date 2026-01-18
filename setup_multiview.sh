#!/bin/bash
# Quick setup script for multi-view reconstruction

echo "🔧 Setting up Multi-View 3D Reconstruction"
echo "=========================================="
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi
echo "✓ Python 3 found"

# Check pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi
echo "✓ pip found"

# Install core dependencies
echo ""
echo "📦 Installing core Python dependencies..."
pip install open3d opencv-python pillow numpy scipy scikit-image

# Check for COLMAP
echo ""
echo "🔍 Checking for COLMAP..."
if command -v colmap &> /dev/null; then
    echo "✅ COLMAP is installed"
    COLMAP_VERSION=$(colmap -h 2>&1 | head -n 1 || echo "unknown")
    echo "   Version: $COLMAP_VERSION"
else
    echo "⚠️  COLMAP not found"
    echo ""
    echo "COLMAP is optional but recommended for best quality."
    echo "Without it, the system will use mesh fusion fallback."
    echo ""
    echo "To install COLMAP:"
    echo "  macOS: brew install colmap"
    echo "  Linux: sudo apt-get install colmap"
    echo ""
    read -p "Continue without COLMAP? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for PyTorch (optional, for depth estimation)
echo ""
echo "🔍 Checking for PyTorch (optional, for depth estimation)..."
if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch is installed"
else
    echo "⚠️  PyTorch not found"
    echo ""
    echo "PyTorch is optional but improves results with depth estimation."
    echo ""
    read -p "Install PyTorch? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing PyTorch (CPU version)..."
        pip install torch torchvision
    fi
fi

# Check for SAM
echo ""
echo "🔍 Checking for Segment Anything (SAM)..."
if python3 -c "import segment_anything" 2>/dev/null; then
    echo "✅ SAM is installed"
else
    echo "⚠️  SAM not found"
    echo ""
    echo "SAM is used for better furniture segmentation."
    echo ""
    read -p "Install SAM? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Installing Segment Anything..."
        pip install git+https://github.com/facebookresearch/segment-anything.git
        
        echo "Downloading SAM checkpoint..."
        mkdir -p ~/.sam_checkpoints
        cd ~/.sam_checkpoints
        if [ ! -f "sam_vit_b_01ec64.pth" ]; then
            curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
        fi
        cd - > /dev/null
    fi
fi

# Check API key
echo ""
echo "🔑 Checking API keys..."
if [ -z "$ROBOFLOW_API_KEY" ]; then
    echo "⚠️  ROBOFLOW_API_KEY not set"
    echo ""
    echo "You need a Roboflow API key for furniture detection."
    echo "Get one free at: https://roboflow.com/settings"
    echo ""
    read -p "Enter your Roboflow API key (or press Enter to skip): " API_KEY
    if [ ! -z "$API_KEY" ]; then
        echo "export ROBOFLOW_API_KEY='$API_KEY'" >> ~/.bashrc
        echo "export ROBOFLOW_API_KEY='$API_KEY'" >> ~/.zshrc
        export ROBOFLOW_API_KEY="$API_KEY"
        echo "✓ API key saved"
    fi
else
    echo "✓ ROBOFLOW_API_KEY is set"
fi

# Summary
echo ""
echo "=========================================="
echo "📊 Setup Summary"
echo "=========================================="
echo ""

# Check what's available
HAS_COLMAP=false
HAS_OPEN3D=false
HAS_TORCH=false
HAS_SAM=false

command -v colmap &> /dev/null && HAS_COLMAP=true
python3 -c "import open3d" 2>/dev/null && HAS_OPEN3D=true
python3 -c "import torch" 2>/dev/null && HAS_TORCH=true
python3 -c "import segment_anything" 2>/dev/null && HAS_SAM=true

echo "Available features:"
if $HAS_COLMAP; then
    echo "  ✅ COLMAP multi-view reconstruction (best quality)"
else
    echo "  ⚠️  COLMAP not available (will use fallback)"
fi

if $HAS_OPEN3D; then
    echo "  ✅ Mesh fusion (robust fallback)"
else
    echo "  ❌ Mesh fusion not available (install open3d)"
fi

if $HAS_TORCH; then
    echo "  ✅ Depth estimation (improves results)"
else
    echo "  ⚠️  Depth estimation not available (optional)"
fi

if $HAS_SAM; then
    echo "  ✅ SAM segmentation (better cropping)"
else
    echo "  ⚠️  SAM not available (will use simple cropping)"
fi

echo ""
echo "=========================================="
echo "✨ Setup Complete!"
echo "=========================================="
echo ""
echo "To process room photos from multiple angles:"
echo ""
echo "  export ROBOFLOW_API_KEY='your_key'"
echo "  python process_multiview_improved.py furniture_photos/angles/"
echo ""
echo "See INSTALL_MULTIVIEW.md for detailed usage instructions."
echo ""

