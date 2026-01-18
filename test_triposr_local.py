#!/usr/bin/env python3
"""
Quick test script for TripoSR on local Mac
Run this after installing TripoSR to verify it works
"""

import sys
import os
from pathlib import Path

# Add TripoSR to path if needed
triposr_path = Path(__file__).parent / "TripoSR"
if triposr_path.exists():
    sys.path.insert(0, str(triposr_path))

try:
    import torch
    from tsr.system import TSR
    from PIL import Image
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nMake sure you've:")
    print("1. Installed dependencies: pip install -r TripoSR/requirements.txt")
    print("2. Activated virtual environment if using one")
    sys.exit(1)

# Check device
if torch.backends.mps.is_available():
    device = "mps"
    print(f"✅ Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"✅ Using CUDA GPU")
else:
    device = "cpu"
    print(f"⚠️  Using CPU (slower)")

# Test image path
test_image = "TripoSR/examples/chair.png"
if not os.path.exists(test_image):
    print(f"\n⚠️  Test image not found: {test_image}")
    print("Downloading sample image...")
    import urllib.request
    os.makedirs("TripoSR/examples", exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/VAST-AI-Research/TripoSR/main/examples/chair.png",
        test_image
    )
    print("✅ Downloaded test image")

print(f"\n📸 Loading test image: {test_image}")
image = Image.open(test_image).convert("RGB")
print(f"   Image size: {image.size}")

print("\n🤖 Loading TripoSR model (this may take a minute on first run)...")
try:
    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model.to(device)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("\nThis might be a network issue. The model downloads from HuggingFace.")
    sys.exit(1)

print("\n🎨 Generating 3D model...")
try:
    with torch.no_grad():
        scene_codes = model([image], device=device)
    print("✅ 3D generation complete!")
except Exception as e:
    print(f"❌ Error during generation: {e}")
    sys.exit(1)

print("\n📦 Extracting mesh...")
try:
    os.makedirs("output", exist_ok=True)
    meshes = model.extract_mesh(scene_codes, resolution=256)
    output_path = "output/test_mesh.obj"
    meshes[0].export(output_path)
    print(f"✅ Mesh saved to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024:.1f} KB")
except Exception as e:
    print(f"❌ Error extracting mesh: {e}")
    sys.exit(1)

print("\n🎉 TripoSR is working correctly on your Mac!")
print(f"\nNext steps:")
print(f"1. Try with your own images: python TripoSR/run.py your_image.jpg --output-dir output/")
print(f"2. Integrate into your backend using the code in TRIPOSR_LOCAL_SETUP.md")

