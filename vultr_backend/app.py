"""
Room Redesign Backend API
Runs on Vultr, calls RunPod for TripoSR
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import requests
from pathlib import Path
import uuid
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configuration
TRIPOSR_SERVER_URL = os.getenv("TRIPOSR_SERVER_URL", "http://your-runpod-ip:8000")
UPLOAD_FOLDER = Path("/opt/room-redesign/uploads")
OUTPUT_FOLDER = Path("/opt/room-redesign/output")
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)

@app.route('/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        "status": "healthy",
        "server": "vultr",
        "triposr_url": TRIPOSR_SERVER_URL
    }), 200

@app.route('/api/upload-room', methods=['POST'])
def upload_room():
    """
    Upload room photos
    
    POST /api/upload-room
    Body: multipart/form-data
      - photos: image files (multiple)
    """
    if 'photos' not in request.files:
        return jsonify({"error": "No photos provided"}), 400
    
    files = request.files.getlist('photos')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        # Save file
        file_id = str(uuid.uuid4())
        file_ext = Path(file.filename).suffix
        file_path = UPLOAD_FOLDER / f"{file_id}{file_ext}"
        file.save(str(file_path))
        
        uploaded_files.append({
            "id": file_id,
            "filename": file.filename,
            "path": str(file_path)
        })
    
    return jsonify({
        "status": "success",
        "files": uploaded_files,
        "count": len(uploaded_files)
    }), 200

@app.route('/api/generate-3d', methods=['POST'])
def generate_3d():
    """
    Generate 3D model from furniture photo
    Calls RunPod TripoSR server
    
    POST /api/generate-3d
    Body: multipart/form-data
      - image: image file
      - device: optional (default: cuda:0)
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image']
    device = request.form.get('device', 'cuda:0')
    
    # Save image temporarily
    temp_path = UPLOAD_FOLDER / f"temp_{uuid.uuid4()}_{image_file.filename}"
    image_file.save(str(temp_path))
    
    try:
        # Forward to RunPod TripoSR
        with open(temp_path, 'rb') as f:
            files = {
                'image': (image_file.filename, f, 'image/jpeg')
            }
            data = {'device': device}
            
            response = requests.post(
                f"{TRIPOSR_SERVER_URL}/generate",
                files=files,
                data=data,
                timeout=120
            )
        
        # Clean up temp file
        temp_path.unlink()
        
        if response.status_code == 200:
            result = response.json()
            return jsonify(result), 200
        else:
            return jsonify({
                "error": "TripoSR generation failed",
                "status_code": response.status_code,
                "details": response.text
            }), 500
    
    except requests.exceptions.Timeout:
        return jsonify({"error": "Request timed out (120s)"}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": f"Failed to connect to TripoSR server",
            "details": str(e),
            "triposr_url": TRIPOSR_SERVER_URL
        }), 500
    except Exception as e:
        # Clean up on error
        if temp_path.exists():
            temp_path.unlink()
        return jsonify({"error": str(e)}), 500

@app.route('/api/test-triposr', methods=['GET'])
def test_triposr():
    """
    Test connection to TripoSR server
    """
    try:
        response = requests.get(
            f"{TRIPOSR_SERVER_URL}/health",
            timeout=5
        )
        
        if response.status_code == 200:
            return jsonify({
                "status": "connected",
                "triposr_health": response.json()
            }), 200
        else:
            return jsonify({
                "status": "error",
                "triposr_status_code": response.status_code
            }), 500
    
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "triposr_url": TRIPOSR_SERVER_URL
        }), 500

if __name__ == '__main__':
    print(f"🚀 Starting Room Redesign API on port 5000")
    print(f"📡 TripoSR server: {TRIPOSR_SERVER_URL}")
    app.run(host='0.0.0.0', port=5000, debug=False)

