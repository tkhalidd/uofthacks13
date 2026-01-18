# Vultr Backend Setup

Quick setup for your Vultr server to run the backend API.

## 🚀 Quick Start

### 1. SSH into Vultr Server

```bash
ssh root@155.138.214.166
```

### 2. Run Setup Script

```bash
# Upload setup.sh to server, then:
chmod +x setup.sh
./setup.sh
```

Or manually:

```bash
# Update system
apt update && apt upgrade -y

# Install Python
apt install -y python3.11 python3.11-venv python3-pip git

# Create project
mkdir -p /opt/room-redesign
cd /opt/room-redesign
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
pip install flask flask-cors requests python-dotenv gunicorn
```

### 3. Copy Files

Upload `app.py` to `/opt/room-redesign/`

### 4. Configure

```bash
cd /opt/room-redesign
source venv/bin/activate

# Create .env file
cat > .env << EOF
TRIPOSR_SERVER_URL=http://your-runpod-ip:8000
FLASK_ENV=production
EOF

# Edit with your RunPod IP
nano .env
```

### 5. Test Connection

```bash
# Test TripoSR connection
python -c "
from app import app
with app.test_client() as client:
    r = client.get('/api/test-triposr')
    print(r.json)
"
```

### 6. Run Server

```bash
# Development
python app.py

# Production (with gunicorn)
gunicorn -w 2 -b 0.0.0.0:5000 app:app
```

### 7. Set Up Firewall

```bash
ufw allow 22/tcp   # SSH
ufw allow 5000/tcp # Flask API
ufw enable
```

## 📡 API Endpoints

- `GET /health` - Health check
- `POST /api/upload-room` - Upload room photos
- `POST /api/generate-3d` - Generate 3D model (calls RunPod)
- `GET /api/test-triposr` - Test TripoSR connection

## 🔧 Configuration

Set environment variables in `.env`:

```bash
TRIPOSR_SERVER_URL=http://your-runpod-ip:8000
FLASK_ENV=production
```

## 🚀 Production Deployment

Use gunicorn with systemd:

```bash
# Create systemd service
cat > /etc/systemd/system/room-redesign.service << EOF
[Unit]
Description=Room Redesign API
After=network.target

[Service]
User=root
WorkingDirectory=/opt/room-redesign
Environment="PATH=/opt/room-redesign/venv/bin"
ExecStart=/opt/room-redesign/venv/bin/gunicorn -w 2 -b 127.0.0.1:5000 app:app

[Install]
WantedBy=multi-user.target
EOF

# Start service
systemctl daemon-reload
systemctl enable room-redesign
systemctl start room-redesign
```

## 📝 Notes

- Vultr server has no GPU, so it's just the API
- All 3D generation happens on RunPod
- This is the recommended setup (fast + cheap)

