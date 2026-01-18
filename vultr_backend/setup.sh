#!/bin/bash
# Quick setup script for Vultr server

set -e

echo "🚀 Setting up Room Redesign Backend on Vultr..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Python 3.11
echo "🐍 Installing Python 3.11..."
apt install -y python3.11 python3.11-venv python3-pip

# Install other tools
echo "🔧 Installing tools..."
apt install -y git curl nginx ufw

# Create project directory
echo "📁 Creating project directory..."
mkdir -p /opt/room-redesign
cd /opt/room-redesign

# Create virtual environment
echo "🔨 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📚 Installing Python packages..."
pip install --upgrade pip
pip install flask flask-cors requests python-dotenv gunicorn

# Create .env file
echo "⚙️ Creating .env file..."
cat > .env << 'EOF'
# Set your RunPod TripoSR server URL here
TRIPOSR_SERVER_URL=http://your-runpod-ip:8000
FLASK_ENV=production
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📝 Next steps:"
echo "   1. Edit .env and set TRIPOSR_SERVER_URL to your RunPod IP"
echo "   2. Copy app.py to /opt/room-redesign/"
echo "   3. Run: cd /opt/room-redesign && source venv/bin/activate && python app.py"
echo ""
echo "🔒 Don't forget to set up firewall:"
echo "   ufw allow 22/tcp   # SSH"
echo "   ufw allow 5000/tcp # Flask API"
echo "   ufw enable"

