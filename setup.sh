#!/bin/bash
# Setup script for crypto trading data fetcher

echo "🚀 Setting up Crypto Trading Data Fetcher..."

# Create virtual environment (optional)
if command -v python3 &> /dev/null; then
    echo "✅ Python3 found"
else
    echo "❌ Python3 not found. Please install Python3 first."
    exit 1
fi

# Create directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed logs config

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Test imports
echo "🧪 Testing imports..."
python3 -c "import requests, websocket, yaml, pandas; print('✅ All imports successful')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start fetching data:"
echo "  python src/main.py"
echo ""
echo "To use WebSocket mode:"
echo "  python src/main.py --websocket"
echo ""
