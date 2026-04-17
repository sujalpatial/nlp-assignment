#!/bin/bash
set -e

echo "🚀 Project Sentinel Setup"
echo "========================="

# Backend setup
echo "📦 Setting up backend..."
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd ..

# Frontend setup
echo "📦 Setting up frontend..."
cd client
npm install
cd ..

# Download model
echo "📥 Downloading model..."
python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('facebook/opt-1.3b'); AutoTokenizer.from_pretrained('facebook/opt-1.3b'); print('✅ Model downloaded')"

echo "✅ Setup complete!"
