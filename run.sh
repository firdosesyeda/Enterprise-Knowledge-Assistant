#!/bin/bash
# run.sh - Start the Enterprise Knowledge Assistant

set -e

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   Enterprise Knowledge Assistant - Startup  ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
  echo "❌ Python 3 not found. Install from python.org"
  exit 1
fi

cd backend

# Check .env
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "⚠️  Created .env from template."
    echo "   → Open backend/.env and set your ANTHROPIC_API_KEY"
    echo "   → Then run this script again"
    exit 1
  fi
fi

# Check API key
source .env 2>/dev/null || true
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
  echo "❌ GEMINI_API_KEY is not set in backend/.env"
  echo "   Get your FREE key at: https://aistudio.google.com/app/apikey"
  exit 1
fi

# Create virtual env if needed
if [ ! -d "venv" ]; then
  echo "🔧 Creating virtual environment..."
  python3 -m venv venv
fi

# Activate
source venv/bin/activate

# Install deps
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo "🚀 Starting backend on http://localhost:8000"
echo "📚 API docs at http://localhost:8000/docs"
echo ""
echo "   Open frontend/index.html in your browser to use the UI"
echo "   Press Ctrl+C to stop"
echo ""

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
