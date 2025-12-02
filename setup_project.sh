#!/bin/bash

echo "🧬 Setting up Proteoforms Project Environment"
echo "=============================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the Proteoforms project root directory"
    exit 1
fi

# Python Environment Setup
if [ ! -d ".venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv .venv
fi
echo "🔧 Activating Python virtual environment..."
source .venv/bin/activate
echo "📦 Installing Python packages from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# R Environment Setup
if ! command -v R &> /dev/null; then
    echo "❌ Error: R is not installed. Please install R first."
    exit 1
fi
echo "🔧 Setting up R environment..."
echo "⚠️  This will install CCprofiler and PeCorA from GitHub (requires internet)"
echo "⚠️  This will open R to initialize renv. Please wait..."
R --slave -e "source('setup_env.R')"
# Setup renv should be separate script then another script installing other packages after R restarts
# Before checks if renv is already initialized
# Then installs packages with 

echo ""
echo "✅ SETUP COMPLETE!"
echo "=================="
echo ""
echo "🐍 Python environment: .venv/"
echo "📊 R environment: renv/"
echo ""
echo "To activate environments:"
echo "  Python: source .venv/bin/activate"
echo "  R: Open R in this directory (automatic via .Rprofile)"
echo ""
echo "📖 See R_ENVIRONMENT.md for detailed R package management"
echo "📄 See requirements.txt for Python package information"
echo ""
echo "🚀 You're ready to start analyzing proteomics data!"
