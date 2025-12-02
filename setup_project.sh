#!/usr/bin/env bash
set -euo pipefail

echo "Setting up ProteoForge Analysis environment"
echo "==========================================="

# Verify project root
if [[ ! -f "requirements.txt" ]]; then
    echo "Error: Run this script from the project root containing requirements.txt"
    exit 1
fi

# Detect platform (informational only)
UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
case "$UNAME_S" in
    Linux)   OS_NAME="linux" ;;
    Darwin)  OS_NAME="macos" ;;
    *)       OS_NAME="other" ;;
esac
echo "Detected platform: $OS_NAME"

# Resolve python command
if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_CMD="python"
else
    echo "Error: Python is not installed or not on PATH. Install Python 3.10+ and retry."
    exit 1
fi

# Python environment
if [[ ! -d ".venv" ]]; then
    echo "Creating Python virtual environment (.venv)"
    "$PYTHON_CMD" -m venv .venv
fi

VENV_BIN=".venv/bin"
if [[ ! -x "$VENV_BIN/python" ]]; then
    echo "Error: virtual environment missing python at $VENV_BIN/python"
    exit 1
fi

echo "Installing Python packages from requirements.txt"
"$VENV_BIN/python" -m pip install --upgrade pip
"$VENV_BIN/python" -m pip install -r requirements.txt

# R environment
if command -v Rscript >/dev/null 2>&1; then
    echo "Configuring R environment via setup_env.R (uses renv + pak)"
    Rscript setup_env.R
elif command -v R >/dev/null 2>&1; then
    echo "Configuring R environment via setup_env.R (fallback to R)"
    R --slave -e "source('setup_env.R')"
else
    echo "Warning: R is not installed or not on PATH. Skipping R environment setup."
    echo "Install R (>= 4.5.0), then run: Rscript setup_env.R"
fi

echo ""
echo "Setup complete"
echo "--------------"
echo "Python venv: .venv/"
echo "R env (renv): renv/ (if R was available)"
echo ""
echo "Activate Python (Linux/macOS): source .venv/bin/activate"
echo "Activate R: start R in this directory (renv auto-activates)"
