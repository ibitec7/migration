#!/bin/bash

# Download Google Trends data from Hugging Face Hub
# Usage: ./download_trends.sh
#
# This script downloads the entire trends dataset from the Hugging Face repository
# and saves it to data/raw/trends/ when run from the project root.

set -e

echo "Downloading Google Trends data from Hugging Face Hub..."
echo ""

# Get the directory where this script is located (project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"
TRENDS_DIR="${PROJECT_ROOT}/data/raw/trends"

echo "Project root: ${PROJECT_ROOT}"
echo "Virtual environment: ${VENV_DIR}"
echo "Target directory: ${TRENDS_DIR}"
echo ""

# Check if virtual environment exists
if [ ! -d "${VENV_DIR}" ]; then
    echo "Virtual environment not found at ${VENV_DIR}"
    echo "Please create a virtual environment first: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"

# Check if huggingface_hub is installed
if ! .venv/bin/python -c "import huggingface_hub" 2>/dev/null; then
    echo "Installing huggingface-hub..."
    uv pip install --upgrade huggingface-hub
fi

echo ""
echo "Starting download..."
echo "Repository: sdsc2005-migration/trends"
echo ""

# Run the Python script
cd "${PROJECT_ROOT}"
.venv/bin/python -m src.collection.trends

echo ""
echo "Download complete!"
