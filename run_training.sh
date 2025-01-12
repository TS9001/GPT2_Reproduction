#!/bin/bash

# Exit on any error
set -e

echo "=== Starting GPT-2 Training Setup ==="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 first."
    exit 1
fi

# Check if CUDA is available
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"
if [ $? -ne 0 ]; then
    echo "Error checking CUDA availability"
    exit 1
fi

# Create and activate virtual environment
echo "=== Setting up virtual environment ==="
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install requirements
echo "=== Installing requirements ==="
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing requirements"
    exit 1
fi

# Login to Weights & Biases
echo "=== Logging into Weights & Biases ==="
wandb login
if [ $? -ne 0 ]; then
    echo "Error logging into Weights & Biases"
    exit 1
fi

# Download the Fineweb dataset
echo "=== Downloading Fineweb dataset ==="
python3 src/download_fineweb_dataset.py
if [ $? -ne 0 ]; then
    echo "Error downloading dataset"
    exit 1
fi

# Start training
echo "=== Starting training ==="
python3 src/train_gpt_2.py

echo "=== Process completed ==="