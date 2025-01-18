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
pip install -e .
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
# Set environment variables
echo "=== Setting environment variables ==="
export VALIDATION_PER_STEPS=500
export HELLSWAG_STEPS=500
export SAVE_STEPS=500
export USE_LIGER=True
export MICRO_BATCH_SIZE=64
export SEQUENCE_LENGTH=1024
export TOTAL_BATCH_SIZE=524288
export LEARNING_RATE=6e-4
export WARMUP_STEPS=715
export WEIGHT_DECAY=0.1
export EPSILON=1e-8
export BETAS1=0.9
export BETA2=0.95
export TOTAL_STEPS=19073
export PRINT_STEPS=50
export EPOCHS=1
export SAVE_ON_LAST=True
export MIN_LR=6e-5

# Start training
echo "=== Starting training ==="
torchrun --standalone --nproc_per_node=auto src/train_gpt_2.py

echo "=== Process completed ==="