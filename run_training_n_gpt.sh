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

# Download the Fineweb dataset only if needed
echo "=== Checking Fineweb dataset ==="
if [ ! -d "resources/edu_fineweb/train" ] || [ ! -d "resources/edu_fineweb/valid" ] || [ -z "$(ls -A resources/edu_fineweb/train)" ] || [ -z "$(ls -A resources/edu_fineweb/valid)" ]; then
    echo "Dataset not found or empty. Downloading Fineweb dataset..."
    python3 src/download_fineweb_dataset.py
    if [ $? -ne 0 ]; then
        echo "Error downloading dataset"
        exit 1
    fi
else
    echo "Dataset already exists, skipping download"
fi

# Set environment variables
echo "=== Setting environment variables ==="
export ARCHITECTURE="TRANSFORMER_PLUS"
export VALIDATION_PER_STEPS=500
export HELLSWAG_STEPS=500
export SAVE_STEPS=500
export USE_LIGER=True
export MICRO_BATCH_SIZE=64
export SEQUENCE_LENGTH=1024
export TOTAL_BATCH_SIZE=524288
export LEARNING_RATE=3e-4
export MIN_LR=3e-5
export WARMUP_STEPS=0 # according to the paper
export WEIGHT_DECAY=0 # according to the paper
export EPSILON=1e-8
export BETAS1=0.9
export BETA2=0.99
export TOTAL_STEPS=19073
export PRINT_STEPS=50
export EPOCHS=1
export SAVE_ON_LAST=True

# Print all training environment variables
echo "=== Training Environment Variables ==="
echo "ARCHITECTURE: $ARCHITECTURE"
echo "VALIDATION_PER_STEPS: $VALIDATION_PER_STEPS"
echo "HELLSWAG_STEPS: $HELLSWAG_STEPS"
echo "SAVE_STEPS: $SAVE_STEPS"
echo "USE_LIGER: $USE_LIGER"
echo "MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "SEQUENCE_LENGTH: $SEQUENCE_LENGTH"
echo "TOTAL_BATCH_SIZE: $TOTAL_BATCH_SIZE"
echo "LEARNING_RATE: $LEARNING_RATE"
echo "MIN_LR: $MIN_LR"
echo "WARMUP_STEPS: $WARMUP_STEPS"
echo "WEIGHT_DECAY: $WEIGHT_DECAY"
echo "EPSILON: $EPSILON"
echo "BETAS1: $BETAS1"
echo "BETA2: $BETA2"
echo "TOTAL_STEPS: $TOTAL_STEPS"
echo "PRINT_STEPS: $PRINT_STEPS"
echo "EPOCHS: $EPOCHS"
echo "SAVE_ON_LAST: $SAVE_ON_LAST"
echo "MIN_LR: $MIN_LR"

# Start training
echo "=== Starting training ==="
torchrun --standalone --nproc_per_node=auto src/train_gpt_2.py

echo "=== Process completed ==="