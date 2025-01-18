@echo off
echo === Starting GPT-2 Training Setup ===

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Python is not installed. Please install Python first.
    exit /b 1
)

:: Create and activate virtual environment
echo === Setting up virtual environment ===
python -m venv venv
call venv\Scripts\activate

:: Install requirements
echo === Installing requirements ===
pip install -r requirements.txt
pip install -e .
:: Login to Weights & Biases
echo === Logging into Weights & Biases ===
wandb login
if errorlevel 1 (
    echo Error logging into Weights ^& Biases
    exit /b 1
)

:: Download the Fineweb dataset
echo === Downloading Fineweb dataset ===
python src/download_fineweb_dataset.py

:: Set environment variables
echo === Setting environment variables ===
set VALIDATION_PER_STEPS=500
set HELLSWAG_STEPS=500
set SAVE_STEPS=500
set USE_LIGER=True
set MICRO_BATCH_SIZE=64
set SEQUENCE_LENGTH=1024
set TOTAL_BATCH_SIZE=524288
set LEARNING_RATE=6e-4
set WARMUP_STEPS=715
set WEIGHT_DECAY=0.1
set EPSILON=1e-8
set BETAS1=0.9
set BETA2=0.95
set TOTAL_STEPS=19073
set PRINT_STEPS=50
set EPOCHS=1
set SAVE_ON_LAST=True
set MIN_LR=6e-5

:: Start training
echo === Starting training ===
torchrun --standalone --nproc_per_node=auto src/train_gpt_2.py

echo === Process completed ===
pause