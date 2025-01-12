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

:: Start training
echo === Starting training ===
python src/train_gpt_2.py

echo === Process completed ===
pause