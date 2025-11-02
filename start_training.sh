#!/bin/bash

echo "======================================"
echo "Agriculture Disease Detection"
echo "Training Pipeline"
echo "======================================"

# Activate virtual environment
source venv/bin/activate

# Download datasets
echo ""
echo "[Step 1/4] Downloading datasets..."
python scripts/01_download_datasets.py

# Prepare data
echo ""
echo "[Step 2/4] Preparing data..."
python scripts/02_prepare_data.py

# Train Stage 1
echo ""
echo "[Step 3/4] Training Stage 1 (Crop Classification)..."
python scripts/03_train_stage1.py

# Train Stage 2
echo ""
echo "[Step 4/4] Training Stage 2 (Disease Detection)..."
python scripts/04_train_stage2.py

echo ""
echo "======================================"
echo "TRAINING COMPLETED!"
echo "======================================"
echo ""
echo "Start API: uvicorn app.main:app --reload"
