#!/bin/bash
# GPT-2 多精度误差分析实验启动脚本

echo "Installing dependencies..."
pip install --upgrade pip setuptools wheel
pip install --only-binary :all: -r requirements.txt

echo ""
echo "Step 1: Generating test sentences..."
if [ ! -f "sentences.json" ]; then
    python3 generate_sentences.py
else
    echo "  sentences.json already exists, skipping generation"
fi

echo ""
echo "Step 2: Running experiment..."
python3 experiment.py

echo ""
echo "Experiment completed!"
echo "Results are saved in: results/{hardware_name}/"

