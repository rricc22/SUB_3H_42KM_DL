#!/bin/bash
# Test script for Lag-Llama inference

echo "=========================================="
echo "Testing Lag-Llama Inference Scripts"
echo "=========================================="
echo ""

echo "1. Running quick inference (metrics only)..."
python3 Inferences/inference.py \
  --checkpoint checkpoints/lag_llama_best.pt \
  --data DATA/processed/test.pt \
  --device cpu \
  --batch_size 16

echo ""
echo "2. Checking if evaluation visualization exists..."
if [ -f "results/lag_llama_test_evaluation.png" ]; then
    echo "✅ Evaluation visualization already generated: results/lag_llama_test_evaluation.png"
else
    echo "⚠️  Evaluation visualization not found. Generate it with:"
    echo "   python3 Inferences/evaluate_test.py --checkpoint checkpoints/lag_llama_best.pt --data DATA/processed/test.pt --device cpu --output results/lag_llama_test_evaluation.png"
fi

echo ""
echo "=========================================="
echo "✅ Testing Complete!"
echo "=========================================="
