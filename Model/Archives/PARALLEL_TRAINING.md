# Parallel Training Guide

## Running Multiple Training Sessions Simultaneously

With the new config string feature, you can train multiple models in parallel without file conflicts!

---

## Example: Test Different Batch Sizes for Lag-Llama

Based on the analysis of `training_lag_llama_20251124_114157.log`, we suspect batch size 256 caused poor R² scores. Let's test smaller batch sizes:

### Terminal 1: Batch Size 32
```bash
nohup python3 Model/train.py \
  --model lag_llama \
  --batch_size 32 \
  --epochs 50 \
  --lr 0.0001 \
  > training_lag_llama_bs32.log 2>&1 &
```

**Output files:**
- `checkpoints/lag_llama_bs32_lr0.0001_e50_h64_l2_emb16_best.pt`
- `checkpoints/lag_llama_bs32_lr0.0001_e50_h64_l2_emb16_training_curves.png`

---

### Terminal 2: Batch Size 64
```bash
nohup python3 Model/train.py \
  --model lag_llama \
  --batch_size 64 \
  --epochs 50 \
  --lr 0.0001 \
  > training_lag_llama_bs64.log 2>&1 &
```

**Output files:**
- `checkpoints/lag_llama_bs64_lr0.0001_e50_h64_l2_emb16_best.pt`
- `checkpoints/lag_llama_bs64_lr0.0001_e50_h64_l2_emb16_training_curves.png`

---

### Terminal 3: Batch Size 128
```bash
nohup python3 Model/train.py \
  --model lag_llama \
  --batch_size 128 \
  --epochs 50 \
  --lr 0.0001 \
  > training_lag_llama_bs128.log 2>&1 &
```

**Output files:**
- `checkpoints/lag_llama_bs128_lr0.0001_e50_h64_l2_emb16_best.pt`
- `checkpoints/lag_llama_bs128_lr0.0001_e50_h64_l2_emb16_training_curves.png`

---

## Check Running Processes

```bash
# View all training processes
ps aux | grep train.py

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check log outputs
tail -f training_lag_llama_bs32.log
tail -f training_lag_llama_bs64.log
tail -f training_lag_llama_bs128.log
```

---

## Compare Results After Training

### Quick Comparison
```bash
# Compare test MAE across all runs
echo "=== Batch Size 32 ==="
grep "MAE:" training_lag_llama_bs32.log | tail -1

echo "=== Batch Size 64 ==="
grep "MAE:" training_lag_llama_bs64.log | tail -1

echo "=== Batch Size 128 ==="
grep "MAE:" training_lag_llama_bs128.log | tail -1

# Compare R² scores
echo "=== R² Scores ==="
grep "R²:" training_lag_llama_bs32.log training_lag_llama_bs64.log training_lag_llama_bs128.log
```

### Detailed Comparison
```bash
# Extract final metrics from all logs
for bs in 32 64 128; do
  echo "========================================"
  echo "Batch Size: $bs"
  echo "========================================"
  grep -A 4 "EVALUATING ON TEST SET" training_lag_llama_bs${bs}.log | tail -5
  echo ""
done
```

---

## Advanced: Test Multiple Hyperparameters

### Learning Rate Grid Search
```bash
# Low learning rate
nohup python3 Model/train.py --model lag_llama --batch_size 32 --lr 0.00005 --epochs 50 > training_lag_llama_bs32_lr5e5.log 2>&1 &

# Medium learning rate (default)
nohup python3 Model/train.py --model lag_llama --batch_size 32 --lr 0.0001 --epochs 50 > training_lag_llama_bs32_lr1e4.log 2>&1 &

# Higher learning rate
nohup python3 Model/train.py --model lag_llama --batch_size 32 --lr 0.0002 --epochs 50 > training_lag_llama_bs32_lr2e4.log 2>&1 &
```

### Architecture Variations
```bash
# Shallow model (2 layers)
nohup python3 Model/train.py --model lag_llama --batch_size 32 --num_layers 2 --epochs 50 > training_lag_llama_l2.log 2>&1 &

# Deep model (4 layers)
nohup python3 Model/train.py --model lag_llama --batch_size 32 --num_layers 4 --epochs 50 > training_lag_llama_l4.log 2>&1 &

# Very deep model (6 layers)
nohup python3 Model/train.py --model lag_llama --batch_size 32 --num_layers 6 --epochs 50 > training_lag_llama_l6.log 2>&1 &
```

### Dropout Regularization Test
```bash
# No dropout
nohup python3 Model/train.py --model lag_llama --batch_size 32 --dropout 0.0 --epochs 50 > training_lag_llama_drop0.log 2>&1 &

# Light dropout (default)
nohup python3 Model/train.py --model lag_llama --batch_size 32 --dropout 0.1 --epochs 50 > training_lag_llama_drop1.log 2>&1 &

# Heavy dropout
nohup python3 Model/train.py --model lag_llama --batch_size 32 --dropout 0.3 --epochs 50 > training_lag_llama_drop3.log 2>&1 &
```

---

## Example: Full Grid Search (9 combinations)

```bash
# Test 3 batch sizes × 3 learning rates = 9 parallel runs
for bs in 32 64 128; do
  for lr in 0.00005 0.0001 0.0002; do
    nohup python3 Model/train.py \
      --model lag_llama \
      --batch_size $bs \
      --lr $lr \
      --epochs 50 \
      > training_lag_llama_bs${bs}_lr${lr}.log 2>&1 &
    
    echo "Started training: batch_size=$bs, lr=$lr"
    sleep 2  # Stagger start times
  done
done

echo "All 9 training sessions started!"
```

**Warning:** This will consume significant GPU memory. Monitor with `nvidia-smi`.

---

## GPU Memory Management

### Estimate Memory Usage
- **Batch Size 32:** ~2-3 GB VRAM
- **Batch Size 64:** ~3-4 GB VRAM  
- **Batch Size 128:** ~4-5 GB VRAM
- **Batch Size 256:** ~5-6 GB VRAM

**Your GPU:** GTX 1060 6GB → Can run 2-3 parallel trainings with small batches

### Safe Parallel Training
```bash
# Run 2 trainings simultaneously (safe for 6GB GPU)
nohup python3 Model/train.py --model lag_llama --batch_size 32 --epochs 50 > training_bs32.log 2>&1 &
sleep 10  # Wait for first to allocate memory
nohup python3 Model/train.py --model lag_llama --batch_size 64 --epochs 50 > training_bs64.log 2>&1 &
```

---

## Organizing Results

### Create Results Directory
```bash
mkdir -p results/batch_size_experiment
mv training_lag_llama_bs*.log results/batch_size_experiment/
cp checkpoints/lag_llama_bs*_best.pt results/batch_size_experiment/
cp checkpoints/lag_llama_bs*_curves.png results/batch_size_experiment/
```

### Generate Comparison Report
```bash
# Create a simple comparison script
cat > results/batch_size_experiment/compare.sh << 'EOF'
#!/bin/bash
echo "===========================================" 
echo "Lag-Llama Batch Size Comparison"
echo "==========================================="
echo ""

for log in training_lag_llama_bs*.log; do
  bs=$(echo $log | grep -oP 'bs\K[0-9]+')
  echo "Batch Size: $bs"
  echo "---"
  grep "Best Val MAE:" $log
  grep -A 4 "Test Metrics:" $log | grep -E "MAE:|R²:"
  echo ""
done
EOF

chmod +x results/batch_size_experiment/compare.sh
./results/batch_size_experiment/compare.sh
```

---

## Best Practices

1. **Stagger starts:** Wait 5-10 seconds between launching parallel jobs
2. **Monitor GPU:** Use `watch -n 1 nvidia-smi` in a separate terminal
3. **Use nohup:** Prevents training from stopping if terminal closes
4. **Redirect logs:** Always use `> training.log 2>&1` to capture all output
5. **Name logs clearly:** Include all varying hyperparameters in log filename
6. **Check disk space:** Training generates large checkpoints (5-30 MB each)

---

## Troubleshooting

### "CUDA Out of Memory" Error
```bash
# Kill all training processes
pkill -f train.py

# Clear GPU cache
python3 -c "import torch; torch.cuda.empty_cache()"

# Reduce batch size or run fewer parallel jobs
```

### Training Stuck
```bash
# Check if process is running
ps aux | grep train.py

# Check GPU utilization (should be 90-100%)
nvidia-smi

# If stuck, check log for errors
tail -50 training_lag_llama_bs32.log
```

### Accidentally Overwriting Files
**No longer possible!** Config strings prevent overwriting. But if you want to rerun with same config:
```bash
# Rename old checkpoint before retraining
mv checkpoints/lag_llama_bs32_lr0.0001_e50_h64_l2_emb16_best.pt \
   checkpoints/lag_llama_bs32_lr0.0001_e50_h64_l2_emb16_best_OLD.pt
```

---

## Quick Reference: Kill All Training

```bash
# Stop all training processes
pkill -f "python3 Model/train.py"

# Verify they stopped
ps aux | grep train.py

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"
```
