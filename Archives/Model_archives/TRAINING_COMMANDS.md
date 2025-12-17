# Training Commands Cheat Sheet

## New Feature: Config String Filenames

**All checkpoints and plots now include config strings to prevent overwriting!**

Example: `lag_llama_bs32_lr0.0001_e50_h64_l2_emb16_best.pt`

This allows running multiple training sessions in parallel with different hyperparameters without file conflicts.

---

## Quick Start - All Models

### 1. Basic LSTM (Baseline)
```bash
python3 Model/train.py --model lstm --epochs 100 --batch_size 32
```
**Expected Performance:** 8-12 BPM MAE  
**Training Time:** ~5 minutes on GPU  
**Parameters:** ~50K

---

### 2. LSTM with User Embeddings
```bash
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32
```
**Expected Performance:** 7-10 BPM MAE  
**Training Time:** ~6 minutes on GPU  
**Parameters:** ~60K  
**Improvement:** Better personalization per user

---

### 3. Lag-Llama (Transformer) - Recommended
```bash
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001
```
**Expected Performance:** 6-9 BPM MAE ⭐  
**Training Time:** ~20-30 minutes on GPU  
**Parameters:** ~2M  
**Note:** Lower batch size (16) and learning rate (0.0001) due to larger model

---

## Advanced Training Options

### GPU Memory Issues (OOM Error)
If you get "out of memory" error with Lag-Llama:
```bash
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 8 --lr 0.0001
```

### Quick Test (Fast Training)
```bash
# Test all models quickly
python3 Model/train.py --model lstm --epochs 20 --batch_size 32
python3 Model/train.py --model lstm_embeddings --epochs 20 --batch_size 32
python3 Model/train.py --model lag_llama --epochs 20 --batch_size 16 --lr 0.0001
```

### Custom Architecture (Lag-Llama)
```bash
# Deeper model (6 layers instead of 4)
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001 --num_layers 6

# With more dropout for regularization
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001 --dropout 0.2

# Larger embedding dimension for users
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001 --embedding_dim 32
```

### Bidirectional LSTM (Better Context)
```bash
python3 Model/train.py --model lstm --epochs 100 --batch_size 32 --bidirectional
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32 --bidirectional
```

---

## Training Strategy

### Recommended Training Order
```bash
# Step 1: Train baseline to establish performance floor
python3 Model/train.py --model lstm --epochs 100 --batch_size 32

# Step 2: Add user embeddings for personalization
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32

# Step 3: Train Transformer for best performance
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001
```

### Compare Results
```bash
# Check all saved checkpoints
ls -lh checkpoints/

# Files created (with config strings to prevent overwriting):
# - lstm_bs32_lr0.001_e100_h64_l2_best.pt
# - lstm_embeddings_bs32_lr0.001_e100_h64_l2_emb16_best.pt
# - lag_llama_bs16_lr0.0001_e100_h64_l2_emb16_best.pt
# - lstm_bs32_lr0.001_e100_h64_l2_training_curves.png
# - lstm_embeddings_bs32_lr0.001_e100_h64_l2_emb16_training_curves.png
# - lag_llama_bs16_lr0.0001_e100_h64_l2_emb16_training_curves.png
```

**Config String Format:**
- `bs<batch_size>_lr<learning_rate>_e<epochs>_h<hidden_size>_l<num_layers>`
- Additional: `_emb<embedding_dim>` for lstm_embeddings/lag_llama
- Additional: `_bidir` for bidirectional models

---

## All Available Arguments

```bash
python3 Model/train.py \
  --model lstm                    # Model type: lstm, lstm_embeddings, lag_llama
  --epochs 100                    # Number of training epochs
  --batch_size 32                 # Batch size (use 16 for lag_llama)
  --lr 0.001                      # Learning rate (use 0.0001 for lag_llama)
  --patience 10                   # Early stopping patience
  --hidden_size 64                # LSTM hidden size
  --num_layers 2                  # Number of layers (LSTM or Transformer)
  --dropout 0.2                   # Dropout probability
  --bidirectional                 # Use bidirectional LSTM
  --embedding_dim 16              # User embedding dimension
  --data_dir DATA/processed       # Directory with preprocessed data
  --checkpoint_dir checkpoints    # Directory to save checkpoints
  --device cuda                   # Device: cuda or cpu
```

---

## Expected Results Comparison

| Model | Parameters | MAE (BPM) | RMSE (BPM) | Training Time |
|-------|-----------|-----------|------------|---------------|
| LSTM | ~50K | 8-12 | 10-15 | 5 min |
| LSTM+Embeddings | ~60K | 7-10 | 9-13 | 6 min |
| **Lag-Llama** | **~2M** | **6-9** ⭐ | **8-12** | **20-30 min** |

**Target:** MAE < 5 BPM (excellent), < 10 BPM (acceptable)

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size
python3 Model/train.py --model lag_llama --batch_size 8 --lr 0.0001

# Or use CPU (slower)
python3 Model/train.py --model lag_llama --device cpu --batch_size 16
```

### Issue: Model Not Converging
**Solution:**
```bash
# Lower learning rate
python3 Model/train.py --model lag_llama --lr 0.00005 --epochs 150

# Or try bidirectional LSTM
python3 Model/train.py --model lstm_embeddings --bidirectional --epochs 100
```

### Issue: Overfitting (Val Loss Increases)
**Solution:**
```bash
# Increase dropout
python3 Model/train.py --model lag_llama --dropout 0.3 --lr 0.0001

# Or reduce model complexity
python3 Model/train.py --model lag_llama --num_layers 2 --lr 0.0001
```

---

## Next Steps After Training

1. **Evaluate on test set** (automatically done by train.py)
2. **Visualize predictions:**
   ```bash
   python3 Inferences/inference.py --model lag_llama --checkpoint checkpoints/lag_llama_best.pt
   ```
3. **Compare models:**
   ```bash
   python3 Model/evaluate_test.py  # Compare all three models
   ```

---

## Questions?

- Check `Model/README.md` for model architecture details
- Check `AGENTS.md` for agent guidelines and common commands
- See training curves in `checkpoints/*.png` for training progress
