# Model

Deep learning models for heart rate prediction from speed/altitude sequences.

## Model Architectures

### `LSTM.py`
Basic LSTM model (no user personalization).

**Architecture:** Speed + Altitude + Gender → LSTM → Heart Rate  
**Parameters:** ~50K

### `LSTM_with_embeddings.py`
LSTM with user embeddings for personalization.

**Architecture:** Speed + Altitude + Gender + UserID → LSTM + Embeddings → Heart Rate  
**Parameters:** ~60K

### `LagLlama_HR.py` ⭐ NEW
Transformer-based model inspired by Lag-Llama (time-series foundation model).

**Architecture:** 
```
Input: [speed, altitude, gender, user_embedding] → [batch, 500, 19]
  ↓
Input Projection: Linear(19 → 128)
  ↓
Positional Encoding: Sinusoidal position embeddings
  ↓
Transformer Encoder (4 layers):
  - Multi-Head Attention (8 heads)
  - Feed-Forward Network (512 hidden)
  - Layer Normalization
  - Dropout (0.1)
  ↓
Output Projection: MLP(128 → 64 → 1)
  ↓
Heart Rate: [batch, 500, 1]
```

**Parameters:** ~805K (16x larger than LSTM)

**Key Differences from LSTM:**
- **Attention Mechanism:** Sees entire sequence simultaneously (vs sequential processing)
- **Positional Encoding:** Explicit temporal information (LSTM has implicit ordering)
- **Multi-Head Attention:** 8 parallel attention heads learn different patterns
- **GELU Activation:** Smoother gradients than ReLU
- **Higher Capacity:** Can capture long-range dependencies better

**Why Lag-Llama?**
- Better for long sequences (500 timesteps)
- Captures delayed physiological responses (e.g., altitude at t=100 affects HR at t=150)
- Learns complex interactions between features via attention

---

## Training

### `train.py` ⭐
Main training script with cuDNN disabled (GTX 1060 compatibility).

**Basic LSTM (small):**
```bash
python3 Model/train.py \
  --model lstm \
  --hidden_size 64 \
  --num_layers 2 \
  --epochs 100 \
  --batch_size 32 \
  --device cpu
```

**Large LSTM (better performance):**
```bash
python3 Model/train.py \
  --model lstm \
  --hidden_size 256 \
  --num_layers 4 \
  --dropout 0.2 \
  --epochs 10 \
  --batch_size 64 \
  --lr 0.005
  --device cuda
```

**LSTM with User Embeddings:**
```bash
python3 Model/train.py \
  --model lstm_embeddings \
  --hidden_size 128 \
  --num_layers 3 \
  --embedding_dim 32 \
  --dropout 0.2 \
  --epochs 100 \
  --batch_size 32 \
  --device cpu
```

**Lag-Llama Transformer (Recommended):** ⭐
```bash
# CORRECT hyperparameters for Transformer:
python3 Model/train.py \
  --model lag_llama \
  --num_layers 4 \
  --embedding_dim 16 \
  --dropout 0.1 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --device cuda
```

** IMPORTANT - Lag-Llama Training Tips:**
1. **Lower batch size (16):** Transformers use more memory
2. **Lower learning rate (0.0001):** Attention is sensitive to LR
3. **More epochs (100):** Transformers converge slower than LSTM
4. **More layers (4):** Deeper = better for Transformers

**Common Mistakes:**
```bash
#  WRONG - These settings give BAD performance (MAE ~38 BPM):
python3 Model/train.py --model lag_llama --epochs 10 --batch_size 128 --lr 0.001 --num_layers 2

#  CORRECT - Proper settings (Expected MAE: 6-9 BPM):
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001 --num_layers 4
```

**Overnight Training:**
```bash
nohup python3 -u Model/train.py \
  --model lag_llama \
  --num_layers 4 \
  --dropout 0.1 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0001 \
  --patience 15 \
  --device cuda \
  > training_lag_llama_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Process ID: $!"
```

nohup python3 -u Model/train_patchtst.py \
    --epochs 50 \
    --batch_size 64 \
    --d_model 256 \
    --num_hidden_layers 6 \
    --num_attention_heads 16 \
    --ffn_dim 512 \
    --patch_length 16 \
    --stride 8 \
    --device cuda \
  > LOGS/training_patchtst_large_$(date +%Y%m%d_%H%M%S).log 2>/dev/null &
echo "Training started! PID: $!"
echo "Monitor with: tail -f LOGS/training_patchtst_large_*.log"
echo "Check GPU: watch -n 5 nvidia-smi"

nohup python3 -u Model/train.py \
  --model lstm \
  --hidden_size 256 \
  --num_layers 5 \
  --dropout 0.2 \
  --epochs 100 \
  --batch_size 128 \
  --lr 0.0005 \
  --device cuda \
  > LOGS/training_lstm_large_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Training started! PID: $!"
echo "Monitor with: tail -f LOGS/training_lstm_large_*.log"
echo "Check GPU: watch -n 5 nvidia-smi"

nohup python3 -u Model/train.py \
  --model lstm \
  --hidden_size 768 \
  --num_layers 5 \
  --dropout 0.35 \
  --epochs 100 \
  --batch_size 16 \
  --lr 0.0002 \
  --bidirectional \
  --device cuda \
  > LOGS/training_lstm_xlarge_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Training started! PID: $!"
echo "Monitor with: tail -f LOGS/training_lstm_xlarge_*.log"
echo "Check GPU: watch -n 5 nvidia-smi"

**Output:**
- Checkpoint: `checkpoints/{model}_best.pt`
- Training curves: `checkpoints/{model}_training_curves.png`

---

## Current Results

**Trained Models Performance:**

| Model | MAE (BPM) | RMSE (BPM) | R² | Parameters | Status |
|-------|-----------|------------|----|------------|--------|
| LSTM | 15.41 | 20.11 | -0.052 | ~50K |  Trained |
| LSTM+Embeddings | 15.79 | 20.61 | -0.106 | ~60K |  Trained |
| **Lag-Llama** | **38.78**  | **64.72** | **-9.41** | **~805K** |  **Bad Hyperparams** |

**Lag-Llama Issues:**
- Trained with WRONG hyperparameters:
  - `epochs=10` (too short, should be 100)
  - `batch_size=128` (too large, should be 16)
  - `lr=0.001` (too high, should be 0.0001)
  - `num_layers=2` (too shallow, should be 4)

**Expected Performance (with correct hyperparameters):**
- LSTM: 8-10 BPM MAE
- LSTM+Embeddings: 7-9 BPM MAE
- **Lag-Llama (fixed): 6-9 BPM MAE** ⭐ (Best expected)

**Target:** MAE < 10 BPM (acceptable), < 5 BPM (excellent)

---

## Key Arguments

### Common Arguments
- `--model`: `lstm`, `lstm_embeddings`, or `lag_llama`
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32, **use 16 for lag_llama**)
- `--lr`: Learning rate (default: 0.001, **use 0.0001 for lag_llama**)
- `--device`: `cpu` or `cuda`
- `--patience`: Early stopping patience (default: 10)

### Model-Specific Arguments

**LSTM / LSTM+Embeddings:**
- `--hidden_size`: LSTM hidden size (64, 128, 256)
- `--num_layers`: Number of LSTM layers (2, 3, 4)
- `--dropout`: Dropout rate (0.0-0.5)
- `--embedding_dim`: User embedding dimension (16, 32, 64)
- `--bidirectional`: Use bidirectional LSTM (flag)

**Lag-Llama (Transformer):**
- `--num_layers`: Number of transformer layers (**recommended: 4**)
- `--dropout`: Dropout rate (**recommended: 0.1**)
- `--embedding_dim`: User embedding dimension (16, 32)
- Note: `hidden_size` argument exists but doesn't affect Lag-Llama (uses d_model=128 internally)

---

## Hyperparameter Recommendations by Model

| Hyperparameter | LSTM | LSTM+Embeddings | Lag-Llama |
|----------------|------|-----------------|-----------|
| **batch_size** | 32 | 32 | **16** ⬇️ |
| **lr** | 0.001 | 0.001 | **0.0001** ⬇️ |
| **num_layers** | 2-3 | 2-3 | **4** ⬆️ |
| **dropout** | 0.2 | 0.2 | **0.1** ⬇️ |
| **epochs** | 100 | 100 | **100** |
| **hidden_size** | 64-256 | 128-256 | N/A (uses d_model=128) |
| **embedding_dim** | N/A | 16-32 | 16 |

**Why different hyperparameters for Lag-Llama?**
- **Lower batch size:** Transformers use more GPU memory (805K vs 50K params)
- **Lower learning rate:** Attention mechanism is sensitive to large LR
- **More layers:** Transformers benefit from depth (4 layers > 2 layers)
- **Lower dropout:** Already regularized via attention dropout

---

## Dataset

- Train: 682 samples | Val: 146 | Test: 146
- Sequence length: 500 timesteps
- Features: speed (normalized), altitude (normalized), gender (binary), userId (embedded)
- Target: heart_rate (BPM, NOT normalized)

---

## Troubleshooting

### Lag-Llama Poor Performance (MAE > 30 BPM)
**Cause:** Wrong hyperparameters (batch_size too large, lr too high, epochs too short)

**Solution:**
```bash
# Re-train with correct settings:
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001 --num_layers 4
```

### GPU Out of Memory (OOM)
**For Lag-Llama:**
```bash
# Reduce batch size to 8
python3 Model/train.py --model lag_llama --batch_size 8 --lr 0.0001

# Or use CPU (slower but works)
python3 Model/train.py --model lag_llama --device cpu --batch_size 16
```

### Model Not Converging
**For Lag-Llama:**
```bash
# Lower learning rate further
python3 Model/train.py --model lag_llama --lr 0.00005 --epochs 150

# Or reduce model complexity
python3 Model/train.py --model lag_llama --num_layers 2 --lr 0.0001
```

---

## Troubleshooting

### Lag-Llama Poor Performance (MAE > 30 BPM)
**Cause:** Wrong hyperparameters (batch_size too large, lr too high, epochs too short)

**Solution:**
```bash
# Re-train with correct settings:
python3 Model/train.py --model lag_llama --epochs 100 --batch_size 16 --lr 0.0001 --num_layers 4
```

### GPU Out of Memory (OOM)
**For Lag-Llama:**
```bash
# Reduce batch size to 8
python3 Model/train.py --model lag_llama --batch_size 8 --lr 0.0001

# Or use CPU (slower but works)
python3 Model/train.py --model lag_llama --device cpu --batch_size 16
```

### Model Not Converging
**For Lag-Llama:**
```bash
# Lower learning rate further
python3 Model/train.py --model lag_llama --lr 0.00005 --epochs 150

# Or reduce model complexity
python3 Model/train.py --model lag_llama --num_layers 2 --lr 0.0001
```
