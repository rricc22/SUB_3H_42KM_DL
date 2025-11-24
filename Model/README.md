# Model

LSTM models for heart rate prediction from speed/altitude sequences.

## Model Architectures

### `LSTM.py`
Basic LSTM model (no user personalization).

**Architecture:** Speed + Altitude + Gender → LSTM → Heart Rate

### `LSTM_with_embeddings.py`
LSTM with user embeddings for personalization.

**Architecture:** Speed + Altitude + Gender + UserID → LSTM + Embeddings → Heart Rate

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

nohup python3 -u Model/train.py \
  --model lstm \
  --hidden_size 256 \
  --num_layers 3 \
  --dropout 0.3 \
  --epochs 2 \
  --batch_size 64 \
  --lr 0.001 \
  --patience 25 \
  --device cuda \
  > training_overnight_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "Process ID: $!"

**Output:**
- Checkpoint: `checkpoints/{model}_best.pt`
- Training curves: `checkpoints/{model}_training_curves.png`

---

## Current Results

**Basic LSTM (64 hidden, 2 layers):**
- MAE: 13.23 BPM ❌ (target: < 10 BPM)
- R²: 0.048 (underfitting)

**Expected with larger models:**
- Large LSTM: MAE ~8-10 BPM
- LSTM + Embeddings: MAE ~7-9 BPM

---

## Key Arguments

- `--model`: `lstm` or `lstm_embeddings`
- `--hidden_size`: LSTM hidden size (64, 128, 256)
- `--num_layers`: Number of LSTM layers (2, 3, 4)
- `--dropout`: Dropout rate (0.0-0.5)
- `--embedding_dim`: User embedding dimension (32, 64) [embeddings only]
- `--device`: `cpu` or `cuda` (use `cpu` to avoid cuDNN issues)
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 32)

---

## Dataset

- Train: 6,368 samples | Val: 1,463 | Test: 1,031
- Sequence length: 500 timesteps
- Features: speed (normalized), altitude (normalized), gender (binary)
- Target: heart_rate (BPM, NOT normalized)
