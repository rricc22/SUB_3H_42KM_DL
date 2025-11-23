# LSTM Models for Heart Rate Prediction

This directory contains LSTM-based models for predicting heart rate time-series from running workout sequences.

## 📁 Files

- **LSTM.py**: Basic LSTM model (51K parameters)
- **LSTM_with_embeddings.py**: LSTM with user embeddings (57K parameters)  
- **train.py**: Generic training script for both models

## 🚀 Quick Start

**⚠️ Important: Activate the conda environment first!**

```bash
conda activate ai-general
```

### Train Basic LSTM
```bash
python3 Model/train.py --model lstm --epochs 100 --batch_size 32
```

### Train LSTM with User Embeddings
```bash
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32
```

**GPU Support**: 
- ✅ GPU training is fully supported and automatically configured
- The script automatically disables cuDNN on older GPUs (GTX 1060, etc.) for compatibility
- Uses PyTorch's native LSTM implementation which is stable and fast
- No manual configuration needed - just use `--device cuda` or `--device auto`

# GPU training (now works automatically!)
python3 Model/train.py --model lstm --epochs 100 --batch_size 32 --device cuda
# Or use auto-detection
python3 Model/train.py --model lstm --epochs 100 --batch_size 32 --device auto
# CPU still works if needed
python3 Model/train.py --model lstm --epochs 100 --batch_size 32 --device cpu

## 🎛️ Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `lstm` | Model type: `lstm` or `lstm_embeddings` |
| `--epochs` | `100` | Number of training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `0.001` | Learning rate |
| `--patience` | `10` | Early stopping patience |
| `--hidden_size` | `64` | LSTM hidden dimension |
| `--num_layers` | `2` | Number of LSTM layers |
| `--dropout` | `0.2` | Dropout probability |
| `--bidirectional` | `False` | Use bidirectional LSTM |
| `--embedding_dim` | `16` | User embedding dimension (for lstm_embeddings) |
| `--data_dir` | `DATA/processed` | Preprocessed data directory |
| `--checkpoint_dir` | `checkpoints` | Checkpoint save directory |
| `--device` | `auto` | Device: `cuda` or `cpu` |

## 📊 Model Architectures

### Basic LSTM
```
Input: Concat[speed, altitude, gender] → [batch, 500, 3]
  ↓
LSTM(3 → 64, layers=2) + Dropout(0.2)
  ↓
FC(64 → 1)
  ↓
Output: heart_rate [batch, 500, 1]
```

**Parameters**: 51,009

### LSTM with User Embeddings
```
Input: speed, altitude, gender, userId
  ↓
userId → Embedding(num_users, 16)
  ↓
Concat all → [batch, 500, 19]
  ↓
LSTM(19 → 64, layers=2) + Dropout(0.2)
  ↓
FC(64 → 1)
  ↓
Output: heart_rate [batch, 500, 1]
```

**Parameters**: 56,705 (depends on num_users)

## 📈 Training Features

- ✅ **Early stopping**: Monitors validation MAE with patience=10
- ✅ **Learning rate scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)
- ✅ **Best model checkpointing**: Saves model with lowest validation MAE
- ✅ **Training visualization**: Plots loss and MAE curves
- ✅ **GPU support**: Auto-detects CUDA availability

## 💾 Output Files

After training, the following files are created in `checkpoints/`:

- `{model}_best.pt`: Best model checkpoint containing:
  - Model state dict
  - Optimizer state dict
  - Training history
  - Test metrics
  - Training arguments

- `{model}_training_curves.png`: Visualization of training/validation loss and MAE

## 🎯 Target Metrics

- 🌟 **Excellent**: MAE < 5 BPM
- ✅ **Acceptable**: MAE < 10 BPM

## 📊 Dataset

- **Train**: 679 samples (70%)
- **Val**: 100 samples (15%)
- **Test**: 168 samples (15%)
- **Sequence length**: 500 timesteps
- **Features**: speed (normalized), altitude (normalized), gender (binary)
- **Target**: heart_rate (BPM, not normalized)

## 🔧 Example Usage

### Basic training
```bash
python3 Model/train.py --model lstm --epochs 100
```

### Custom hyperparameters
```bash
python3 Model/train.py \
  --model lstm_embeddings \
  --epochs 50 \
  --batch_size 16 \
  --lr 0.0001 \
  --hidden_size 128 \
  --num_layers 3 \
  --dropout 0.3
```

### With bidirectional LSTM
```bash
python3 Model/train.py --model lstm --bidirectional
```

## 🧪 Testing Models

Test model architectures without training:

```bash
# Test basic LSTM
python3 Model/LSTM.py

# Test LSTM with embeddings
python3 Model/LSTM_with_embeddings.py
```

## 📝 Code Style

All code follows PEP 8:
- `snake_case` for functions/variables
- `CamelCase` for classes
- Docstrings with Args/Returns sections
- Type hints where applicable

## 🔍 Next Steps

1. Train both models
2. Compare performance (MAE, RMSE, R²)
3. If MAE > 10: Try deeper LSTM, attention mechanisms
4. If overfitting: Increase dropout, add regularization
5. If underfitting: Increase model capacity
