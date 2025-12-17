# Multi-Stage Fine-Tuning

Transfer learning pipeline for adapting pretrained LSTM models to personal Apple Watch workout data.

## Overview

This module implements a **two-stage progressive unfreezing** strategy for fine-tuning:

**Stage 1**: Freeze layer 0 → Train layer 1 + FC  
**Stage 2**: Unfreeze all → Fine-tune entire model

## Architecture

**Base Model**: 2-layer LSTM (64 hidden units, non-bidirectional)
- Pretrained on Endomondo dataset (974 workouts)
- Input: speed, altitude, heart_rate (3 features)
- Output: predicted heart_rate

**Fine-tuning Dataset**: Apple Watch processed data
- Location: `DATA/apple_watch_processed/`
- Format: train.pt, val.pt, test.pt
- Sequence length: 500 timesteps

## Quick Start

### Stage 1: Train upper layers
```bash
python3 launch_training.py
```

This will:
- Load pretrained model from `experiments/batch_size_search/bs16/`
- Freeze layer 0 (lower LSTM layer)
- Train layer 1 + FC layer for 50 epochs
- Save checkpoints to `checkpoints/stage1/`
- Generate training curves in `results/stage1/`

### Stage 2: Fine-tune all layers
```bash
python3 finetune/train_stage2.py
```

This will:
- Load best model from Stage 1
- Unfreeze all layers
- Fine-tune entire model for 50 epochs
- Save checkpoints to `checkpoints/stage2/`

### Evaluate Model
```bash
python3 finetune/evaluate.py
```

## Configuration

Edit `finetune/config.py` to customize:

### Stage 1 Settings
```python
STAGE1_CONFIG = {
    'freeze_layers': [0],        # Freeze first layer
    'learning_rate': 5e-4,       # Conservative LR
    'batch_size': 32,
    'epochs': 50,
    'patience': 10,              # Early stopping
}
```

### Stage 2 Settings
```python
STAGE2_CONFIG = {
    'freeze_layers': [],         # Unfreeze all
    'learning_rate': 1e-4,       # Lower LR for stability
    'batch_size': 32,
    'epochs': 50,
    'patience': 10,
}
```

### Model Architecture
```python
MODEL_CONFIG = {
    'input_size': 3,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False,
}
```

## File Structure

```
finetune/
├── __init__.py              # Package initialization
├── config.py                # Configuration for both stages
├── model.py                 # HeartRateLSTM model class
├── dataset.py               # WorkoutDataset and dataloaders
├── train_stage1.py          # Stage 1 training script
├── train_stage2.py          # Stage 2 training script
├── evaluate.py              # Model evaluation script
└── README.md                # This file

checkpoints/
├── stage1/
│   ├── best_model.pt        # Best model from Stage 1
│   └── latest_model.pt      # Latest checkpoint
└── stage2/
    ├── best_model.pt        # Best model from Stage 2
    └── latest_model.pt      # Latest checkpoint

results/
├── stage1/
│   ├── training_curves.png  # Training/validation loss curves
│   └── training_history.json
└── stage2/
    ├── training_curves.png
    └── training_history.json

logs/
└── training_stage1_*.log    # Timestamped training logs
```

## Key Features

### 1. Masked Loss Function
Uses `MaskedMSELoss` to handle variable-length sequences:
- Original lengths stored in dataset
- Padding masked during loss computation
- Prevents model from learning padding patterns

### 2. Layer Freezing
```python
model.freeze_layers([0])  # Freeze layer 0
model.unfreeze_all_layers()  # Unfreeze all
model.print_parameter_status()  # Check frozen/trainable params
```

### 3. Learning Rate Scheduling
- `ReduceLROnPlateau` scheduler
- Reduces LR when validation loss plateaus
- Helps prevent overfitting on small datasets

### 4. Early Stopping
- Monitors validation loss
- Stops training after `patience` epochs without improvement
- Saves best model based on validation performance

### 5. Training Logs
- Real-time logging with `tee` command
- Timestamped log files in `logs/`
- Training curves saved as PNG

## Data Format

The Apple Watch data should be preprocessed into `.pt` files with:

```python
{
    'speed': (N, 500, 1),
    'altitude': (N, 500, 1),
    'heart_rate': (N, 500, 1),
    'timestamps': (N, 500, 1),
    'original_lengths': (N, 1),
}
```

See `experiments/apple_watch_analysis/` for preprocessing scripts.

## Hyperparameter Tuning

### If validation loss not decreasing:
```python
# Lower learning rate
STAGE1_CONFIG['learning_rate'] = 1e-4

# Increase patience
STAGE1_CONFIG['patience'] = 15
```

### If model overfitting:
```python
# Increase dropout
MODEL_CONFIG['dropout'] = 0.3

# Reduce batch size
STAGE1_CONFIG['batch_size'] = 16
```

### If training too slow:
```python
# Increase batch size
STAGE1_CONFIG['batch_size'] = 64

# Reduce epochs
STAGE1_CONFIG['epochs'] = 30
```

## Training Tips

1. **Monitor GPU usage**: `watch -n 5 nvidia-smi`
2. **Track training progress**: `tail -f logs/training_stage1_*.log`
3. **Early stopping**: Let model train until early stopping triggers
4. **Stage 1 first**: Always complete Stage 1 before Stage 2
5. **Compare results**: Evaluate Stage 1 vs Stage 2 performance

## Expected Performance

**Stage 1 (partial fine-tuning)**:
- Should improve over base model on Apple Watch data
- May underfit if only upper layers adapted

**Stage 2 (full fine-tuning)**:
- Should achieve best performance on Apple Watch data
- Risk of overfitting on small datasets

**Target**: MAE < 10 BPM on Apple Watch test set

## Troubleshooting

### Issue: "No pretrained model found"
```bash
# Check if pretrained model exists
ls experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_best.pt

# Update path in config.py if different
STAGE1_PATHS['pretrained_model'] = 'path/to/your/model.pt'
```

### Issue: "No Stage 1 checkpoint found"
```bash
# Run Stage 1 first
python3 launch_training.py

# Or specify Stage 1 checkpoint manually
python3 finetune/train_stage2.py --checkpoint checkpoints/stage1/best_model.pt
```

### Issue: GPU out of memory
```python
# Reduce batch size in config.py
STAGE1_CONFIG['batch_size'] = 16  # or 8

# Or use CPU
STAGE1_CONFIG['device'] = 'cpu'
```

## Citation

If using this fine-tuning approach, please cite:
- Base dataset: Endomondo Fitness Tracking Dataset
- Fine-tuning data: Apple Health Export (personal data)
