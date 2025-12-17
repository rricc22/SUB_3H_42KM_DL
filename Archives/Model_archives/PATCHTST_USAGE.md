# PatchTST Model - Usage Guide

## Overview

PatchTST (Patch Time Series Transformer) is a state-of-the-art transformer-based model for time-series forecasting. This implementation adapts PatchTST for heart rate prediction from speed and altitude sequences.

## Files Created

```
Model/
├── PatchTST_HR.py        # PatchTST model architecture + HuggingFace dataset loader
└── train_patchtst.py     # Training script with early stopping, checkpointing, etc.
```

## Prerequisites

### 1. Install Dependencies

The required packages are already in `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers>=4.35.0` - HuggingFace transformers (includes PatchTST)
- `datasets>=2.15.0` - HuggingFace datasets
- `torch>=2.0.0` - PyTorch

### 2. Preprocess Data in HuggingFace Format

PatchTST requires HuggingFace Dataset format:
```bash
python3 Preprocessing/prepare_sequences_hf.py
```

This creates: `DATA/processed_hf/` with train/validation/test splits in Arrow format.

## Training Commands

### Basic Training
```bash
python3 Model/train_patchtst.py --epochs 50 --batch_size 32
```

### Custom Architecture (Larger Model)
```bash
python3 Model/train_patchtst.py \
    --epochs 100 \
    --batch_size 16 \
    --d_model 256 \
    --num_hidden_layers 6 \
    --num_attention_heads 16 \
    --ffn_dim 512 \
    --patch_length 32 \
    --stride 16
```

### Small/Fast Model (for testing)
```bash
python3 Model/train_patchtst.py \
    --epochs 20 \
    --batch_size 64 \
    --d_model 64 \
    --num_hidden_layers 2 \
    --patch_length 8
```

### With Custom Data Directory
```bash
python3 Model/train_patchtst.py \
    --data_dir DATA/processed_hf \
    --checkpoint_dir checkpoints/patchtst_exp1 \
    --epochs 50
```

## Hyperparameters

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.0001 | Learning rate (AdamW) |
| `--weight_decay` | 0.01 | Weight decay for regularization |
| `--patience` | 10 | Early stopping patience |

### Model Architecture (PatchTST Specific)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--patch_length` | 16 | Length of each patch (subsequence) |
| `--stride` | 8 | Stride between patches |
| `--d_model` | 128 | Model dimension (embedding size) |
| `--num_attention_heads` | 8 | Number of attention heads |
| `--num_hidden_layers` | 4 | Number of transformer layers |
| `--ffn_dim` | 256 | Feed-forward network dimension |
| `--dropout` | 0.1 | Dropout probability |

### Path Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `DATA/processed_hf` | HuggingFace dataset directory |
| `--checkpoint_dir` | `checkpoints` | Where to save model checkpoints |
| `--device` | auto-detect | `cuda` or `cpu` |

## What Gets Saved

After training, the following files are saved to `checkpoints/`:

```
checkpoints/
└── patchtst_bs32_lr0.0001_e50_d128_l4_p16_TIMESTAMP_*
    ├── *_model.pt              # Model weights + config
    ├── *_history.json          # Training history (losses, MAE per epoch)
    ├── *_metrics.json          # Final test metrics
    └── *_training_curves.png   # Training/validation curves plot
```

## Testing the Model (Without Training)

Test that the model architecture works:
```bash
python3 Model/PatchTST_HR.py
```

This runs a test with dummy data to verify:
- Model initialization
- Forward pass
- Backward pass
- DataLoader compatibility

## Model Architecture Details

### How PatchTST Works

1. **Patching**: Input sequence [500 timesteps] → patches [num_patches × patch_length]
   - With `patch_length=16` and `stride=8`: ~61 patches
   
2. **Patch Embedding**: Each patch → d_model dimensional vector
   
3. **Transformer Encoder**: Multi-head self-attention layers process patches
   
4. **Output Projection**: Reconstruct full sequence [500 timesteps × 1 channel]

### Input/Output
- **Input**: `[batch, 500, 3]` - (speed, altitude, gender)
- **Output**: `[batch, 500, 1]` - predicted heart rate

### Advantages Over LSTM
- ✅ Better at capturing long-range dependencies
- ✅ Parallel processing (faster training on GPU)
- ✅ State-of-the-art performance on time-series tasks
- ✅ Can leverage pre-trained weights (future work)

## Performance Expectations

Based on the paper and similar tasks:

- **Target MAE**: < 5 BPM (excellent), < 10 BPM (good)
- **Training time**: ~2-5 min/epoch (GPU), ~10-20 min/epoch (CPU)
- **Convergence**: Typically 20-50 epochs with early stopping

## Comparison with LSTM

| Model | Parameters | Training Speed | Long-range Deps | Use Case |
|-------|-----------|---------------|----------------|----------|
| LSTM | ~50K | Fast | Moderate | Baseline, quick experiments |
| LSTM+Embeddings | ~100K | Fast | Moderate | User-specific patterns |
| PatchTST | ~500K-2M | Medium | Excellent | Best performance |

## Next Steps

1. **Run preprocessing**:
   ```bash
   python3 Preprocessing/prepare_sequences_hf.py
   ```

2. **Train PatchTST**:
   ```bash
   python3 Model/train_patchtst.py --epochs 50
   ```

3. **Compare with LSTM**: Train LSTM baseline to compare
   ```bash
   python3 Model/train.py --model lstm --epochs 100
   ```

4. **Hyperparameter tuning**: Try different architectures
   - Increase `d_model` for more capacity
   - Increase `num_hidden_layers` for deeper network
   - Adjust `patch_length` based on data characteristics

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size` (try 16 or 8)
- Reduce `--d_model` (try 64 or 96)
- Reduce `--num_hidden_layers`

### Slow Training
- Increase `--batch_size` if memory allows
- Reduce `--num_hidden_layers`
- Use GPU if available

### Poor Performance (High MAE)
- Train longer (`--epochs 100`)
- Increase model capacity (`--d_model 256`)
- Try different `--patch_length` (8, 16, 32)
- Check data preprocessing quality

## References

- **Paper**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
- **HuggingFace Docs**: https://huggingface.co/docs/transformers/model_doc/patchtst
- **PatchTST GitHub**: https://github.com/yuqinie98/PatchTST
