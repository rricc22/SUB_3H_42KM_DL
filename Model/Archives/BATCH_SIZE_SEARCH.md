# Batch Size Search Scripts

Automated scripts to find the optimal batch size for heart rate prediction models by running parallel training experiments.

## Overview

The batch size search workflow consists of two scripts:

1. **`batch_size_search.py`** - Launches parallel training runs with different batch sizes
2. **`visualize_batch_comparison.py`** - Creates comprehensive visualizations and analysis

## Quick Start

### Basic Usage

```bash
# Test batch sizes 16, 32, 64 for basic LSTM
python3 Model/batch_size_search.py --model lstm --batch_sizes 16 32 64 --epochs 50

# Visualize results
python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search
```

### For LSTM with Embeddings

```bash
# Run batch size search
python3 Model/batch_size_search.py --model lstm_embeddings --batch_sizes 16 32 64 --epochs 50

# Visualize results
python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search --model lstm_embeddings
```

## Detailed Usage

### 1. Batch Size Search (`batch_size_search.py`)

Launches parallel training processes, one for each batch size.

**Key Features:**
- Launches all training runs in parallel (fully utilizing your GPU)
- Monitors each process and collects results automatically
- Saves separate logs and checkpoints for each batch size
- Exports results to JSON and CSV formats

**Arguments:**

```bash
python3 Model/batch_size_search.py \
    --model lstm \                      # Model type: lstm, lstm_embeddings, lag_llama, patchtst
    --batch_sizes 16 32 64 \           # Batch sizes to test (space-separated)
    --epochs 50 \                       # Number of epochs per run
    --lr 0.001 \                        # Learning rate
    --patience 10 \                     # Early stopping patience
    --hidden_size 64 \                  # LSTM hidden size
    --num_layers 2 \                    # Number of LSTM layers
    --embedding_dim 16 \                # User embedding dimension (for lstm_embeddings)
    --data_dir DATA/processed \         # Preprocessed data directory
    --output_dir experiments/batch_size_search \  # Output directory
    --device cuda                       # Device: cuda or cpu
```

**Output Structure:**

```
experiments/batch_size_search/
├── bs16/                                    # Batch size 16 results
│   ├── lstm_bs16_lr0.001_e50_h64_l2_best.pt
│   └── lstm_bs16_lr0.001_e50_h64_l2_training_curves.png
├── bs32/                                    # Batch size 32 results
│   ├── lstm_bs32_lr0.001_e50_h64_l2_best.pt
│   └── lstm_bs32_lr0.001_e50_h64_l2_training_curves.png
├── bs64/                                    # Batch size 64 results
│   └── ...
├── logs/                                    # Training logs
│   ├── training_bs16_20251124_150000.log
│   ├── training_bs32_20251124_150002.log
│   └── training_bs64_20251124_150004.log
├── batch_size_results_lstm.json             # Results in JSON format
└── batch_size_results_lstm.csv              # Results in CSV format
```

### 2. Visualization (`visualize_batch_comparison.py`)

Creates comprehensive comparison plots and analysis.

**Key Features:**
- 6 comparison plots: Test MAE, Val MAE, Train/Val/Test comparison, R², Training time, RMSE
- Summary table with all metrics
- Best batch size recommendation based on multiple criteria
- Exports as both PNG (high-res) and PDF (publication-ready)

**Arguments:**

```bash
python3 Model/visualize_batch_comparison.py \
    --input experiments/batch_size_search \   # Input directory with results
    --model lstm \                             # Model type
    --output results/batch_comparison.png     # Optional: custom output path
```

**Output:**
- `batch_size_comparison_lstm.png` - High-resolution comparison plots (150 DPI)
- `batch_size_comparison_lstm.pdf` - Publication-ready PDF version
- Console output with detailed summary table and recommendations

## Example Workflows

### 1. Quick Test (20 epochs, 2 batch sizes)

Fast exploration to get initial insights:

```bash
# Run search (takes ~30-40 minutes)
python3 Model/batch_size_search.py \
    --model lstm \
    --batch_sizes 16 64 \
    --epochs 20

# Visualize
python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search
```

### 2. Comprehensive Search (50 epochs, 3 batch sizes)

More thorough evaluation:

```bash
# Run search (takes ~2-3 hours depending on early stopping)
python3 Model/batch_size_search.py \
    --model lstm \
    --batch_sizes 16 32 64 \
    --epochs 50

# Visualize
python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search
```

### 3. Compare with Larger Architecture

Test batch sizes with larger model:

```bash
# Run search with larger hidden size
python3 Model/batch_size_search.py \
    --model lstm \
    --batch_sizes 16 32 64 \
    --epochs 50 \
    --hidden_size 256 \
    --num_layers 5 \
    --output_dir experiments/batch_size_search_large

# Visualize
python3 Model/visualize_batch_comparison.py \
    --input experiments/batch_size_search_large
```

### 4. LSTM with Embeddings

Test batch sizes for personalized model:

```bash
# Run search
python3 Model/batch_size_search.py \
    --model lstm_embeddings \
    --batch_sizes 16 32 64 \
    --epochs 50 \
    --embedding_dim 16

# Visualize
python3 Model/visualize_batch_comparison.py \
    --input experiments/batch_size_search \
    --model lstm_embeddings
```

## Understanding the Results

### Metrics Tracked

1. **Test MAE** - Mean Absolute Error on test set (primary metric)
2. **Test RMSE** - Root Mean Squared Error on test set
3. **Test R²** - R-squared score (goodness of fit)
4. **Best Val MAE** - Lowest validation MAE during training
5. **Final Train MAE** - Training MAE at end of training
6. **Epochs Trained** - Number of epochs before early stopping
7. **Training Time** - Total training time in minutes

### What to Look For

**Generalization Gap:**
- Compare Test MAE vs Val MAE
- Smaller gap indicates better generalization
- Large gap suggests overfitting

**Training Stability:**
- Check training curves for each batch size
- Smoother curves with larger batch sizes
- More noise with smaller batch sizes (can help escape local minima)

**Performance vs Efficiency:**
- Smaller batch sizes: Better generalization, longer training
- Larger batch sizes: Faster training, may overfit

### Typical Findings

Based on theory and your existing results:

- **Batch Size 16-32**: Often best generalization, noisy gradients help exploration
- **Batch Size 64**: Good balance between speed and performance
- **Batch Size 128+**: Faster but may converge to suboptimal solutions

## Monitoring Running Jobs

### Check GPU Usage

```bash
# Monitor GPU memory and utilization
watch -n 1 nvidia-smi
```

### Check Training Progress

```bash
# Monitor latest log file
tail -f experiments/batch_size_search/logs/training_bs*.log
```

### Check Process Status

```bash
# List running Python training processes
ps aux | grep train.py
```

## Troubleshooting

### Out of Memory (OOM) Errors

If you get CUDA OOM errors:

1. Reduce the number of batch sizes tested simultaneously
2. Use smaller batch sizes only (e.g., `--batch_sizes 16 32`)
3. Reduce model size (`--hidden_size 32 --num_layers 2`)
4. Close other GPU applications

### Process Killed

If processes are killed:

1. Check system memory: `free -h`
2. Check logs in `experiments/batch_size_search/logs/`
3. Reduce `--epochs` or test fewer batch sizes

### Missing Results

If visualization fails:

1. Check that training completed: `ls experiments/batch_size_search/bs*/`
2. Verify JSON file exists: `ls experiments/batch_size_search/*.json`
3. Check logs for errors

## Tips for Best Results

1. **Start small**: Test with `--epochs 20` and 2 batch sizes first
2. **Monitor early**: Check logs after 5-10 minutes to catch issues early
3. **Compare architectures**: Run searches for different model sizes
4. **Save everything**: Keep all checkpoints for later comparison
5. **Document findings**: Note which batch size works best for your data

## Integration with Existing Training

Once you find the optimal batch size:

```bash
# Use the best batch size for final training
python3 Model/train.py \
    --model lstm \
    --batch_size 32 \        # Use optimal value found
    --epochs 100 \           # Train for longer
    --lr 0.001 \
    --checkpoint_dir checkpoints/final_model
```

## Expected Timeline

With GTX 1060 6GB on your dataset (13,855 train samples):

- **Batch Size 16**: ~60 min per 50 epochs
- **Batch Size 32**: ~40 min per 50 epochs  
- **Batch Size 64**: ~30 min per 50 epochs

**Total time for 3 batch sizes (parallel)**: ~60 minutes (limited by slowest)

**Note**: Early stopping may reduce times significantly (e.g., stops at epoch 30-40)

## Next Steps

After finding the optimal batch size:

1. Train final model with optimal batch size for more epochs
2. Test learning rate schedules with optimal batch size
3. Experiment with model architecture (hidden size, layers)
4. Try data augmentation or regularization techniques

## Questions or Issues?

Check:
- Training logs in `experiments/batch_size_search/logs/`
- Individual training curves in `experiments/batch_size_search/bs*/`
- GPU status with `nvidia-smi`
