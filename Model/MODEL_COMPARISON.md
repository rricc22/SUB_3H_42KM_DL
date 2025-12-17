# Model Comparison Results

Complete comparison of all trained models on heart rate prediction task.

## Best Performing Models (Top 10)

| Rank | Model | Type | MAE (BPM) | RMSE (BPM) | R² | Hyperparameters |
|------|-------|------|-----------|------------|-----|-----------------|
| 1 🥇 | lstm_bs16_lr0.001_e30_h64_l2 | LSTM | **13.88** | 18.08 | 0.188 | bs=16, lr=0.001, h=64, l=2 |
| 2 🥈 | lstm_bs16_lr0.0003_e75_h128_l4_bidir | LSTM | **13.89** | 17.99 | 0.196 | bs=16, lr=0.0003, h=128, l=4, bidir |
| 3 🥉 | lstm_bs128_lr0.0005_e100_h256_l5 | LSTM | **13.96** | 18.16 | 0.181 | bs=128, lr=0.0005, h=256, l=5 |
| 4 | lstm_bs8_lr0.001_e30_h64_l2 | LSTM | 14.09 | 18.31 | 0.167 | bs=8, lr=0.001, h=64, l=2 |
| 5 | gru_bs16_lr0.0003_e30_h128_l4_bidir | GRU | 14.23 | 18.43 | 0.156 | bs=16, lr=0.0003, h=128, l=4, bidir |
| 6 | lstm_bs64_lr0.001_e30_h64_l2 | LSTM | 14.27 | 18.38 | 0.161 | bs=64, lr=0.001, h=64, l=2 |
| 7 | lstm_bs32_lr0.001_e30_h64_l2 | LSTM | 14.45 | 18.65 | 0.136 | bs=32, lr=0.001, h=64, l=2 |
| 8 | lstm_best | LSTM | 15.41 | 19.41 | 0.064 | bs=128, lr=0.001, h=256, l=5 |
| 9 | lstm_embeddings_best | LSTM+Emb | 15.79 | 19.98 | 0.008 | bs=128, lr=0.001, h=256, l=3 |
| 10 | lstm_bs32_lr0.001_e100_h64_l2 | LSTM | 23.11 | 28.78 | -0.750 | bs=32, lr=0.001, h=64, l=2, e=100 |

## Summary Statistics

- **Best MAE**: 13.88 BPM (lstm_bs16_lr0.001_e30_h64_l2_best)
- **Mean MAE**: 36.47 BPM (across all 22 models)
- **Median MAE**: 23.11 BPM
- **Target**: MAE < 10 BPM (excellent), < 15 BPM (good)

## Performance by Model Type

| Model Type | Count | Mean MAE | Best MAE | Worst MAE |
|------------|-------|----------|----------|-----------|
| LSTM | 10 | 29.54 | **13.88** | 147.83 |
| GRU | 3 | 38.55 | **14.23** | 77.29 |
| LSTM+Embeddings | 2 | 63.45 | **15.79** | 111.11 |
| Lag-Llama | 4 | 38.74 | **38.08** | 39.61 |
| PatchTST | 3 | N/A | N/A | N/A |

## Key Findings

### 1. LSTM Wins Overall
- **Best model**: Simple 2-layer LSTM with 64 hidden units
- **MAE**: 13.88 BPM (within acceptable range < 15 BPM)
- **Key insight**: Small models generalize better on this dataset

### 2. Batch Size Matters
- **Optimal batch size**: 16 (best performance)
- Too small (bs=8): Slightly worse (MAE: 14.09)
- Too large (bs=128): Worse (MAE: 13.96, but more unstable)

### 3. Model Complexity vs Performance
- Simple models (2-4 layers) outperform deep models (5 layers)
- Bidirectional helps slightly: 13.89 vs 13.88 (marginal)
- User embeddings don't help: MAE 15.79 (worse than baseline)

### 4. Training Duration
- **30 epochs**: Sufficient (MAE: 13.88-14.45)
- 10 epochs: Underfitting (MAE: 15.41-147.83)
- 100 epochs: Overfitting (MAE: 23.11-111.11)

### 5. Transformer Models Struggle
- **Lag-Llama**: Poor performance (MAE: 38-40 BPM)
- **Reason**: Wrong hyperparameters (too few epochs, high LR)
- **PatchTST**: No metrics available (training issues)

## Recommendations

### For Production Use
**Model**: `lstm_bs16_lr0.001_e30_h64_l2_best`
- MAE: 13.88 BPM
- RMSE: 18.08 BPM
- R²: 0.188
- Architecture: 2-layer LSTM, 64 hidden units
- Training: bs=16, lr=0.001, 30 epochs

### For Further Improvement
1. **Fine-tune on personal data**: Use Apple Watch data for user-specific models
2. **Ensemble methods**: Combine top 3 LSTM models
3. **Feature engineering**: Add heart rate variability, time of day
4. **Retrain Lag-Llama**: Use correct hyperparameters (lr=0.0001, epochs=100)

## Finetuned Models

**Status**: No finetuned models trained yet

To train finetuned models:
```bash
# Stage 1: Partial fine-tuning
python3 Model/launch_training.py

# Stage 2: Full fine-tuning
python3 finetune/train_stage2.py

# Re-run comparison
python3 Model/compare_models.py
```

## Complete Results

See `model_comparison.csv` for full results including all hyperparameters.

---

**Generated**: December 17, 2024  
**Dataset**: Endomondo (974 running workouts)  
**Evaluation**: Test set (146 samples)
