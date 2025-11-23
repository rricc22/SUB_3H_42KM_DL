# Inferences

Evaluate trained models and run predictions on test data.

## Scripts

### `evaluate_test.py` ⭐ (Recommended)
Full test set evaluation with detailed metrics and 8-panel visualization.

```bash
python3 Inferences/evaluate_test.py \
  --checkpoint checkpoints/lstm_best.pt \
  --data DATA/processed/test.pt \
  --output results/test_evaluation.png \
  --device cpu
```

**Output:**
- Detailed metrics (MAE, RMSE, R², accuracy breakdown)
- 8 visualizations:
  1. Predicted vs True scatter plot
  2. Error distribution
  3. Absolute error histogram
  4. Per-workout MAE distribution
  5. Example workout 1 (time series)
  6. Example workout 2 (time series)
  7. Error by HR range
  8. Summary metrics panel
- Saved as PNG figure (default: `results/test_evaluation.png`)

**Metrics shown:**
- MAE, RMSE, R²
- Accuracy within ±5, ±10, ±15 BPM
- Per-workout statistics (best, worst, median)

---

### `inference.py`
Batch inference on test set (metrics only, no plots - faster).

```bash
python3 Inferences/inference.py \
  --checkpoint checkpoints/lstm_best.pt \
  --data DATA/processed/test.pt \
  --device cpu
```

**Output:**
- MAE, RMSE, R² on entire test set
- Faster than evaluate_test.py (no plotting)

---

## Important: Use `--device cpu`

Due to cuDNN compatibility issues with GTX 1060, always use `--device cpu`:

```bash
# ✅ Correct
python3 Inferences/evaluate_test.py --checkpoint ... --device cpu

# ❌ Will crash with cuDNN error
python3 Inferences/evaluate_test.py --checkpoint ... --device cuda
```

---

## Available Checkpoints

- `checkpoints/lstm_best.pt` - Basic LSTM (MAE: 14.67 BPM on test)
- `checkpoints/lstm_embeddings_best.pt` - LSTM with user embeddings

---

## Example Workflow

1. **Train model:**
   ```bash
   python3 Model/train.py --model lstm --epochs 100 --device cpu
   ```

2. **Evaluate on test set:**
   ```bash
   python3 Inferences/evaluate_test.py \
     --checkpoint checkpoints/lstm_best.pt \
     --data DATA/processed/test.pt \
     --output results/test_evaluation.png \
     --device cpu
   ```

3. **View results:**
   - Terminal: Metrics summary
   - File: `results/test_evaluation.png` (8 plots)

---

## Target Performance

- 🌟 **Excellent**: MAE < 5 BPM
- ✅ **Acceptable**: MAE < 10 BPM
- ⚠️ **Needs improvement**: MAE > 10 BPM

Current best: **14.67 BPM** (needs larger model)

---

## Recent Improvements ✨

**v2.0 (Latest):**
- ✅ Fixed dict/namespace compatibility in checkpoint loading
- ✅ Improved imports: `from Model.LSTM import ...` (clearer)
- ✅ Better path handling: `sys.path.insert(0, ...)` (more reliable)
- ✅ All scripts tested and working

See `IMPORTS_EXPLAINED.md` for technical details on import improvements.
