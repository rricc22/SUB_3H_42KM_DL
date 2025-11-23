# Agent Guidelines - Heart Rate Prediction Project

## Commands
```bash
# Data preprocessing
python3 Preprocessing/prepare_sequences_v2.py     # Preprocess & pad sequences to 500 timesteps
jupyter notebook Preprocessing/verify_preprocessing.ipynb  # Verify preprocessing output

# Model training
python3 Model/train.py --model lstm --epochs 100 --batch_size 32          # Train basic LSTM
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32  # Train LSTM with user embeddings

# Test models (without training)
python3 Model/LSTM.py                             # Test basic LSTM architecture
python3 Model/LSTM_with_embeddings.py             # Test LSTM with embeddings architecture

# Legacy EDA (deprecated - was for fraud detection)
python3 EDA/quick_test.py                         # Quick test on 500 samples
python3 EDA/run_full_eda.py                       # Full EDA with visualizations
jupyter notebook EDA/EDA_baseline.ipynb           # Interactive exploration
```

## Code Style (PEP 8)
- **Naming**: `snake_case` functions/vars, `CamelCase` classes, ALL_CAPS constants
- **Imports**: stdlib → third-party (numpy, pandas, torch, sklearn) → local; blank lines between groups
- **Docstrings**: Triple quotes `"""` at module/function level with Args/Returns sections
- **Error handling**: try-except with `continue` for parsing; check key existence before access
- **Types**: Optional type hints for function signatures
- **Data loading**: Use `ast.literal_eval()` for JSON (handles single quotes); handle NaN/Inf with `fillna(0)` or masking
- **File headers**: Shebang `#!/usr/bin/env python3` + module docstring

## Project Context
**Goal**: Predict heart rate time-series from speed/altitude sequences  
**Inputs**: Speed [seq_len], altitude [seq_len], gender (binary), userId  
**Output**: Heart rate [seq_len] (time-series regression)  
**Dataset**: 974 genuine "run" workouts; sequences variable length (median ~500 timesteps)  
**Preprocessing**: Pad/truncate to 500 timesteps, normalize speed/altitude (Z-score on train), split by userId (70/15/15)  
**Target metric**: MAE < 5 BPM (strong), < 10 BPM (acceptable)
