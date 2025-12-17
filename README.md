# Heart Rate Prediction from Activity Data

**Deep Learning Course Project - CentraleSupélec**

## Overview

Predict heart rate time-series from running activity sequences using deep learning. Given speed, altitude, gender, and user information, the model forecasts corresponding heart rate responses throughout workouts.

**Dataset**: Endomondo fitness tracking (974 running workouts)  
**Target**: MAE < 5 BPM (excellent), < 10 BPM (acceptable)

## Quick Start

```bash
# 1. Preprocess data (pad to 500 timesteps, normalize, split 70/15/15)
python3 Preprocessing/prepare_sequences.py

# 2. Train models
python3 Model/train.py --model lstm --epochs 100 --batch_size 32
python3 Model/train.py --model gru --epochs 100 --batch_size 32
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32

# 3. Fine-tune on personal data (two-stage progressive unfreezing)
python3 finetune/train_stage1.py
python3 finetune/train_stage2.py

# 4. Evaluate
python3 Model/evaluate_test.py
```

## Project Structure

```
├── DATA/                     # Raw Endomondo dataset + documentation
├── EDA/                      # Exploratory data analysis notebooks
├── Preprocessing/            # Sequence preparation & normalization
├── Model/                    # LSTM, GRU, PatchTST, Lag-Llama implementations
├── finetune/                 # Multi-stage fine-tuning on Apple Watch data
├── Inferences/               # Evaluation scripts & prediction visualization
├── experiments/              # Apple Watch data parsing & preparation
└── Presentation/             # Report (LaTeX) and slides
```

## Models Implemented

| Model | Architecture | Parameters | Best MAE |
|-------|-------------|------------|----------|
| **LSTM Baseline** | 2-layer LSTM (64 units) | ~50K | TBD |
| **LSTM + Embeddings** | LSTM with user/gender embeddings | ~65K | TBD |
| **GRU** | 2-layer GRU (64 units) | ~40K | TBD |
| **PatchTST** | Patch-based Transformer | ~2M | TBD |
| **Lag-Llama** | Transfer learning from pretrained | ~2M | TBD |

**Input**: Speed [500], altitude [500], gender, userId  
**Output**: Heart rate [500]

## Key Features

- **Multi-stage fine-tuning**: Progressive layer unfreezing for transfer learning
- **User embeddings**: Personalized predictions accounting for fitness levels
- **Data augmentation**: Jitter, scaling, time warping for robustness
- **Sequence padding**: Handles variable-length workouts (median ~500 timesteps)

## Results Summary

See `Model/FINAL_RESULTS.md` for detailed comparison and `Presentation/Report/main.pdf` for full analysis.

## Dataset Citation

Endomondo dataset from FitRec research project.
