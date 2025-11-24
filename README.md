# Heart Rate Prediction from Activity Data

**Deep Learning Course Project - CentraleSupélec**

## Project Goal

Predict heart rate time-series from running activity sequences using deep learning. Given speed, altitude, gender, and user information, the model forecasts the corresponding heart rate response throughout the workout.

## Dataset

**Endomondo Fitness Tracking Dataset**
- `endomondoHR.json`: 253,020 workouts with heart rate data
- `endomondoMeta.json`: 962,190 workouts with metadata

### Data Structure

Each workout contains:
- **Time-series data**: Speed, altitude, heart rate, GPS coordinates (lat/lon), timestamps
- **Metadata**: User ID, sport type, gender, distance, duration, elevation gain/loss
- **Labels**: Sport types include "run", "bike", "bike (transport)", "mountain bike", etc.

## Files

```
Project/
├── endomondoHR.json          # Main dataset with heart rate
├── endomondoMeta.json        # Metadata-only dataset
├── EDA_baseline.ipynb        # Full exploratory data analysis & baseline model
├── quick_test.py             # Quick validation script
├── Data_explained.md         # Original dataset documentation
└── README.md                 # This file
```

## Quick Start

### 1. Prepare Data
```bash
cd Project/
python3 prepare_sequences.py
```

This will:
- Load 974 running workouts with complete data
- Pad/truncate sequences to fixed length (300 timesteps)
- Normalize speed and altitude features
- Split into train/val/test sets (70/15/15)
- Save preprocessed PyTorch tensors

### 2. Train LSTM Baseline
```bash
python3 train_lstm_baseline.py
```

Trains a simple LSTM model:
- **Input**: Speed + altitude sequences, gender, userId
- **Output**: Heart rate sequence predictions
- **Loss**: Mean Squared Error (MSE)
- **Target**: MAE < 5 BPM

### 3. Visualize Results

```bash
python3 visualize_predictions.py --checkpoint checkpoints/best_model.pt
```

Generates:
- Predicted vs actual HR curves
- Error distribution plots
- Per-user performance analysis

## Model Architectures

### 1. LSTM Baseline (Implemented)
```python
Input Features:
  - Speed sequence: [batch, 300, 1]
  - Altitude sequence: [batch, 300, 1]
  - Gender: [batch, 1] (embedded)
  - UserId: [batch, 1] (embedded)

Architecture:
  Concat[speed, altitude] → LSTM(64) → LSTM(64) → Dense(1) → HR[batch, 300, 1]
```

### 2. Transformer / Lag-Llama (Implemented)
```python
Input Features:
  - Speed sequence: [batch, 500, 1]
  - Altitude sequence: [batch, 500, 1]
  - Gender: [batch, 1] (embedded)
  - UserId: [batch, 1] (embedded)

Architecture:
  Input Projection → Positional Encoding → Transformer Encoder (4 layers) → Output Projection → HR[batch, 500, 1]
  - d_model: 128
  - num_heads: 8
  - num_layers: 4
  - Parameters: ~2M (40x larger than LSTM)
```

### 3. Pretrained Fine-tuning (Future)
- Chronos (Amazon): T5-based time-series model
- TimeGPT: Foundation model for forecasting
- Transfer learning from large-scale time-series data

## Dataset Details

**Preprocessed Data**: 974 running workouts
- **Average sequence length**: ~300 timesteps
- **Heart rate range**: 100-180 BPM (running intensity)
- **Speed range**: 0-15 km/h (running pace)
- **Altitude range**: Variable terrain

**Input Features**:
1. **Speed** (km/h): GPS-derived velocity
2. **Altitude** (m): Elevation profile
3. **Gender**: Binary feature (male/female)
4. **UserId**: User identifier for personalization

**Target**:
- **Heart Rate** (BPM): Physiological response to activity

## Key Findings

### Heart Rate Patterns

1. **Speed-HR correlation**:
   - Strong positive correlation (r ≈ 0.6-0.8)
   - Higher speed → higher heart rate
   - Non-linear relationship (effort increases faster)

2. **Altitude-HR correlation**:
   - Uphill segments → increased HR
   - Downhill segments → decreased HR
   - Delayed response (lag ~5-10 seconds)

3. **Individual variability**:
   - Fitness level affects HR response
   - Gender differences in average HR
   - User-specific patterns (importance of userId embedding)

## Evaluation Metrics

**Primary**: MAE (Mean Absolute Error)
- Target: < 5 BPM (excellent)
- Acceptable: < 10 BPM

**Secondary**:
- MSE (Mean Squared Error)
- R² score (coefficient of determination)
- Per-timestep accuracy

## Next Steps

### Phase 1: Baseline Models ✅
- [x] Data preprocessing pipeline
- [x] LSTM baseline implementation
- [x] LSTM with user embeddings
- [x] Training and evaluation infrastructure
- [ ] Hyperparameter tuning

### Phase 2: Advanced Models ✅
- [x] Transformer architecture (Lag-Llama inspired)
- [ ] Attention visualization
- [ ] Multi-task learning (predict speed from HR)

### Phase 3: Transfer Learning
- [ ] Fine-tune Chronos pretrained model
- [ ] Compare with LSTM baseline
- [ ] Ensemble methods

## Project Deliverables

According to course requirements:
1. ✅ **One-page project description** (to be written)
2. [ ] **Presentation** of main results
3. [ ] **Final report** with methods and results + code

## Authors

Your names here

## Dataset Citation

Endomondo dataset from FitRec research project.
