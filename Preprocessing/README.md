# Preprocessing

Convert raw workout data into padded sequences for LSTM training.

## Scripts

### `prepare_sequences_v2.py`
Original preprocessing (loads all data in RAM).

```bash
python3 Preprocessing/prepare_sequences_v2.py
```

**Output:** `DATA/processed/{train,val,test}.pt`, `scaler_params.json`

---

### `prepare_sequences_streaming.py` ⭐ (Recommended)
Memory-efficient streaming version (~500 MB RAM constant).

```bash
python3 Preprocessing/prepare_sequences_streaming.py
```

**Features:**
- Batch processing (500 workouts at a time)
- Low RAM usage
- Processes 253K workouts → ~9K valid samples
- Train/val/test split: 70/15/15

**Output:** Same as v2

---

### `verify_preprocessing.ipynb`
Jupyter notebook to verify preprocessing output and inspect data.

```bash
jupyter notebook Preprocessing/verify_preprocessing.ipynb
```

---

## Data Format

**Input:** Raw CSV with columns: `workoutId, userId, speed, altitude, heart_rate, gender, type`

**Output:** PyTorch tensors
- Sequences padded/truncated to 500 timesteps
- Speed & altitude normalized (Z-score)
- Heart rate NOT normalized (kept in BPM)
- Train: 6,368 samples | Val: 1,463 | Test: 1,031
