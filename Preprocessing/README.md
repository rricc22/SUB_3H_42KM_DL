# Preprocessing Scripts - Overview

This directory contains scripts for preprocessing the EndomondoHR dataset for heart rate prediction.

## Files

### Main Preprocessing Scripts

1. **`prepare_sequences_v2.py`** - PyTorch format (for LSTM models)
   - Output: `DATA/processed/train.pt`, `val.pt`, `test.pt`
   - Use for: LSTM, LSTM with embeddings

2. **`prepare_sequences_hf.py`** - HuggingFace format (from raw JSON)
   - Output: `DATA/processed_hf/` (Arrow format)
   - Use for: PatchTST, transformer models
   - **Slow** (processes raw JSON from scratch)

3. **`convert_pt_to_hf.py`** ⭐ **NEW - RECOMMENDED**
   - Output: `DATA/processed_hf/` (Arrow format)
   - Use for: PatchTST, transformer models
   - **Fast** (converts existing .pt files)

### Utility Scripts

- `prepare_sequences_streaming.py` - Streaming version (memory efficient)
- `verify_preprocessing.ipynb` - Jupyter notebook to verify preprocessing output

---

## Quick Start: PatchTST Training

If you already have `DATA/processed/train.pt`, `val.pt`, `test.pt`:

### **FAST PATH** (Recommended)

```bash
# Step 1: Convert existing .pt files to HuggingFace format (takes ~30 seconds)
python3 Preprocessing/convert_pt_to_hf.py

# Step 2: Train PatchTST
python3 Model/train_patchtst.py --epochs 50 --batch_size 32
```

### **SLOW PATH** (From Scratch)

```bash
# Step 1: Preprocess from raw JSON (takes ~5-10 minutes)
python3 Preprocessing/prepare_sequences_hf.py

# Step 2: Train PatchTST
python3 Model/train_patchtst.py --epochs 50 --batch_size 32
```

---

## Workflow Diagram

```
Raw Data (endomondoHR.json)
    │
    ├─────────────────────┬─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
prepare_sequences_v2  prepare_sequences_hf  prepare_sequences_streaming
    │                     │                     │
    ▼                     ▼                     ▼
DATA/processed/       DATA/processed_hf/    DATA/processed_streaming/
(PyTorch .pt)         (HuggingFace)         (HDF5 + streaming)
    │                     │                     │
    ├─────────────────────┤                     │
    │                     │                     │
    │      convert_pt_to_hf.py ⭐               │
    │                     │                     │
    ▼                     ▼                     ▼
LSTM models         PatchTST models       Large-scale experiments
(train.py)          (train_patchtst.py)
```

---

## File Format Comparison

| Format | Files | Size | Load Time | Use Case |
|--------|-------|------|-----------|----------|
| **PyTorch** | `train.pt`, `val.pt`, `test.pt` | ~40MB | Fast | LSTM, custom models |
| **HuggingFace** | `train/`, `validation/`, `test/` | ~50MB | Medium | PatchTST, transformers |
| **Streaming** | `*.h5` + `*.pt` | ~60MB | Lazy | Large datasets |

---

## Commands Reference

### Generate PyTorch Format
```bash
python3 Preprocessing/prepare_sequences_v2.py
```
**Output:** `DATA/processed/{train,val,test}.pt`

### Generate HuggingFace Format (from raw JSON)
```bash
python3 Preprocessing/prepare_sequences_hf.py
```
**Output:** `DATA/processed_hf/`

### Convert PyTorch → HuggingFace ⭐ **FASTEST**
```bash
python3 Preprocessing/convert_pt_to_hf.py
```
**Input:** `DATA/processed/{train,val,test}.pt`  
**Output:** `DATA/processed_hf/`

### Verify Preprocessing
```bash
jupyter notebook Preprocessing/verify_preprocessing.ipynb
```

---

## What Gets Preprocessed?

All scripts perform the same preprocessing steps:

1. **Load** workouts from `DATA/endomondoHR.json`
2. **Filter** valid running workouts (7 filters applied)
3. **Pad/truncate** sequences to 500 timesteps
4. **Normalize** speed and altitude (Z-score using train stats)
5. **Split** by userId (70/15/15 train/val/test)
6. **Save** in the target format

**Key differences:**
- Same data, different storage format
- Same splits (uses `RANDOM_SEED=42`)
- Same normalization parameters

---

## Troubleshooting

### "PyTorch .pt files not found"
Run PyTorch preprocessing first:
```bash
python3 Preprocessing/prepare_sequences_v2.py
```

### "HuggingFace datasets not installed"
Install dependencies:
```bash
pip install datasets transformers
```

### Conversion is slow
This is normal for ~20k samples. Expected time: 30-60 seconds.

### Out of memory
Use the streaming version:
```bash
python3 Preprocessing/prepare_sequences_streaming.py
```

---

## Recommended Workflow

1. **First time setup:**
   ```bash
   # Generate PyTorch format (for LSTM baseline)
   python3 Preprocessing/prepare_sequences_v2.py
   ```

2. **For PatchTST experiments:**
   ```bash
   # Convert to HuggingFace format
   python3 Preprocessing/convert_pt_to_hf.py
   
   # Train PatchTST
   python3 Model/train_patchtst.py --epochs 50
   ```

3. **Compare models:**
   ```bash
   # Train LSTM baseline
   python3 Model/train.py --model lstm --epochs 100
   
   # Train PatchTST
   python3 Model/train_patchtst.py --epochs 50
   ```

---

## Questions?

See:
- `Model/PATCHTST_USAGE.md` - PatchTST training guide
- `AGENTS.md` - Project-wide agent guidelines
- `README.md` - Project overview
