# Heart Rate Prediction from Running Activity Data

Deep Learning project for predicting heart rate time-series from running workout sequences using LSTM neural networks.

## Project Overview

This project uses deep learning to predict heart rate responses during running workouts based on:
- Speed sequences
- Altitude/elevation changes
- User characteristics (gender, userId)

**Goal**: Given activity data (speed and altitude over time), predict the corresponding heart rate throughout the workout.

## Dataset

**Endomondo Fitness Tracking Dataset**
- Source: FitRec research project
- Contains 253,020+ running workouts with heart rate data
- Features: Speed, altitude, GPS coordinates, timestamps
- Metadata: User ID, gender, distance, duration

**Note**: The dataset files are NOT included in this repository due to size constraints. See [Data Setup](#data-setup) section below.

## Project Structure

```
SUB_3H_42KM_DL/
├── Model/                          # Neural network architectures
│   ├── LSTM.py                    # Basic LSTM model
│   ├── LSTM_with_embeddings.py    # LSTM with user embeddings
│   ├── train.py                   # Training script
│   ├── evaluate_test.py           # Evaluation on test set
│   ├── inference.py               # Inference/prediction
│   └── README.md                  # Model documentation
├── Preprocessing/                  # Data preprocessing scripts
│   ├── prepare_sequences.py       # Basic preprocessing
│   ├── prepare_sequences_v2.py    # Enhanced version (recommended)
│   └── prepare_sequences_streaming.py  # Low-RAM streaming version
├── DATA/                          # Data directory (not tracked)
│   ├── endomondoHR.json          # Raw dataset (download separately)
│   ├── processed/                # Preprocessed tensors
│   └── temp/                     # Temporary processing files
├── checkpoints/                   # Saved model weights
├── run_pipeline.sh               # Complete training pipeline
├── .gitignore
├── AGENTS.md                     # Agent guidelines
├── Project_Description.pdf
└── README.md                     # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/rricc22/SUB_3H_42KM_DL.git
cd SUB_3H_42KM_DL

# Install dependencies
pip install torch numpy pandas scikit-learn tqdm h5py
```

### 2. Data Setup

Download the Endomondo dataset and place `endomondoHR.json` in the `DATA/` directory:

```bash
# Create DATA directory if it doesn't exist
mkdir -p DATA

# Place your endomondoHR.json file here
# The file should be at: DATA/endomondoHR.json
```

### 3. Run Complete Pipeline

```bash
# Process data and train both models (basic LSTM + LSTM with embeddings)
bash run_pipeline.sh

# Or customize:
# Process 10K samples and train only basic LSTM
bash run_pipeline.sh -n 10000 -m lstm -e 50

# Skip preprocessing and train with custom hyperparameters
bash run_pipeline.sh -s -m lstm_embeddings --lr 0.0005 --hidden-size 128
```

### 4. Manual Step-by-Step

If you prefer to run each step manually:

```bash
# Step 1: Preprocess data (recommended version)
python3 Preprocessing/prepare_sequences_v2.py

# Step 2: Train basic LSTM
python3 Model/train.py --model lstm --epochs 100 --batch_size 32

# Step 3: Train LSTM with embeddings
python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32

# Step 4: Evaluate on test set
python3 Model/evaluate_test.py --model lstm_embeddings

# Step 5: Run inference
python3 Model/inference.py --checkpoint checkpoints/lstm_embeddings_best.pt
```

## Data Preprocessing

Three preprocessing scripts available:

### 1. `prepare_sequences.py` (Basic)
- Simple preprocessing
- Fixed sequence length: 300 timesteps
- Good for quick testing

### 2. `prepare_sequences_v2.py` (Recommended)
- Enhanced version with improvements
- Sequence length: 500 timesteps (preserves more data)
- Robust gender encoding
- Saves timestamps for future use
- Better documentation

### 3. `prepare_sequences_streaming.py` (For large datasets)
- Low RAM usage (~500 MB constant)
- Processes full 253K workouts
- Batch processing
- Outputs both .pt and .h5 formats

**Output**: Preprocessed data saved in `DATA/processed/`:
- `train.pt`, `val.pt`, `test.pt`: PyTorch tensors
- `scaler_params.json`: Feature normalization parameters
- `metadata.json`: Dataset statistics

## Model Architectures

### 1. Basic LSTM (`LSTM.py`)
Simple sequence-to-sequence model:
```
Input: [speed, altitude] sequences
     ↓
LSTM Layers (64 hidden units, 2 layers)
     ↓
Dense Layer
     ↓
Output: Heart rate predictions
```

### 2. LSTM with Embeddings (`LSTM_with_embeddings.py`)
Enhanced model with user personalization:
```
Input: [speed, altitude] sequences + gender + userId
     ↓
User Embedding Layer (captures individual patterns)
     ↓
LSTM Layers (64 hidden units, 2 layers)
     ↓
Dense Layer
     ↓
Output: Personalized heart rate predictions
```

## Training Pipeline

The `run_pipeline.sh` script provides a complete training workflow:

### Options:
```bash
-n, --max-samples N       Number of samples to preprocess (default: all ~253K)
-s, --skip-preprocessing  Skip data preprocessing step
-m, --model MODEL         Model to train: lstm, lstm_embeddings, or both
-e, --epochs N            Number of training epochs (default: 100)
-b, --batch-size N        Batch size (default: 32)
--lr RATE                 Learning rate (default: 0.001)
--hidden-size N           LSTM hidden dimension (default: 64)
--num-layers N            Number of LSTM layers (default: 2)
--dropout RATE            Dropout probability (default: 0.2)
--patience N              Early stopping patience (default: 10)
--device DEVICE           Device: cuda, cpu, or auto (default: auto)
```

### Examples:
```bash
# Full pipeline with all data
bash run_pipeline.sh

# Quick test with 1,000 samples
bash run_pipeline.sh -n 1000 -m lstm -e 20

# Custom hyperparameters
bash run_pipeline.sh -n 10000 -e 50 -b 16 --lr 0.0005 --hidden-size 128
```

## Evaluation Metrics

**Primary**: Mean Absolute Error (MAE)
- Target: < 5 BPM (excellent)
- Acceptable: < 10 BPM

**Secondary**:
- MSE (Mean Squared Error)
- R² score (coefficient of determination)
- Per-timestep accuracy

## Results

After training, outputs include:
- `checkpoints/lstm_best.pt`: Best basic LSTM weights
- `checkpoints/lstm_embeddings_best.pt`: Best LSTM with embeddings weights
- `checkpoints/*_training_curves.png`: Loss and accuracy plots

## Key Findings

1. **Speed-HR Correlation**: Strong positive correlation (r ≈ 0.6-0.8)
2. **Altitude Impact**: Uphill segments increase HR, with ~5-10 second lag
3. **Individual Variability**: User embeddings significantly improve predictions
4. **Gender Differences**: Observable patterns in average HR response

## Development Guidelines

See `AGENTS.md` for:
- Build & test commands
- Code style guidelines
- AI/ML specific conventions
- Python best practices

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy
- Pandas
- scikit-learn
- tqdm
- h5py (for streaming preprocessing)

## Citation

If you use this code or the Endomondo dataset, please cite the FitRec research project.

## Authors

Riccardo & Team
CentraleSupélec - Deep Learning Course Project

## License

This project is for educational purposes as part of the CentraleSupélec Deep Learning course.

## Troubleshooting

### Data not found error
```bash
# Ensure endomondoHR.json is in the correct location:
ls DATA/endomondoHR.json
```

### Out of memory during preprocessing
```bash
# Use the streaming version:
python3 Preprocessing/prepare_sequences_streaming.py
```

### CUDA out of memory during training
```bash
# Reduce batch size:
bash run_pipeline.sh -b 16

# Or use CPU:
bash run_pipeline.sh --device cpu
```

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
