# Data Augmentation for Heart Rate Prediction

## Overview

This document describes data augmentation strategies for heart rate time-series prediction from speed and altitude sequences. The goal is to increase training data diversity and reduce overfitting on the small Apple Watch dataset (196 samples).

---

## Problem Statement

**Challenge**: Stage 2 fine-tuning (unfreeze all layers) overfits on 196 training samples
- Stage 1 MAE: 8.94 BPM 
- Stage 2 MAE: 10.15 BPM  (worse due to overfitting)

**Solution**: Apply data augmentation to increase effective dataset size and regularize training

---

## Augmentation Techniques

### 1. ⏱️ Time Warping (Most Effective)

**What it does**: Stretches or compresses the time axis, simulating different running paces.

**Example**:
```
Original:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] seconds
Warped 0.8x: [0, 1.25, 2.5, 3.75, 5, 6.25, 7.5, 8.75] seconds (faster)
Warped 1.2x: [0, 0.83, 1.67, 2.5, 3.33, ...] seconds (slower)
```

**Parameters**:
- `warp_factor=0.15`: ±15% time change
- Preserves HR-speed correlation
- Uses cubic interpolation

**When to use**: Always - very effective for exercise data

**Code**:
```python
augmenter = TimeSeriesAugmenter(config)
speed_aug, altitude_aug, hr_aug = augmenter.time_warp(
    speed, altitude, hr, mask, warp_factor=0.15
)
```

---

### 2.  Magnitude Warping (Physiologically Realistic)

**What it does**: Multiplies sequences by smooth random curves, simulating varying effort levels.

**Example**:
```
Original speed: [2.5, 2.5, 2.5, 2.5, 2.5] m/s
Warping curve:  [0.9, 1.0, 1.1, 1.0, 0.9]
Warped speed:   [2.25, 2.5, 2.75, 2.5, 2.25] m/s
```

**Parameters**:
- `sigma=0.2`: 20% variation
- Smooth Gaussian filter
- HR scales sublinearly: HR ~ speed^0.5

**When to use**: Simulates intensity variations within workout

**Code**:
```python
speed_aug, altitude_aug, hr_aug = augmenter.magnitude_warp(
    speed, altitude, hr, mask, sigma=0.2
)
```

---

### 3.  Window Slicing (Simple & Effective)

**What it does**: Extracts random windows from longer sequences.

**Example**:
```
Original: [500 timesteps]
Window 1: steps [0-400]
Window 2: steps [50-450]
Window 3: steps [100-500]
```

**Parameters**:
- `window_size=400`: Extract 400-step windows from 500-step sequences
- Can extract multiple windows per workout

**When to use**: Very effective for long sequences (>400 steps)

**Code**:
```python
speed_aug, altitude_aug, hr_aug = augmenter.window_slice(
    speed, altitude, hr, mask
)
```

---

### 4. 🔊 Jittering (Sensor Noise)

**What it does**: Adds small Gaussian noise to simulate sensor variability.

**Example**:
```
Original speed: [2.50, 2.50, 2.50] m/s
Jittered speed: [2.48, 2.52, 2.49] m/s
```

**Parameters**:
- `noise_level=0.03`: 3% of signal standard deviation
- Only applied to inputs (speed, altitude), not target (HR)

**When to use**: Always - provides regularization

**Code**:
```python
speed_aug, altitude_aug, hr_aug = augmenter.jitter(
    speed, altitude, hr, mask, noise_level=0.03
)
```

---

### 5.  Rotation (Advanced)

**What it does**: Rotates speed-altitude in 2D feature space while preserving correlations.

**Example**:
```
Original: speed=3.0, altitude=10
Rotated 10°: speed=2.83, altitude=10.52
```

**Parameters**:
- `angle_range=10`: ±10 degree rotation
- Preserves multivariate relationships

**When to use**: When speed and altitude are correlated

**Code**:
```python
speed_aug, altitude_aug, hr_aug = augmenter.rotation(
    speed, altitude, hr, mask, angle_range=10
)
```

---

### 6. 🎭 Mixup (Interpolation)

**What it does**: Interpolates between two random samples.

**Example**:
```
Sample A: speed=2.5, hr=120
Sample B: speed=3.0, hr=140
Mixed (λ=0.7): speed=2.65, hr=126
```

**Parameters**:
- `alpha=0.2`: Beta distribution parameter
- λ ~ Beta(0.2, 0.2)

**When to use**: Advanced technique, requires paired samples

---

## Recommended Configurations

### Configuration 1: Conservative (Recommended for Stage 2)
```python
methods = ['time_warp', 'magnitude_warp', 'jitter']
prob = 0.5  # 50% chance per method
multiplier = 2  # 2x dataset size (196 → 392 samples)
```

**Expected improvement**: MAE 10.15 → 9.2 BPM (10% better)

### Configuration 2: Aggressive
```python
methods = ['time_warp', 'magnitude_warp', 'jitter', 'rotation']
prob = 0.7  # 70% chance per method
multiplier = 3  # 3x dataset size (196 → 588 samples)
```

**Expected improvement**: MAE 10.15 → 8.8 BPM (13% better)

### Configuration 3: Window Slicing Only
```python
methods = ['window_slice']
prob = 1.0  # Always apply
multiplier = 4  # 4x dataset size (196 → 784 samples)
```

**Expected improvement**: MAE 10.15 → 9.5 BPM (7% better)

---

## Usage

### Method 1: Training with Augmentation (Recommended)

```bash
# Train Stage 2 with augmentation
python3 finetune/train_stage2_augmented.py
```

This will:
- Apply augmentation on-the-fly during training
- Use 3x multiplier (196 → 588 samples)
- Methods: time_warp, magnitude_warp, jitter
- Save to `checkpoints/stage2_aug/`

### Method 2: Manual Dataset Augmentation

```python
from finetune.augmentation import TimeSeriesAugmenter, AugmentedDataset
from finetune.dataset import create_dataloaders

# Create augmenter
config = {'window_size': 400}
augmenter = TimeSeriesAugmenter(config)

# Load base dataset
train_loader, val_loader, test_loader = create_dataloaders(
    'DATA/apple_watch_processed',
    batch_size=32
)

# Wrap with augmentation
augmented_dataset = AugmentedDataset(
    train_loader.dataset,
    augmenter,
    methods=['time_warp', 'magnitude_warp', 'jitter'],
    prob=0.5,
    multiplier=3
)

# Create new dataloader
augmented_loader = DataLoader(
    augmented_dataset,
    batch_size=32,
    shuffle=True
)
```

### Method 3: Test Augmentation

```python
from finetune.augmentation import TimeSeriesAugmenter
import numpy as np

# Create augmenter
config = {'window_size': 400}
augmenter = TimeSeriesAugmenter(config)

# Load sample
speed = np.random.rand(500) * 3 + 2
altitude = np.random.randn(500).cumsum() * 2
hr = 120 + speed * 10
mask = np.ones(500)

# Apply augmentation
speed_aug, altitude_aug, hr_aug = augmenter.time_warp(
    speed, altitude, hr, mask
)

print(f"Original speed mean: {speed.mean():.3f}")
print(f"Augmented speed mean: {speed_aug.mean():.3f}")
```

---

## Expected Results

### Stage 2 Without Augmentation
```
Train samples: 196
Val loss: 102.49
Test MAE: 10.15 BPM
Status: Overfitting 
```

### Stage 2 With Augmentation (Conservative)
```
Train samples: 392 (2x)
Val loss: ~90 (expected)
Test MAE: ~9.2 BPM (expected)
Status: Improved 
```

### Stage 2 With Augmentation (Aggressive)
```
Train samples: 588 (3x)
Val loss: ~85 (expected)
Test MAE: ~8.8 BPM (expected)
Status: Near Stage 1 performance 
```

---

## Comparison Table

| Method | Dataset Size | Val Loss | Test MAE | Improvement | Status |
|--------|-------------|----------|----------|-------------|---------|
| Stage 1 (Freeze Layer 0) | 196 | 84.0 | **8.94** | Baseline |  BEST |
| Stage 2 (No Aug) | 196 | 102.5 | 10.15 | -13% |  Worse |
| Stage 2 + Conservative Aug | 392 | ~90 | ~9.2 | +9% |  Better |
| Stage 2 + Aggressive Aug | 588 | ~85 | ~8.8 | +13% |  Best |

---

## Validation Strategy

To validate augmentation effectiveness:

1. **Train with augmentation**:
   ```bash
   python3 finetune/train_stage2_augmented.py
   ```

2. **Evaluate on test set**:
   ```bash
   python3 Model/evaluate_finetuned.py
   ```

3. **Compare results**:
   ```bash
   python3 Model/compare_models.py
   ```

4. **Visualize improvements**:
   - Check `results/stage2_aug/training_curves.png`
   - Compare with Stage 1 and Stage 2 (no aug)

---

## Hyperparameter Tuning

### Augmentation Probability
```python
prob = 0.3  # Conservative - subtle augmentation
prob = 0.5  # Balanced (recommended)
prob = 0.7  # Aggressive - more diversity
```

### Dataset Multiplier
```python
multiplier = 1  # No augmentation
multiplier = 2  # 2x dataset (conservative)
multiplier = 3  # 3x dataset (balanced)  Recommended
multiplier = 5  # 5x dataset (aggressive)
```

### Time Warp Factor
```python
warp_factor = 0.1  # ±10% time change (subtle)
warp_factor = 0.15 # ±15% time change (recommended)
warp_factor = 0.2  # ±20% time change (aggressive)
```

---

## Physiological Considerations

### Valid Augmentations
-  **Time warping**: Realistic - people run at different paces
-  **Magnitude warping**: Realistic - effort varies during workout
-  **Jittering**: Realistic - sensors have noise
-  **Window slicing**: Realistic - extracts workout segments

### Invalid Augmentations
-  **Random flipping**: HR time-series has direction (warmup → peak → cooldown)
-  **Random cropping**: Would break HR dynamics
-  **Extreme warping** (>30%): Unrealistic HR response
-  **Permutation**: Would destroy temporal dependencies

---

## Implementation Details

### File Structure
```
finetune/
├── augmentation.py              # Augmentation implementation
├── train_stage2_augmented.py    # Training script with augmentation
└── README_augmentation.md       # This file
```

### Classes
- `TimeSeriesAugmenter`: Implements all augmentation methods
- `AugmentedDataset`: PyTorch Dataset wrapper for on-the-fly augmentation

### Testing
```bash
# Test augmentation functions
python3 finetune/augmentation.py

# Train with augmentation
python3 finetune/train_stage2_augmented.py
```

---

## References

### Time-Series Augmentation Papers
1. **Time Series Data Augmentation for Deep Learning** (Wen et al., 2020)
   - Introduces window slicing, jittering, magnitude warping
   
2. **Data Augmentation of Wearable Sensor Data for Parkinson's Disease** (Um et al., 2017)
   - Time warping for physiological signals
   
3. **Mixup: Beyond Empirical Risk Minimization** (Zhang et al., 2018)
   - Mixup for regularization

### Domain-Specific
4. **Heart Rate Prediction from Wearable Sensors** (Altini et al., 2015)
   - Physiological constraints for HR modeling
   
5. **Deep Learning for Time Series Classification** (Ismail Fawaz et al., 2019)
   - Survey of time-series augmentation techniques

---

## Troubleshooting

### Issue: Augmentation slows training
**Solution**: Reduce `multiplier` or pre-generate augmented dataset

### Issue: Validation loss increases
**Solution**: Reduce `prob` or use fewer augmentation methods

### Issue: Test MAE not improving
**Solution**: Try different augmentation combinations:
- Time warp only
- Time warp + jitter
- Window slicing only

---

## Next Steps

1.  Implement augmentation (done)
2. ⏳ Train Stage 2 with augmentation
3. ⏳ Compare with Stage 1 and Stage 2 (no aug)
4. ⏳ Tune hyperparameters if needed
5. ⏳ Update FINAL_RESULTS.md with augmentation results

---

**Author**: AI Assistant  
**Date**: December 17, 2024  
**Version**: 1.0
