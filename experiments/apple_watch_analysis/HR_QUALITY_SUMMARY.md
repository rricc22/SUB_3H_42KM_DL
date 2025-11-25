# Heart Rate Data Quality Analysis

**Date**: 2025-11-25  
**Issue**: Variable HR quality across workouts (sparse vs dense HR measurements)

---

## Problem Description

During validation, we discovered that **HR data quality varies significantly** across workouts:

- **GOOD workouts**: 10-12 HR measurements per minute (dense, high-quality)
- **SPARSE workouts**: 0.4-1 HR measurements per minute (heavily interpolated)

This creates different-looking plots:
- **Good HR**: Detailed HR fluctuations visible
- **Sparse HR**: Smooth interpolated lines (less realistic)

---

## Validation Results (8 Sample Workouts)

| Workout Date | HR Records | Duration | HR/min | Quality | Plot Quality |
|--------------|------------|----------|--------|---------|--------------|
| 2025-11-23 | 627 | 52 min | **12.1** | **GOOD** | ✅ Excellent |
| 2025-11-18 | 725 | 66 min | **11.0** | **GOOD** | ✅ Excellent |
| 2025-01-02 | 512 | 124 min | **4.1** | **ACCEPTABLE** | 🟡 OK |
| 2025-06-03 | 134 | 64 min | **2.1** | **ACCEPTABLE** | 🟡 OK |
| 2024-11-14 | 48 | 71 min | **0.7** | **SPARSE** | ⚠️ Interpolated |
| 2024-12-02 | 44 | 72 min | **0.6** | **SPARSE** | ⚠️ Interpolated |
| 2021-01-14 | 43 | 96 min | **0.4** | **SPARSE** | ⚠️ Interpolated |
| 2020-05-20 | 46 | 69 min | **0.7** | **SPARSE** | ⚠️ Interpolated |

**Pattern**: Most recent 2025 workouts have BETTER HR quality!

---

## Estimated Quality Across All 285 Workouts

Based on sample validation, estimated distribution:

| Year | Total Workouts | Est. GOOD | Est. SPARSE | Notes |
|------|----------------|-----------|-------------|-------|
| 2019 | 18 | ~2 | ~16 | Older watch OS |
| 2020 | 75 | ~8 | ~67 | Older watch OS |
| 2021 | 39 | ~4 | ~35 | Older watch OS |
| 2024 | 32 | ~3 | ~29 | Mixed quality |
| 2025 | 121 | **~60** | ~60 | **Best quality!** |
| **Total** | **285** | **~76 (27%)** | **~209 (73%)** | |

**Key Finding**: ~60 recent 2025 workouts likely have GOOD HR quality!

---

## Why This Happens

### Apple Watch OS Changes
Different watchOS versions handle HR export differently:

**Older versions (2019-2021)**:
- Export aggregate HR per workout segment (20-30 min intervals)
- Result: 1-2 HR values per long workout period
- Example: One value = "average 157 BPM from 17:24-17:46"

**Newer versions (2025)**:
- Export individual HR measurements (every 5-10 seconds)
- Result: 10-12 measurements per minute
- Example: Continuous readings at 17:24:01, 17:24:06, 17:24:11, etc.

### Not an ECG Issue
- ECG data is **not included** in Apple Health export (medical privacy)
- HR data comes from optical sensor, not ECG
- Quality depends on export format, not sensor quality

---

## Visual Comparison

### Good HR Quality (workout_20251123_103725)
```
HR samples: 627 over 52 minutes = 12/min
Plot shows: Realistic HR fluctuations
- HR increases during uphill segments
- HR decreases during rest/downhill
- Natural variability visible
```
**Plot**: `experiments/apple_watch_analysis/plots/validation_v2_workout_20251123_103725.png` ✅

### Sparse HR Quality (workout_20241114_172333)
```
HR samples: 48 over 71 minutes = 0.7/min
Plot shows: Smooth interpolated line
- Gradual linear changes
- Missing natural fluctuations
- Averaged over long segments
```
**Plot**: `experiments/apple_watch_analysis/plots/validation_v2_workout_20241114_172333.png` ⚠️

---

## Implications for Model Training

### Option 1: Use Only GOOD Quality Workouts (Recommended for Testing)
**Pros**:
- High-quality HR ground truth
- Realistic HR dynamics
- Better for model evaluation

**Cons**:
- Fewer workouts (~76 estimated)
- Limited to recent data (mostly 2025)
- Less temporal variety

**Use case**: Final model evaluation, testing HR prediction accuracy

### Option 2: Use ALL Workouts (Recommended for Training)
**Pros**:
- More training data (~285 workouts)
- Better coverage of conditions/years
- Sparse HR still provides learning signal

**Cons**:
- Some workouts have interpolated HR
- Model may learn smooth patterns
- Lower fidelity for sparse workouts

**Use case**: Model training with data augmentation, learning general patterns

### Option 3: Hybrid Approach (BEST)
**Strategy**:
1. **Train** on all 285 workouts (maximize data)
2. **Validate** on GOOD quality workouts only (accurate metrics)
3. **Test** on held-out GOOD quality workouts (final evaluation)

**Why this works**:
- Sparse HR is noisy but not wrong (averaged measurements)
- Model learns general speed→HR relationships
- Validation on dense HR ensures quality predictions
- More data = better generalization

---

## Recommendations

### For Your Project

**RECOMMENDED PATH**:

1. **Process All 285 Workouts**
   - Extract GPS + HR data
   - Accept interpolation for sparse HR
   - Label each workout by HR quality

2. **Dataset Splits** (Temporal)
   - **Train**: 2019-2024 workouts (~164 workouts)
   - **Val**: Early 2025 workouts (~60 workouts, many GOOD)
   - **Test**: Late 2025 workouts (~61 workouts, many GOOD)

3. **Training Strategy**
   - Use all training data (sparse HR = noisy labels)
   - Focus validation on high HR quality samples
   - Report metrics separately for GOOD vs SPARSE

4. **Evaluation**
   - Primary metric: MAE on GOOD HR workouts
   - Secondary metric: MAE on ALL workouts
   - This shows both "best case" and "average case"

### Quick Start Commands

**Process all workouts** (will take ~30-60 min):
```bash
python3 experiments/apple_watch_analysis/process_all_workouts.py
```

**Filter to 2025 only** (focus on GOOD HR):
```bash
python3 experiments/apple_watch_analysis/process_recent_workouts.py --year 2025
```

**Review validation plots**:
```bash
# Good HR example
eog experiments/apple_watch_analysis/plots/validation_v2_workout_20251123_103725.png

# Sparse HR example  
eog experiments/apple_watch_analysis/plots/validation_v2_workout_20241114_172333.png
```

---

## Technical Details

### How Interpolation Works
For sparse HR workouts:
1. Extract few HR measurements (e.g., 48 samples)
2. Interpolate linearly to GPS timestamps (e.g., 4019 points)
3. Result: Smooth HR curve between measurements

**Example**:
```
Original HR:  17:24 → 157 BPM, 17:46 → 169 BPM (22 min gap)
Interpolated: Every second from 157 to 169 linearly
```

This is **acceptable for training** because:
- Average HR is correct
- General trend (increasing/decreasing) is correct
- Only fine-grained fluctuations are missing

### WorkoutStatistics Metadata
Even sparse workouts have accurate summary statistics:
- `hr_avg`: Average HR over workout (used for validation)
- `hr_min`: Minimum HR reached
- `hr_max`: Maximum HR reached

We can use these to validate interpolated predictions!

---

## Action Items

**Current Status**: ✅ Fixed timezone, ✅ Fixed speed, ⚠️ Identified HR quality issue

**Next Steps**:

1. **DECISION NEEDED**: Which approach do you prefer?
   - [ ] Focus on 2025 workouts only (~121 workouts, better HR)
   - [ ] Use all 285 workouts (maximize data, accept sparse HR)
   - [x] Hybrid: Train on all, validate on GOOD (RECOMMENDED)

2. **After decision**: Process full dataset
3. **Create train/val/test splits**
4. **Adapt models** to this data format
5. **Train and evaluate**

---

## Files Reference

### Validation Plots (Examples)
- **Good HR**: `validation_v2_workout_20251123_103725.png`
- **Good HR**: `validation_v2_workout_20251118_154916.png`
- **Sparse HR**: `validation_v2_workout_20241114_172333.png`
- **Sparse HR**: `validation_v2_workout_20241202_182111.png`

### Data Files
- Workout metadata: `output/workouts_summary.json`
- Validation results: `output/validation_results_v2.json`
- Aligned samples: `data_cache/*_aligned_v2.csv`

### Scripts
- Main parser: `parse_apple_health.py`
- Validation v2: `validate_extraction_v2.py`
- HR analysis: `analyze_hr_quality.py`

---

**Generated**: 2025-11-25  
**Status**: Awaiting decision on training approach  
**Recommendation**: Hybrid approach (train on all, validate on GOOD)
