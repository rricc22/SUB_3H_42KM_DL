# Data Analysis Findings - Potential Issues Affecting Model Performance

**Generated:** 2025-11-24 20:33
**Analyst:** OpenCode EDA
**Context:** Training accuracy not meeting expectations despite multiple training runs

---

## Executive Summary

The data quality analysis reveals **THREE CRITICAL ISSUES** that are likely contributing to poor model performance:

1. ⚠️ **WEAK FEATURE-TARGET CORRELATION** - Speed has only 0.25 correlation with heart rate
2. ⚠️ **DISTRIBUTION MISMATCH ACROSS SPLITS** - Test set has different statistics than train/val
3. ⚠️ **HIGH PADDING RATIO** - 43% of sequences are artificially padded

---

## Critical Findings

### 🔴 ISSUE #1: Weak Feature-Target Correlation

**Problem:**
- **Speed → Heart Rate correlation: 0.254** (weak)
- **Altitude → Heart Rate correlation: 0.022** (negligible)

**What this means:**
- The input features (speed, altitude) have **very weak linear relationships** with the target (heart rate)
- This suggests the relationship is either:
  - Non-linear (requires more complex models)
  - Mediated by missing features (e.g., user fitness level, age, terrain type)
  - Noisy or inherently difficult to predict

**Impact on training:**
- Models struggle to learn meaningful patterns
- MAE improvements plateau quickly
- High variance in predictions

**Recommendations:**
1. Consider **feature engineering**: 
   - Speed changes (acceleration/deceleration)
   - Rolling statistics (moving averages)
   - Speed × altitude interactions
   - Cumulative elevation gain
2. Try **more complex models** (attention mechanisms, transformers)
3. Incorporate **user embeddings** more heavily (user fitness is key predictor)

---

### 🟡 ISSUE #2: Distribution Mismatch Between Splits

**Problem:**
The test set shows **different statistics** compared to train/val:

| Metric | Train | Val | Test | Issue |
|--------|-------|-----|------|-------|
| Speed mean | -0.00 | 0.04 | **0.13** | Test shifted higher |
| Speed std | 1.00 | 1.02 | **1.19** | Test more variable |
| Altitude mean | 0.00 | -0.13 | **0.13** | Test shifted higher |
| Altitude std | 1.00 | 0.58 | **1.53** | Test MUCH more variable |
| HR mean | 149.93 | 149.32 | **147.67** | Test lower HR |

**What this means:**
- Test set contains **different types of workouts** (more variable terrain, different pacing)
- Model trained on train distribution will **generalize poorly** to test
- This explains why test MAE is higher than validation MAE

**Impact on training:**
- Overfitting to training distribution
- Poor generalization to test set
- Misleading validation performance

**Recommendations:**
1. **Investigate the split strategy**: 
   - Are test users systematically different?
   - Consider stratified splitting by workout characteristics
2. **Apply data augmentation** to increase train/val variability
3. **Use robust normalization** (e.g., robust scaler instead of Z-score)

---

### 🟡 ISSUE #3: High Padding Ratio

**Problem:**
- **43.8% of training sequences** are padded (length < 500)
- Mean actual length: 410 timesteps
- Padding adds **~90 timesteps** of artificial zeros on average

**What this means:**
- Models learn to predict on **synthetic padded data**
- Padding introduces artificial patterns
- Models may struggle to distinguish real vs. padded regions
- Effective training data is reduced by ~22% (90/410)

**Impact on training:**
- Diluted gradients from padded regions
- Models may memorize padding patterns
- Harder to learn temporal dependencies

**Recommendations:**
1. **Use masking** in loss function (ignore padded timesteps)
2. Consider **variable-length sequences** instead of padding
3. Try **shorter sequence length** (e.g., 400) to reduce padding
4. Implement **attention mechanisms** that can learn to ignore padding

---

## Positive Findings ✅

### Data Quality (GOOD)
- ✅ **No data leakage**: Clean user splits (0 overlap)
- ✅ **Normalization correct**: Train set has mean≈0, std≈1
- ✅ **No invalid values**: No zeros, no physiologically impossible heart rates
- ✅ **Outliers minimal**: <2% outliers (acceptable for real-world data)
- ✅ **Sample size adequate**: 13,855 train / 3,539 val / 3,581 test

### Data Integrity (GOOD)
- No missing heart rate values
- Reasonable HR range (50-220 BPM)
- Consistent sequence structure across splits
- Proper train/val/test distribution (70/15/15)

---

## Detailed Statistics

### Heart Rate Target Variable
| Split | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Train | 149.93 | 18.80 | 50 | 220 |
| Val   | 149.32 | 19.03 | 50 | 219 |
| Test  | 147.67 | 20.07 | 50 | 220 |

**Observations:**
- Test set has **higher variance** (std=20.07 vs 18.80)
- Test set has **lower mean** HR (147.67 vs 149.93)
- Suggests test users may have different fitness profiles

### Sequence Length Distribution
| Split | Mean | Median | % Padded | % Exact 500 |
|-------|------|--------|----------|-------------|
| Train | 410.2 | 500.0 | 43.8% | 56.2% |
| Val   | 409.9 | 500.0 | 43.3% | 56.7% |
| Test  | 416.9 | 500.0 | 38.6% | 61.4% |

**Observations:**
- Test set has **longer sequences** on average
- More test sequences reach max length (61.4% vs 56.2%)
- Padding is significant across all splits

---

## Model Training Recommendations

### Immediate Actions (High Priority)

1. **Implement Padding Mask in Loss Function**
   ```python
   # Pseudocode
   mask = (lengths > timestep_index).float()
   loss = ((predictions - targets) ** 2 * mask).sum() / mask.sum()
   ```

2. **Add User Embeddings** (if not already using)
   - User fitness level is likely the strongest predictor
   - Current userId should be embedded and used

3. **Feature Engineering**
   - Speed changes (diff): `speed[t] - speed[t-1]`
   - Acceleration: `(speed[t] - speed[t-1]) / dt`
   - Cumulative elevation: `cumsum(altitude_change)`
   - Rolling mean speed (window=10)

### Medium-Term Improvements

4. **Try Different Architectures**
   - **Attention-based models** (PatchTST, Transformers)
   - **Temporal Convolutional Networks (TCN)**
   - **Hybrid CNN-LSTM**

5. **Adjust Normalization**
   - Try **robust scaling** (median/IQR instead of mean/std)
   - Normalize per-user or per-workout

6. **Experiment with Sequence Length**
   - Try 400 timesteps (reduce padding)
   - Try dynamic batching with variable lengths

### Long-Term Considerations

7. **Data Augmentation**
   - Add Gaussian noise to speed/altitude
   - Time warping
   - Magnitude scaling

8. **Investigate Test Set Distribution**
   - Analyze user characteristics in test set
   - Consider re-splitting with stratification

9. **Ensemble Methods**
   - Combine multiple model types
   - User-specific fine-tuning

---

## Expected Performance Ceiling

Given the **weak correlation (0.254)**, the theoretical limit for this prediction task is constrained:

- **Best-case MAE**: ~10-15 BPM (given feature limitations)
- **Current expectation**: MAE < 10 BPM may be **unrealistic** without:
  - Additional features (user fitness, age, previous workouts)
  - More sophisticated models (attention, ensemble)
  - Better handling of temporal dependencies

**Current MAE goal (< 5 BPM) may be too ambitious** for this feature set.

---

## Next Steps

1. ✅ **Data quality verified** - No major data issues found
2. 🔄 **Implement padding masks** in training loop (URGENT)
3. 🔄 **Add feature engineering** (speed changes, rolling stats)
4. 🔄 **Try user embeddings** if not already using
5. 🔄 **Test PatchTST or Transformer models** (better for weak correlations)
6. 🔄 **Re-evaluate performance expectations** based on correlation analysis

---

## Conclusion

The data is **clean and well-preprocessed**, but the **task itself is inherently difficult** due to:
- Weak linear relationships between features and target
- Distribution mismatches across splits  
- High padding ratio diluting learning signal

**Recommendations prioritize:**
1. Proper padding masking (immediate impact)
2. Feature engineering (unlock non-linear patterns)
3. Advanced architectures (attention mechanisms)
4. Realistic performance expectations (MAE 10-15 may be best achievable)

The models are not reaching expected accuracy likely because **the expectations are too high** for the given feature set, not because of training issues.

---

## Generated Files

All visualizations available in `EDA/EDA_Generation/`:
- `sequence_lengths.png` - Distribution of sequence lengths by split
- `feature_distributions.png` - Speed, altitude, HR distributions
- `heart_rate_analysis.png` - Detailed HR analysis (target variable)
- `sample_time_series.png` - 5 sample sequences visualized
- `correlation_matrix.png` - Feature correlation heatmap
- `padding_analysis.png` - Padding statistics by split
- `DATA_QUALITY_REPORT.md` - Detailed statistical report
- `FINDINGS_SUMMARY.md` - This document
