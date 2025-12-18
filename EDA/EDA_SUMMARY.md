# EDA and Baseline Model - Summary Report

**Date:** November 11, 2025  
**Dataset:** Endomondo Fitness Tracking Data  
**Sample Size:** 1,000 workouts

---

## Executive Summary

Successfully completed exploratory data analysis and implemented a baseline Random Forest classifier to detect fraudulent running activities (those performed with bikes/scooters instead of genuine running).

**Key Result: 84.3% accuracy** using only 6 simple statistical features.

---

## Data Overview

### Sport Distribution (Top 10)
| Sport | Count |
|-------|-------|
| Run | 435 |
| Bike | 427 |
| Bike (transport) | 57 |
| Kayaking | 40 |
| Rowing | 29 |
| Mountain bike | 5 |
| Core stability training | 4 |
| Others | 3 |

### Label Distribution
- **Genuine runs:** 435 (43.5%)
- **Fraudulent (bikes, etc.):** 565 (56.5%)
- **Class balance:** Slightly imbalanced but acceptable

---

## Features Extracted

### 6 Statistical Features

1. **speed_mean** - Average speed over workout
2. **speed_std** - Standard deviation of speed
3. **hr_mean** - Average heart rate (BPM)
4. **hr_std** - Standard deviation of heart rate
5. **hr_speed_corr** - Correlation between HR and speed
6. **speed_variability** - Coefficient of variation for speed

### Data Quality
- All 1,000 workouts successfully processed
- No NaN or Inf values in final feature matrix
- Robust handling of missing data

---

## Key Findings

### 1. Heart Rate is the Strongest Indicator

**Feature Importance Ranking:**
1. **hr_mean: 52.1%** ⭐ (Most important!)
2. hr_std: 29.4%
3. speed_std: 6.0%
4. speed_mean: 5.7%
5. speed_variability: 4.8%
6. hr_speed_corr: 2.0%

**Insight:** Heart rate metrics dominate the classification, accounting for >82% of feature importance.

### 2. Heart Rate Patterns Differ Significantly

**Running (Genuine):**
- Mean HR: 144.1 ± 13.2 BPM
- HR variability: 11.5 ± 5.4 BPM

**Biking (Fraud):**
- Mean HR: 127.7 ± 17.8 BPM
- HR variability: 12.7 ± 5.0 BPM

**Key Observation:** Runners maintain higher average heart rates with more consistent patterns.

### 3. Speed Patterns Are Less Discriminative

**Running:**
- Mean speed: 1.0 ± 3.3 km/h (many zeros - GPS issues?)
- Speed std: 0.1 ± 0.5 km/h

**Biking:**
- Mean speed: 5.0 ± 9.7 km/h
- Speed std: 1.5 ± 3.1 km/h

**Note:** Speed features are noisy, suggesting data quality issues with GPS measurements.

---

## Model Performance

### Random Forest Classifier
- **Architecture:** 100 trees, max depth 10
- **Training set:** 700 samples
- **Test set:** 300 samples

### Results

**Overall Accuracy:** 84.3%

**Detailed Metrics:**

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Run (Genuine) | 0.84 | 0.79 | 0.81 | 130 |
| Bike (Fraud) | 0.85 | 0.88 | 0.86 | 170 |

**Confusion Matrix:**
```
              Predicted
              Run  Bike
Actual Run    103   27
       Bike    20  150
```

### Performance Analysis

**Strengths:**
- High precision for both classes (84-85%)
- Better at detecting fraud (88% recall for bikes)
- Balanced performance across classes

**Weaknesses:**
- 21% false negative rate for genuine runs
- Some runs misclassified as fraud
- Speed features underutilized (noisy data)

---

## Visualizations Generated

### 1. `feature_distributions.png`
Shows distribution of all 6 features split by class (Run vs Bike). Clear separation visible in heart rate features.

### 2. `confusion_matrix.png`
Heatmap showing model predictions vs actual labels. Diagonal dominance indicates good performance.

### 3. `feature_importance.png`
Bar chart showing relative importance of features. Heart rate features dominate.

---

## Data Quality Issues Identified

1. **GPS Speed Data:**
   - Many zeros in running speed data
   - High variance suggests measurement errors
   - Could be improved with better preprocessing

2. **Missing Fields:**
   - Not all workouts have complete time-series data
   - Some missing 'speed' or 'heart_rate' arrays
   - Required robust handling in feature extraction

3. **Activity Labeling:**
   - Mix of "bike" and "bike (transport)" labels
   - Some edge cases (kayaking, rowing) treated as fraud
   - Consider more nuanced labeling scheme

---

## Next Steps for Deep Learning Project

### Phase 1: Data Preparation
1. **Expand dataset** to 10,000+ workouts
2. **Clean speed data** - remove zeros, interpolate missing values
3. **Standardize labels** - focus on run vs bike only
4. **Data augmentation** - create synthetic fraud examples

### Phase 2: Deep Learning Models

**Architecture 1: LSTM for Time Series**
```
Input: Speed + HR sequences (variable length)
→ LSTM layers (128 units)
→ Attention mechanism
→ Dense layers
→ Binary classification
```

**Architecture 2: Multi-Input CNN-LSTM**
```
Branch 1: Speed → 1D CNN → LSTM
Branch 2: HR → 1D CNN → LSTM
Branch 3: Altitude → 1D CNN → LSTM
→ Concatenate → Dense → Output
```

**Architecture 3: Transformer**
```
Input: Multi-modal sequences
→ Positional encoding
→ Transformer encoder blocks
→ Classification head
```

### Phase 3: Advanced Features
1. **Attention visualization** - identify suspicious segments
2. **Anomaly detection** - unsupervised learning
3. **Transfer learning** - pre-train on all activities
4. **Ensemble methods** - combine multiple models

---

## Recommendations

### For Course Project

 **Current baseline (84.3%) is strong** - provides good comparison point

 **Clear path to improvement** - deep learning can leverage full time-series

 **Interesting problem** - real-world application with clear impact

 **Good dataset** - large enough for deep learning (250K+ workouts available)

### Implementation Priority

**High Priority:**
1. Implement LSTM baseline (should beat 84.3%)
2. Use proper train/validation/test split
3. Focus on heart rate + speed sequences
4. Create learning curves and ablation studies

**Medium Priority:**
1. Add attention mechanism
2. Multi-modal fusion
3. Data augmentation strategies
4. Hyperparameter optimization

**Low Priority (if time permits):**
1. Transformer architecture
2. Anomaly detection approach
3. Transfer learning experiments

---

## Deliverables Generated

 `EDA_baseline.ipynb` - Complete exploratory analysis (Jupyter notebook)  
 `run_full_eda.py` - Automated EDA script  
 `features_sample.csv` - Extracted features (1,000 samples)  
 `feature_distributions.png` - Feature visualization  
 `confusion_matrix.png` - Model performance  
 `feature_importance.png` - Feature ranking  
 `EDA_SUMMARY.md` - This summary report  

---

## Conclusion

The baseline analysis demonstrates that **fraud detection in running data is feasible** with simple statistical features achieving 84.3% accuracy. Heart rate patterns are the strongest indicators of genuine vs fraudulent activity.

**Next milestone:** Implement LSTM model to leverage full time-series data and target **>90% accuracy**.

The project is well-positioned for the deep learning course requirements with:
-  Clear problem definition
-  Substantial dataset
-  Strong baseline
-  Multiple deep learning approaches to explore
-  Real-world application value

---

**Project Status:**  Phase 1 (EDA & Baseline) COMPLETE  
**Ready for:** Phase 2 (Deep Learning Implementation)
