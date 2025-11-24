# Data Quality Analysis Report
**Generated:** 2025-11-24 20:32:55
**Dataset:** Heart Rate Prediction (Processed Data)

## 1. Basic Statistics

### TRAIN Split

**Sample count:** 13855

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| speed | -0.00 | 1.00 | -3.62 | -0.47 | -0.03 | 0.46 | 72.54 |
| altitude | 0.00 | 1.00 | -1.36 | -0.30 | -0.19 | 0.02 | 25.39 |
| heart_rate | 149.93 | 18.80 | 50.00 | 139.00 | 150.00 | 163.00 | 220.00 |
| original_lengths | 410.24 | 137.54 | 50.00 | 340.00 | 500.00 | 500.00 | 500.00 |

### VAL Split

**Sample count:** 3539

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| speed | 0.04 | 1.02 | -3.62 | -0.50 | -0.01 | 0.52 | 66.20 |
| altitude | -0.13 | 0.58 | -1.29 | -0.31 | -0.27 | -0.16 | 11.71 |
| heart_rate | 149.32 | 19.03 | 50.00 | 138.00 | 151.00 | 163.00 | 219.00 |
| original_lengths | 409.89 | 136.42 | 50.00 | 331.00 | 500.00 | 500.00 | 500.00 |

### TEST Split

**Sample count:** 3581

| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |
|---------|------|-----|-----|-----|-----|-----|-----|
| speed | 0.13 | 1.19 | -3.62 | -0.36 | 0.12 | 0.55 | 72.72 |
| altitude | 0.13 | 1.53 | -1.36 | -0.31 | -0.22 | -0.07 | 25.03 |
| heart_rate | 147.67 | 20.07 | 50.00 | 136.00 | 149.00 | 161.00 | 220.00 |
| original_lengths | 416.89 | 138.08 | 50.00 | 376.00 | 500.00 | 500.00 | 500.00 |

## 2. Normalization Verification

### Training Set Normalized Statistics

- **Speed:** mean=-0.000000, std=1.000000
- **Altitude:** mean=0.000000, std=1.000000

✅ **Speed normalization:** CORRECT

✅ **Altitude normalization:** CORRECT

## 3. Outlier Detection

### TRAIN Split

- **Speed:** 100087 outliers (1.44%)
- **Altitude:** 79978 outliers (1.15%)
- **Heart Rate:** 69339 outliers (1.00%)

### VAL Split

- **Speed:** 25769 outliers (1.46%)
- **Altitude:** 20226 outliers (1.14%)
- **Heart Rate:** 18392 outliers (1.04%)

### TEST Split

- **Speed:** 35054 outliers (1.96%)
- **Altitude:** 28825 outliers (1.61%)
- **Heart Rate:** 10734 outliers (0.60%)

## 4. Data Leakage Check

**Unique users:**
- Train: 428
- Val: 92
- Test: 92

**User overlaps:**
- Train-Val: 0 users
- Train-Test: 0 users
- Val-Test: 0 users

✅ **No data leakage detected**

## 5. Sequence Length Analysis

### TRAIN
- Mean: 410.2
- Median: 500.0
- Padded: 43.8%

### VAL
- Mean: 409.9
- Median: 500.0
- Padded: 43.3%

### TEST
- Mean: 416.9
- Median: 500.0
- Padded: 38.6%

## 6. Feature Distributions

## 7. Heart Rate Analysis (Target Variable)

### TRAIN - Invalid Values
- Zeros: 0 (0.000%)
- < 40 BPM: 0 (0.000%)
- > 220 BPM: 0 (0.000%)

### VAL - Invalid Values
- Zeros: 0 (0.000%)
- < 40 BPM: 0 (0.000%)
- > 220 BPM: 0 (0.000%)

### TEST - Invalid Values
- Zeros: 0 (0.000%)
- < 40 BPM: 0 (0.000%)
- > 220 BPM: 0 (0.000%)

## 8. Correlation Analysis

**Correlation with Heart Rate:**
- Speed: 0.2540
- Altitude: 0.0222

## 9. Padding Analysis

### TRAIN
- Padded (< 500): 6073 (43.8%)
- Exact (= 500): 7782 (56.2%)
- Truncated (> 500): 0 (0.0%)

### VAL
- Padded (< 500): 1531 (43.3%)
- Exact (= 500): 2008 (56.7%)
- Truncated (> 500): 0 (0.0%)

### TEST
- Padded (< 500): 1384 (38.6%)
- Exact (= 500): 2197 (61.4%)
- Truncated (> 500): 0 (0.0%)

