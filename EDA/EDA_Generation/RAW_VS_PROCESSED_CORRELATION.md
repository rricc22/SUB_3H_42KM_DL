# RAW vs PROCESSED DATA - CORRELATION ANALYSIS

**Generated:** 2025-11-25 10:48:28

## Summary

- **Raw workouts analyzed:** 11,442
- **Total timesteps (raw):** 5,721,000
- **Processed timesteps:** 6,927,500
- **Analysis method:** Streaming (low RAM usage)

## Correlation Comparison

| Feature Pair | Raw Data | Processed Data | Difference |
|--------------|----------|----------------|------------|
| Speed → Heart Rate | 0.212884 | 0.253977 | +0.041093 |
| Altitude → Heart Rate | 0.045590 | 0.022200 | -0.023390 |
| Speed → Altitude | -0.038368 | 0.008822 | +0.047190 |

## Key Findings

✅ **Preprocessing preserves correlations well** - Speed-HR correlation difference < 0.05

🔴 **Weak correlation in raw data confirmed** - Speed-HR correlation is 0.213 in original data

This confirms that the weak correlation is **inherent to the data**, not caused by preprocessing.

## Temporal Lag Analysis

- **Best Speed lag:** 2 timesteps (correlation: 0.195785)
- **Best Altitude lag:** 30 timesteps (correlation: 0.079828)

💡 **Insight:** Heart rate responds to speed changes with a **2-timestep delay**. Consider adding lagged features to your model.

## Recommendations

1. The weak correlation is **inherent to the dataset**, not a preprocessing artifact
2. Consider adding **lagged features** (speed[t-k] to predict HR[t])
3. Feature engineering suggestions:
   - Speed changes (acceleration/deceleration)
   - Rolling averages of speed and altitude
   - Cumulative elevation gain
   - User-specific features (fitness level, historical average HR)

