# Normalization Impact on Correlation

**Generated:** 2025-11-24 21:27:20

## Summary

Analyzed 6,927,500 timesteps from processed training data.

## Correlation Comparison

| Feature Pair | Original Scale | Normalized Scale | Difference |
|--------------|----------------|------------------|------------|
| Speed → Heart Rate | 0.253977 | 0.253977 | +0.000000 |
| Altitude → Heart Rate | 0.022200 | 0.022200 | +0.000000 |

## Key Findings

✅ **Normalization preserves correlation** (difference < 0.001)

This is expected - Z-score normalization is a linear transformation, which does not affect correlation coefficients.

🔴 **ROOT CAUSE IDENTIFIED: Weak Correlation**

- Speed → Heart Rate correlation = **0.254** (weak)
- Altitude → Heart Rate correlation = **0.022** (negligible)

**Conclusion:** The input features have **inherently weak predictive power** for heart rate. This explains why models struggle to achieve high accuracy.

## Recommendations

Since normalization is not the issue, focus on:

1. **Feature engineering** - Create derived features with stronger correlations
2. **User embeddings** - User fitness level is likely the strongest predictor
3. **Temporal features** - Lagged speed/altitude, acceleration, rolling averages
4. **Lower performance expectations** - With weak correlations, MAE < 10 BPM may be realistic ceiling

