# EDA (Exploratory Data Analysis)

Data exploration and analysis notebooks (mostly legacy from fraud detection baseline).

## Files

### `EDA_baseline.ipynb`
Interactive Jupyter notebook for data exploration.

```bash
jupyter notebook EDA/EDA_baseline.ipynb
```

**Contents:**
- Data distribution analysis
- Feature correlations
- Outlier detection
- Basic statistics

---

### `Files_size_EDA.ipynb`
Analyze dataset sizes and memory usage.

```bash
jupyter notebook EDA/Files_size_EDA.ipynb
```

---

### Legacy Scripts (Archived)

**Note:** These were for a previous fraud detection project and are now archived:

- `Archives/quick_test.py` - Quick test on 500 samples (deprecated)
- `Archives/run_full_eda.py` - Full EDA with visualizations (deprecated)

---

## Current Project EDA Summary

See `EDA_SUMMARY.md` for detailed exploratory analysis of the heart rate prediction dataset.

**Key findings:**
- 8,862 valid workouts from 253K total
- Median sequence length: ~500 timesteps
- Speed range: 0-6 m/s (running pace)
- Altitude range: -100 to +2000m
- Heart rate range: 50-220 BPM

---

## Quick Start

For current project analysis:

```bash
# View summary
cat EDA/EDA_SUMMARY.md

# Interactive exploration
jupyter notebook EDA/EDA_baseline.ipynb
```
