#!/usr/bin/env python3
"""
EDA script for processed heart rate prediction data.
Validates data quality, normalization, and checks for anomalies.
"""

import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Paths
DATA_DIR = Path("DATA/processed")
OUTPUT_DIR = Path("EDA/EDA_Generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata and scaler params
with open(DATA_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)
    
with open(DATA_DIR / "scaler_params.json", "r") as f:
    scaler = json.load(f)

print("="*80)
print("HEART RATE PREDICTION - DATA QUALITY ANALYSIS")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nMetadata: {metadata}")
print(f"\nScaler Parameters:")
for key, val in scaler.items():
    print(f"  {key}: {val:.4f}")

# Load all splits
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

train_data = torch.load(DATA_DIR / "train.pt")
val_data = torch.load(DATA_DIR / "val.pt")
test_data = torch.load(DATA_DIR / "test.pt")

splits = {
    "train": train_data,
    "val": val_data,
    "test": test_data
}

print("\nData loaded successfully!")
for split_name, data in splits.items():
    print(f"\n{split_name.upper()} split:")
    for key, tensor in data.items():
        print(f"  {key}: {tensor.shape}")

# Initialize report
report = []
report.append("# Data Quality Analysis Report\n")
report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
report.append(f"**Dataset:** Heart Rate Prediction (Processed Data)\n\n")

# ============================================================================
# SECTION 1: BASIC STATISTICS
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: BASIC STATISTICS")
print("="*80)

report.append("## 1. Basic Statistics\n\n")

for split_name, data in splits.items():
    print(f"\n{split_name.upper()} Split:")
    report.append(f"### {split_name.upper()} Split\n\n")
    
    # Extract data
    speed = data['speed'].numpy().squeeze()  # [N, seq_len]
    altitude = data['altitude'].numpy().squeeze()
    heart_rate = data['heart_rate'].numpy().squeeze()
    original_lengths = data['original_lengths'].numpy().squeeze()
    
    stats = {
        'speed': speed,
        'altitude': altitude,
        'heart_rate': heart_rate,
        'original_lengths': original_lengths
    }
    
    report.append(f"**Sample count:** {len(speed)}\n\n")
    report.append("| Feature | Mean | Std | Min | 25% | 50% | 75% | Max |\n")
    report.append("|---------|------|-----|-----|-----|-----|-----|-----|\n")
    
    for name, values in stats.items():
        if name == 'original_lengths':
            vals = values
        else:
            # For sequences, compute stats across all timesteps
            vals = values.flatten()
        
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        min_val = np.min(vals)
        p25 = np.percentile(vals, 25)
        p50 = np.percentile(vals, 50)
        p75 = np.percentile(vals, 75)
        max_val = np.max(vals)
        
        print(f"  {name:20s}: mean={mean_val:8.2f}, std={std_val:8.2f}, "
              f"min={min_val:8.2f}, max={max_val:8.2f}")
        
        report.append(f"| {name} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | "
                     f"{p25:.2f} | {p50:.2f} | {p75:.2f} | {max_val:.2f} |\n")
    
    report.append("\n")

# ============================================================================
# SECTION 2: NORMALIZATION VERIFICATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: NORMALIZATION VERIFICATION")
print("="*80)

report.append("## 2. Normalization Verification\n\n")

train_speed = train_data['speed'].numpy().squeeze()
train_altitude = train_data['altitude'].numpy().squeeze()

# Expected: normalized data should have mean≈0, std≈1
speed_mean_actual = np.mean(train_speed)
speed_std_actual = np.std(train_speed)
altitude_mean_actual = np.mean(train_altitude)
altitude_std_actual = np.std(train_altitude)

print(f"\nTRAIN set normalized statistics:")
print(f"  Speed:    mean={speed_mean_actual:.6f}, std={speed_std_actual:.6f}")
print(f"  Altitude: mean={altitude_mean_actual:.6f}, std={altitude_std_actual:.6f}")

report.append("### Training Set Normalized Statistics\n\n")
report.append(f"- **Speed:** mean={speed_mean_actual:.6f}, std={speed_std_actual:.6f}\n")
report.append(f"- **Altitude:** mean={altitude_mean_actual:.6f}, std={altitude_std_actual:.6f}\n\n")

# Check if normalization is correct
if abs(speed_mean_actual) < 0.01 and abs(speed_std_actual - 1.0) < 0.1:
    print("  ✓ Speed normalization looks correct")
    report.append("✅ **Speed normalization:** CORRECT\n\n")
else:
    print("  ⚠ Speed normalization may be incorrect!")
    report.append("⚠️ **Speed normalization:** POTENTIAL ISSUE\n\n")
    
if abs(altitude_mean_actual) < 0.01 and abs(altitude_std_actual - 1.0) < 0.1:
    print("  ✓ Altitude normalization looks correct")
    report.append("✅ **Altitude normalization:** CORRECT\n\n")
else:
    print("  ⚠ Altitude normalization may be incorrect!")
    report.append("⚠️ **Altitude normalization:** POTENTIAL ISSUE\n\n")

# ============================================================================
# SECTION 3: OUTLIER DETECTION
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: OUTLIER DETECTION (Z-score > 3)")
print("="*80)

report.append("## 3. Outlier Detection\n\n")

for split_name, data in splits.items():
    print(f"\n{split_name.upper()} Split:")
    report.append(f"### {split_name.upper()} Split\n\n")
    
    speed = data['speed'].numpy().squeeze()
    altitude = data['altitude'].numpy().squeeze()
    heart_rate = data['heart_rate'].numpy().squeeze()
    
    # Compute outliers (|z-score| > 3)
    speed_outliers = np.abs(speed) > 3
    altitude_outliers = np.abs(altitude) > 3
    
    # For heart rate, compute z-scores
    hr_mean = np.mean(heart_rate)
    hr_std = np.std(heart_rate)
    hr_z = (heart_rate - hr_mean) / hr_std
    hr_outliers = np.abs(hr_z) > 3
    
    speed_outlier_pct = 100 * np.sum(speed_outliers) / speed_outliers.size
    altitude_outlier_pct = 100 * np.sum(altitude_outliers) / altitude_outliers.size
    hr_outlier_pct = 100 * np.sum(hr_outliers) / hr_outliers.size
    
    print(f"  Speed:     {np.sum(speed_outliers):6d} outliers ({speed_outlier_pct:.2f}%)")
    print(f"  Altitude:  {np.sum(altitude_outliers):6d} outliers ({altitude_outlier_pct:.2f}%)")
    print(f"  Heart Rate:{np.sum(hr_outliers):6d} outliers ({hr_outlier_pct:.2f}%)")
    
    report.append(f"- **Speed:** {np.sum(speed_outliers)} outliers ({speed_outlier_pct:.2f}%)\n")
    report.append(f"- **Altitude:** {np.sum(altitude_outliers)} outliers ({altitude_outlier_pct:.2f}%)\n")
    report.append(f"- **Heart Rate:** {np.sum(hr_outliers)} outliers ({hr_outlier_pct:.2f}%)\n\n")
    
    # Check for extreme values in heart rate
    hr_min = np.min(heart_rate)
    hr_max = np.max(heart_rate)
    
    if hr_min < 40 or hr_max > 220:
        print(f"  ⚠ WARNING: Heart rate range [{hr_min:.1f}, {hr_max:.1f}] contains physiologically unlikely values!")
        report.append(f"⚠️ **WARNING:** Heart rate range [{hr_min:.1f}, {hr_max:.1f}] may contain errors\n\n")

# ============================================================================
# SECTION 4: DATA LEAKAGE CHECK
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: DATA LEAKAGE CHECK")
print("="*80)

report.append("## 4. Data Leakage Check\n\n")

train_users = set(train_data['userId'].numpy().squeeze().tolist())
val_users = set(val_data['userId'].numpy().squeeze().tolist())
test_users = set(test_data['userId'].numpy().squeeze().tolist())

train_val_overlap = train_users & val_users
train_test_overlap = train_users & test_users
val_test_overlap = val_users & test_users

print(f"\nUnique users per split:")
print(f"  Train: {len(train_users)}")
print(f"  Val:   {len(val_users)}")
print(f"  Test:  {len(test_users)}")
print(f"\nOverlaps:")
print(f"  Train-Val:  {len(train_val_overlap)} users")
print(f"  Train-Test: {len(train_test_overlap)} users")
print(f"  Val-Test:   {len(val_test_overlap)} users")

report.append(f"**Unique users:**\n")
report.append(f"- Train: {len(train_users)}\n")
report.append(f"- Val: {len(val_users)}\n")
report.append(f"- Test: {len(test_users)}\n\n")
report.append(f"**User overlaps:**\n")
report.append(f"- Train-Val: {len(train_val_overlap)} users\n")
report.append(f"- Train-Test: {len(train_test_overlap)} users\n")
report.append(f"- Val-Test: {len(val_test_overlap)} users\n\n")

if len(train_val_overlap) == 0 and len(train_test_overlap) == 0 and len(val_test_overlap) == 0:
    print("  ✓ No data leakage detected!")
    report.append("✅ **No data leakage detected**\n\n")
else:
    print("  ⚠ WARNING: Data leakage detected!")
    report.append("⚠️ **WARNING: Data leakage detected!**\n\n")

# ============================================================================
# SECTION 5: SEQUENCE LENGTH ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: SEQUENCE LENGTH ANALYSIS")
print("="*80)

report.append("## 5. Sequence Length Analysis\n\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (split_name, data) in enumerate(splits.items()):
    original_lengths = data['original_lengths'].numpy().squeeze()
    
    axes[idx].hist(original_lengths, bins=50, edgecolor='black', alpha=0.7)
    axes[idx].axvline(500, color='red', linestyle='--', linewidth=2, label='Padding threshold')
    axes[idx].set_title(f'{split_name.upper()} - Sequence Lengths')
    axes[idx].set_xlabel('Original Length')
    axes[idx].set_ylabel('Count')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)
    
    # Stats
    mean_len = np.mean(original_lengths)
    median_len = np.median(original_lengths)
    padded_pct = 100 * np.sum(original_lengths < 500) / len(original_lengths)
    
    print(f"\n{split_name.upper()}:")
    print(f"  Mean length:   {mean_len:.1f}")
    print(f"  Median length: {median_len:.1f}")
    print(f"  Sequences with padding: {padded_pct:.1f}%")
    
    report.append(f"### {split_name.upper()}\n")
    report.append(f"- Mean: {mean_len:.1f}\n")
    report.append(f"- Median: {median_len:.1f}\n")
    report.append(f"- Padded: {padded_pct:.1f}%\n\n")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sequence_lengths.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'sequence_lengths.png'}")
plt.close()

# ============================================================================
# SECTION 6: FEATURE DISTRIBUTIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: FEATURE DISTRIBUTIONS")
print("="*80)

report.append("## 6. Feature Distributions\n\n")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

features = ['speed', 'altitude', 'heart_rate']
colors = ['blue', 'green', 'red']

for col_idx, (split_name, data) in enumerate(splits.items()):
    for row_idx, (feature, color) in enumerate(zip(features, colors)):
        ax = axes[row_idx, col_idx]
        
        values = data[feature].numpy().squeeze().flatten()
        
        ax.hist(values, bins=100, edgecolor='black', alpha=0.7, color=color)
        ax.set_title(f'{split_name.upper()} - {feature.replace("_", " ").title()}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.2f}')
        ax.legend()

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'feature_distributions.png'}")
plt.close()

# ============================================================================
# SECTION 7: HEART RATE ANALYSIS (CRITICAL)
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: HEART RATE ANALYSIS (TARGET VARIABLE)")
print("="*80)

report.append("## 7. Heart Rate Analysis (Target Variable)\n\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Subplot 1: Distribution across splits
ax = axes[0, 0]
for split_name, data in splits.items():
    hr = data['heart_rate'].numpy().squeeze().flatten()
    ax.hist(hr, bins=100, alpha=0.5, label=split_name, edgecolor='black')
ax.set_title('Heart Rate Distribution by Split')
ax.set_xlabel('Heart Rate (BPM)')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True, alpha=0.3)

# Subplot 2: Box plot
ax = axes[0, 1]
hr_data = [data['heart_rate'].numpy().squeeze().flatten() for data in splits.values()]
ax.boxplot(hr_data, labels=splits.keys())
ax.set_title('Heart Rate Box Plot by Split')
ax.set_ylabel('Heart Rate (BPM)')
ax.grid(True, alpha=0.3)

# Subplot 3: Check for zeros/invalid values
ax = axes[1, 0]
for split_name, data in splits.items():
    hr = data['heart_rate'].numpy().squeeze().flatten()
    zeros = np.sum(hr == 0)
    below_40 = np.sum(hr < 40)
    above_220 = np.sum(hr > 220)
    
    print(f"\n{split_name.upper()} - Invalid heart rates:")
    print(f"  Zeros:    {zeros:6d} ({100*zeros/len(hr):.3f}%)")
    print(f"  < 40 BPM: {below_40:6d} ({100*below_40/len(hr):.3f}%)")
    print(f"  > 220 BPM:{above_220:6d} ({100*above_220/len(hr):.3f}%)")
    
    report.append(f"### {split_name.upper()} - Invalid Values\n")
    report.append(f"- Zeros: {zeros} ({100*zeros/len(hr):.3f}%)\n")
    report.append(f"- < 40 BPM: {below_40} ({100*below_40/len(hr):.3f}%)\n")
    report.append(f"- > 220 BPM: {above_220} ({100*above_220/len(hr):.3f}%)\n\n")

ax.bar(splits.keys(), 
       [np.sum(data['heart_rate'].numpy() == 0) for data in splits.values()],
       alpha=0.7, edgecolor='black')
ax.set_title('Zero Heart Rate Values by Split')
ax.set_ylabel('Count')
ax.grid(True, alpha=0.3)

# Subplot 4: Variance across sequences
ax = axes[1, 1]
for split_name, data in splits.items():
    hr = data['heart_rate'].numpy().squeeze()  # [N, seq_len]
    variances = np.var(hr, axis=1)  # Variance per sequence
    ax.hist(variances, bins=50, alpha=0.5, label=split_name, edgecolor='black')
ax.set_title('Heart Rate Variance per Sequence')
ax.set_xlabel('Variance')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "heart_rate_analysis.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'heart_rate_analysis.png'}")
plt.close()

# ============================================================================
# SECTION 8: SAMPLE TIME SERIES
# ============================================================================
print("\n" + "="*80)
print("SECTION 8: SAMPLE TIME SERIES VISUALIZATION")
print("="*80)

# Plot 5 random samples from training set
train_speed = train_data['speed'].numpy().squeeze()
train_altitude = train_data['altitude'].numpy().squeeze()
train_hr = train_data['heart_rate'].numpy().squeeze()
train_lengths = train_data['original_lengths'].numpy().squeeze()

fig, axes = plt.subplots(5, 3, figsize=(15, 12))

np.random.seed(42)
sample_indices = np.random.choice(len(train_speed), 5, replace=False)

for row_idx, sample_idx in enumerate(sample_indices):
    orig_len = int(train_lengths[sample_idx])
    
    # Speed
    axes[row_idx, 0].plot(train_speed[sample_idx, :orig_len], color='blue', linewidth=1)
    axes[row_idx, 0].axvline(orig_len, color='red', linestyle='--', alpha=0.5)
    axes[row_idx, 0].set_ylabel('Speed (norm)')
    axes[row_idx, 0].grid(True, alpha=0.3)
    if row_idx == 0:
        axes[row_idx, 0].set_title('Speed')
    
    # Altitude
    axes[row_idx, 1].plot(train_altitude[sample_idx, :orig_len], color='green', linewidth=1)
    axes[row_idx, 1].axvline(orig_len, color='red', linestyle='--', alpha=0.5)
    axes[row_idx, 1].set_ylabel('Altitude (norm)')
    axes[row_idx, 1].grid(True, alpha=0.3)
    if row_idx == 0:
        axes[row_idx, 1].set_title('Altitude')
    
    # Heart Rate
    axes[row_idx, 2].plot(train_hr[sample_idx, :orig_len], color='red', linewidth=1)
    axes[row_idx, 2].axvline(orig_len, color='red', linestyle='--', alpha=0.5)
    axes[row_idx, 2].set_ylabel('Heart Rate (BPM)')
    axes[row_idx, 2].grid(True, alpha=0.3)
    if row_idx == 0:
        axes[row_idx, 2].set_title('Heart Rate')

axes[4, 0].set_xlabel('Timestep')
axes[4, 1].set_xlabel('Timestep')
axes[4, 2].set_xlabel('Timestep')

plt.suptitle('Sample Time Series (Red line = original length)', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "sample_time_series.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'sample_time_series.png'}")
plt.close()

# ============================================================================
# SECTION 9: CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 9: CORRELATION ANALYSIS")
print("="*80)

report.append("## 8. Correlation Analysis\n\n")

# Compute correlation on training set
train_speed_flat = train_data['speed'].numpy().squeeze().flatten()
train_altitude_flat = train_data['altitude'].numpy().squeeze().flatten()
train_hr_flat = train_data['heart_rate'].numpy().squeeze().flatten()

# Create correlation matrix
corr_data = np.column_stack([train_speed_flat, train_altitude_flat, train_hr_flat])
corr_matrix = np.corrcoef(corr_data.T)

fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'Heart Rate'])
ax.set_yticklabels(['Speed', 'Altitude', 'Heart Rate'])

# Add correlation values
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{corr_matrix[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=12)

plt.colorbar(im, ax=ax)
ax.set_title('Feature Correlation Matrix (Training Set)')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'correlation_matrix.png'}")
plt.close()

print(f"\nCorrelation with Heart Rate:")
print(f"  Speed:    {corr_matrix[0, 2]:.4f}")
print(f"  Altitude: {corr_matrix[1, 2]:.4f}")

report.append(f"**Correlation with Heart Rate:**\n")
report.append(f"- Speed: {corr_matrix[0, 2]:.4f}\n")
report.append(f"- Altitude: {corr_matrix[1, 2]:.4f}\n\n")

# ============================================================================
# SECTION 10: PADDING ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("SECTION 10: PADDING ANALYSIS")
print("="*80)

report.append("## 9. Padding Analysis\n\n")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, (split_name, data) in enumerate(splits.items()):
    lengths = data['original_lengths'].numpy().squeeze()
    padded_count = np.sum(lengths < 500)
    truncated_count = np.sum(lengths > 500)
    exact_count = np.sum(lengths == 500)
    
    categories = ['Padded\n(< 500)', 'Exact\n(= 500)', 'Truncated\n(> 500)']
    counts = [padded_count, exact_count, truncated_count]
    colors_bar = ['orange', 'green', 'red']
    
    axes[idx].bar(categories, counts, color=colors_bar, alpha=0.7, edgecolor='black')
    axes[idx].set_title(f'{split_name.upper()}')
    axes[idx].set_ylabel('Count')
    axes[idx].grid(True, alpha=0.3, axis='y')
    
    for i, (cat, count) in enumerate(zip(categories, counts)):
        axes[idx].text(i, count, f'{count}\n({100*count/len(lengths):.1f}%)', 
                      ha='center', va='bottom')
    
    print(f"\n{split_name.upper()}:")
    print(f"  Padded (< 500):     {padded_count:5d} ({100*padded_count/len(lengths):5.1f}%)")
    print(f"  Exact (= 500):      {exact_count:5d} ({100*exact_count/len(lengths):5.1f}%)")
    print(f"  Truncated (> 500):  {truncated_count:5d} ({100*truncated_count/len(lengths):5.1f}%)")
    
    report.append(f"### {split_name.upper()}\n")
    report.append(f"- Padded (< 500): {padded_count} ({100*padded_count/len(lengths):.1f}%)\n")
    report.append(f"- Exact (= 500): {exact_count} ({100*exact_count/len(lengths):.1f}%)\n")
    report.append(f"- Truncated (> 500): {truncated_count} ({100*truncated_count/len(lengths):.1f}%)\n\n")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "padding_analysis.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'padding_analysis.png'}")
plt.close()

# ============================================================================
# SAVE REPORT
# ============================================================================
report_path = OUTPUT_DIR / "DATA_QUALITY_REPORT.md"
with open(report_path, "w") as f:
    f.writelines(report)

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print(f"\nReport saved: {report_path}")
print(f"Visualizations saved in: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  - sequence_lengths.png")
print("  - feature_distributions.png")
print("  - heart_rate_analysis.png")
print("  - sample_time_series.png")
print("  - correlation_matrix.png")
print("  - padding_analysis.png")
print("  - DATA_QUALITY_REPORT.md")
print("\n" + "="*80)
