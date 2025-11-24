#!/usr/bin/env python3
"""
Simple analysis: Check if normalization affects correlation.
Uses the processed data to compare normalized vs denormalized correlations.
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

# Paths
OUTPUT_DIR = Path("EDA/EDA_Generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("NORMALIZATION IMPACT ON CORRELATION ANALYSIS")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load processed data
print("Loading processed data...")
train_data = torch.load("DATA/processed/train.pt", weights_only=False)

train_speed_norm = train_data['speed'].numpy().squeeze().flatten()
train_altitude_norm = train_data['altitude'].numpy().squeeze().flatten()
train_hr = train_data['heart_rate'].numpy().squeeze().flatten()

# Load scaler
with open("DATA/processed/scaler_params.json", 'r') as f:
    scaler = json.load(f)

# Denormalize
train_speed_orig = train_speed_norm * scaler['speed_std'] + scaler['speed_mean']
train_altitude_orig = train_altitude_norm * scaler['altitude_std'] + scaler['altitude_mean']

print(f"✓ Loaded {len(train_speed_norm):,} timesteps\n")

# ============================================================================
# COMPUTE CORRELATIONS - ORIGINAL SCALE
# ============================================================================

print("="*80)
print("CORRELATIONS ON ORIGINAL SCALE (DENORMALIZED)")
print("="*80)

corr_data_orig = np.column_stack([train_speed_orig, train_altitude_orig, train_hr])
corr_matrix_orig = np.corrcoef(corr_data_orig.T)

print(f"\n  Speed    → Heart Rate: {corr_matrix_orig[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_orig[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_orig[0, 1]:.6f}")

# Statistics
print(f"\nOriginal scale statistics:")
print(f"  Speed:    mean={np.mean(train_speed_orig):8.2f}, std={np.std(train_speed_orig):8.2f}")
print(f"  Altitude: mean={np.mean(train_altitude_orig):8.2f}, std={np.std(train_altitude_orig):8.2f}")
print(f"  HR:       mean={np.mean(train_hr):8.2f}, std={np.std(train_hr):8.2f}")

# ============================================================================
# COMPUTE CORRELATIONS - NORMALIZED SCALE
# ============================================================================

print("\n" + "="*80)
print("CORRELATIONS ON NORMALIZED SCALE")
print("="*80)

corr_data_norm = np.column_stack([train_speed_norm, train_altitude_norm, train_hr])
corr_matrix_norm = np.corrcoef(corr_data_norm.T)

print(f"\n  Speed    → Heart Rate: {corr_matrix_norm[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_norm[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_norm[0, 1]:.6f}")

# Statistics
print(f"\nNormalized scale statistics:")
print(f"  Speed:    mean={np.mean(train_speed_norm):8.6f}, std={np.std(train_speed_norm):8.6f}")
print(f"  Altitude: mean={np.mean(train_altitude_norm):8.6f}, std={np.std(train_altitude_norm):8.6f}")
print(f"  HR:       mean={np.mean(train_hr):8.2f}, std={np.std(train_hr):8.2f}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("IMPACT OF NORMALIZATION ON CORRELATION")
print("="*80)

print(f"\n{'Correlation':<30} {'Original':>12} {'Normalized':>12} {'Difference':>12}")
print("-" * 70)
print(f"{'Speed → Heart Rate':<30} {corr_matrix_orig[0,2]:>12.6f} {corr_matrix_norm[0,2]:>12.6f} "
      f"{corr_matrix_norm[0,2]-corr_matrix_orig[0,2]:>12.6f}")
print(f"{'Altitude → Heart Rate':<30} {corr_matrix_orig[1,2]:>12.6f} {corr_matrix_norm[1,2]:>12.6f} "
      f"{corr_matrix_norm[1,2]-corr_matrix_orig[1,2]:>12.6f}")
print(f"{'Speed → Altitude':<30} {corr_matrix_orig[0,1]:>12.6f} {corr_matrix_norm[0,1]:>12.6f} "
      f"{corr_matrix_norm[0,1]-corr_matrix_orig[0,1]:>12.6f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Original scale
ax = axes[0, 0]
im = ax.imshow(corr_matrix_orig, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'HR'])
ax.set_yticklabels(['Speed', 'Altitude', 'HR'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr_matrix_orig[i, j]:.3f}',
                ha="center", va="center", color="black", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.set_title('Original Scale Correlations', fontsize=12, fontweight='bold')

ax = axes[0, 1]
sample_indices = np.random.choice(len(train_speed_orig), min(10000, len(train_speed_orig)), replace=False)
ax.scatter(train_speed_orig[sample_indices], train_hr[sample_indices], alpha=0.3, s=1)
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Original: Speed vs HR (r={corr_matrix_orig[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(train_altitude_orig[sample_indices], train_hr[sample_indices], alpha=0.3, s=1, color='green')
ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Original: Altitude vs HR (r={corr_matrix_orig[1,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Row 2: Normalized scale
ax = axes[1, 0]
im = ax.imshow(corr_matrix_norm, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'HR'])
ax.set_yticklabels(['Speed', 'Altitude', 'HR'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr_matrix_norm[i, j]:.3f}',
                ha="center", va="center", color="black", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.set_title('Normalized Scale Correlations', fontsize=12, fontweight='bold')

ax = axes[1, 1]
ax.scatter(train_speed_norm[sample_indices], train_hr[sample_indices], alpha=0.3, s=1, color='orange')
ax.set_xlabel('Speed (normalized)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Normalized: Speed vs HR (r={corr_matrix_norm[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.scatter(train_altitude_norm[sample_indices], train_hr[sample_indices], alpha=0.3, s=1, color='red')
ax.set_xlabel('Altitude (normalized)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Normalized: Altitude vs HR (r={corr_matrix_norm[1,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('NORMALIZATION IMPACT ON CORRELATION', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "normalization_impact_on_correlation.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'normalization_impact_on_correlation.png'}")
plt.close()

# ============================================================================
# KEY FINDINGS
# ============================================================================

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

if abs(corr_matrix_orig[0,2] - corr_matrix_norm[0,2]) < 0.001:
    print("\n✅ Normalization does NOT change correlation (difference < 0.001)")
    print("   This is mathematically expected - linear transformations preserve correlation.")
else:
    print(f"\n⚠️ Normalization changes correlation by {abs(corr_matrix_orig[0,2] - corr_matrix_norm[0,2]):.6f}")
    print("   This is unexpected and may indicate an issue.")

if corr_matrix_orig[0,2] < 0.3:
    print(f"\n🔴 WEAK CORRELATION: Speed-HR correlation is {corr_matrix_orig[0,2]:.3f}")
    print("   This is the ROOT CAUSE of poor model performance.")
    print("   The data simply doesn't have strong predictive power.")
else:
    print(f"\n✅ Moderate correlation: Speed-HR = {corr_matrix_orig[0,2]:.3f}")

if corr_matrix_orig[1,2] < 0.1:
    print(f"\n🔴 NEGLIGIBLE CORRELATION: Altitude-HR correlation is {corr_matrix_orig[1,2]:.3f}")
    print("   Altitude alone is almost useless for predicting heart rate.")

# Generate report
report = []
report.append("# Normalization Impact on Correlation\n\n")
report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

report.append("## Summary\n\n")
report.append(f"Analyzed {len(train_speed_norm):,} timesteps from processed training data.\n\n")

report.append("## Correlation Comparison\n\n")
report.append("| Feature Pair | Original Scale | Normalized Scale | Difference |\n")
report.append("|--------------|----------------|------------------|------------|\n")
report.append(f"| Speed → Heart Rate | {corr_matrix_orig[0,2]:.6f} | {corr_matrix_norm[0,2]:.6f} | "
             f"{corr_matrix_norm[0,2]-corr_matrix_orig[0,2]:+.6f} |\n")
report.append(f"| Altitude → Heart Rate | {corr_matrix_orig[1,2]:.6f} | {corr_matrix_norm[1,2]:.6f} | "
             f"{corr_matrix_norm[1,2]-corr_matrix_orig[1,2]:+.6f} |\n\n")

report.append("## Key Findings\n\n")

if abs(corr_matrix_orig[0,2] - corr_matrix_norm[0,2]) < 0.001:
    report.append("✅ **Normalization preserves correlation** (difference < 0.001)\n\n")
    report.append("This is expected - Z-score normalization is a linear transformation, which does not affect correlation coefficients.\n\n")

if corr_matrix_orig[0,2] < 0.3:
    report.append(f"🔴 **ROOT CAUSE IDENTIFIED: Weak Correlation**\n\n")
    report.append(f"- Speed → Heart Rate correlation = **{corr_matrix_orig[0,2]:.3f}** (weak)\n")
    report.append(f"- Altitude → Heart Rate correlation = **{corr_matrix_orig[1,2]:.3f}** (negligible)\n\n")
    report.append("**Conclusion:** The input features have **inherently weak predictive power** for heart rate. ")
    report.append("This explains why models struggle to achieve high accuracy.\n\n")

report.append("## Recommendations\n\n")
report.append("Since normalization is not the issue, focus on:\n\n")
report.append("1. **Feature engineering** - Create derived features with stronger correlations\n")
report.append("2. **User embeddings** - User fitness level is likely the strongest predictor\n")
report.append("3. **Temporal features** - Lagged speed/altitude, acceleration, rolling averages\n")
report.append("4. **Lower performance expectations** - With weak correlations, MAE < 10 BPM may be realistic ceiling\n\n")

report_path = OUTPUT_DIR / "NORMALIZATION_IMPACT_ANALYSIS.md"
with open(report_path, "w") as f:
    f.writelines(report)

print(f"✓ Report saved: {report_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
