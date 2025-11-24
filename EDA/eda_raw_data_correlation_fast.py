#!/usr/bin/env python3
"""
Fast correlation analysis on raw endomondoHR_proper.json data.
Samples first N valid workouts for quick comparison.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import ast

# Set style
sns.set_style("whitegrid")

# Paths
RAW_DATA_FILE = Path("DATA/endomondoHR_proper.json")
OUTPUT_DIR = Path("EDA/EDA_Generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Sample size for analysis
MAX_WORKOUTS = 200000  # Analyze first 25k workouts (should be enough)

print("="*80)
print("RAW DATA CORRELATION ANALYSIS (FAST VERSION)")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Analyzing first {MAX_WORKOUTS:,} valid workouts from: {RAW_DATA_FILE}")

# ============================================================================
# LOAD AND FILTER WORKOUTS
# ============================================================================

def parse_workout(line):
    """Parse a single workout JSON line."""
    try:
        return json.loads(line)
    except:
        try:
            return ast.literal_eval(line)
        except:
            return None

def is_valid_workout(workout):
    """Quick validation."""
    try:
        if workout.get('sport') != 'run':
            return False
        
        if not all(k in workout for k in ['speed', 'altitude', 'heart_rate']):
            return False
        
        speed = np.array(workout['speed'], dtype=float)
        altitude = np.array(workout['altitude'], dtype=float)
        hr = np.array(workout['heart_rate'], dtype=float)
        
        if len(speed) < 50 or len(altitude) != len(speed) or len(hr) != len(speed):
            return False
        
        if not (np.isfinite(speed).all() and np.isfinite(altitude).all() and np.isfinite(hr).all()):
            return False
        
        if np.any(hr < 40) or np.any(hr > 220):
            return False
        
        return True
    except:
        return False

print("\nLoading and filtering workouts...")

workouts = []
lines_processed = 0

with open(RAW_DATA_FILE, 'r') as f:
    for line in f:
        lines_processed += 1
        
        if lines_processed % 10000 == 0:
            print(f"\rProcessed {lines_processed:,} lines, found {len(workouts):,} valid workouts...", end='', flush=True)
        
        if len(workouts) >= MAX_WORKOUTS:
            break
        
        workout = parse_workout(line.strip())
        if workout and is_valid_workout(workout):
            workouts.append(workout)

print(f"\n\n✓ Loaded {len(workouts):,} valid workouts from {lines_processed:,} lines")

# ============================================================================
# EXTRACT FEATURES AND COMPUTE CORRELATIONS
# ============================================================================

print("\nExtracting features from all valid workouts...")

all_speed = []
all_altitude = []
all_hr = []

for i, workout in enumerate(workouts):
    if i % 5000 == 0:
        print(f"\rProcessing workout {i:,}/{len(workouts):,}...", end='', flush=True)
    
    speed = np.array(workout['speed'], dtype=float)
    altitude = np.array(workout['altitude'], dtype=float)
    hr = np.array(workout['heart_rate'], dtype=float)
    
    all_speed.extend(speed.tolist())
    all_altitude.extend(altitude.tolist())
    all_hr.extend(hr.tolist())

print(f"\n\n✓ Extracted {len(all_speed):,} total timesteps")

# Convert to numpy
all_speed = np.array(all_speed)
all_altitude = np.array(all_altitude)
all_hr = np.array(all_hr)

# Statistics
print("\n" + "="*80)
print("RAW DATA STATISTICS")
print("="*80)
print(f"\nSpeed:     mean={np.mean(all_speed):8.2f}, std={np.std(all_speed):8.2f}, "
      f"min={np.min(all_speed):8.2f}, max={np.max(all_speed):8.2f}")
print(f"Altitude:  mean={np.mean(all_altitude):8.2f}, std={np.std(all_altitude):8.2f}, "
      f"min={np.min(all_altitude):8.2f}, max={np.max(all_altitude):8.2f}")
print(f"Heart Rate:mean={np.mean(all_hr):8.2f}, std={np.std(all_hr):8.2f}, "
      f"min={np.min(all_hr):8.2f}, max={np.max(all_hr):8.2f}")

# Compute correlations
print("\n" + "="*80)
print("RAW DATA CORRELATIONS")
print("="*80)

corr_data = np.column_stack([all_speed, all_altitude, all_hr])
corr_matrix_raw = np.corrcoef(corr_data.T)

print(f"\n  Speed    → Heart Rate: {corr_matrix_raw[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_raw[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_raw[0, 1]:.6f}")

# ============================================================================
# LOAD PROCESSED DATA FOR COMPARISON
# ============================================================================

print("\n" + "="*80)
print("PROCESSED DATA COMPARISON")
print("="*80)

import torch

train_data = torch.load("DATA/processed/train.pt", weights_only=False)

train_speed = train_data['speed'].numpy().squeeze().flatten()
train_altitude = train_data['altitude'].numpy().squeeze().flatten()
train_hr = train_data['heart_rate'].numpy().squeeze().flatten()

# Denormalize
with open("DATA/processed/scaler_params.json", 'r') as f:
    scaler = json.load(f)

train_speed_orig = train_speed * scaler['speed_std'] + scaler['speed_mean']
train_altitude_orig = train_altitude * scaler['altitude_std'] + scaler['altitude_mean']

print(f"\n✓ Loaded {len(train_speed):,} timesteps from processed data")

# Compute correlations on processed (denormalized)
corr_data_processed = np.column_stack([train_speed_orig, train_altitude_orig, train_hr])
corr_matrix_processed = np.corrcoef(corr_data_processed.T)

print(f"\nPROCESSED DATA CORRELATIONS (denormalized):")
print(f"  Speed    → Heart Rate: {corr_matrix_processed[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_processed[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_processed[0, 1]:.6f}")

# Also show normalized
corr_data_norm = np.column_stack([train_speed, train_altitude, train_hr])
corr_matrix_norm = np.corrcoef(corr_data_norm.T)

print(f"\nPROCESSED DATA CORRELATIONS (normalized):")
print(f"  Speed    → Heart Rate: {corr_matrix_norm[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_norm[1, 2]:.6f}")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

print(f"\n{'Correlation':<30} {'Raw Data':>12} {'Processed':>12} {'Difference':>12}")
print("-" * 70)
print(f"{'Speed → Heart Rate':<30} {corr_matrix_raw[0,2]:>12.6f} {corr_matrix_processed[0,2]:>12.6f} "
      f"{corr_matrix_processed[0,2]-corr_matrix_raw[0,2]:>12.6f}")
print(f"{'Altitude → Heart Rate':<30} {corr_matrix_raw[1,2]:>12.6f} {corr_matrix_processed[1,2]:>12.6f} "
      f"{corr_matrix_processed[1,2]-corr_matrix_raw[1,2]:>12.6f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Raw data
ax = axes[0, 0]
im = ax.imshow(corr_matrix_raw, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'HR'])
ax.set_yticklabels(['Speed', 'Altitude', 'HR'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr_matrix_raw[i, j]:.3f}',
                ha="center", va="center", color="black", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.set_title('Raw Data Correlations', fontsize=12, fontweight='bold')

ax = axes[0, 1]
sample_indices = np.random.choice(len(all_speed), min(10000, len(all_speed)), replace=False)
ax.scatter(all_speed[sample_indices], all_hr[sample_indices], alpha=0.3, s=1)
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Raw: Speed vs HR (r={corr_matrix_raw[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.scatter(all_altitude[sample_indices], all_hr[sample_indices], alpha=0.3, s=1, color='green')
ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Raw: Altitude vs HR (r={corr_matrix_raw[1,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Row 2: Processed data
ax = axes[1, 0]
im = ax.imshow(corr_matrix_processed, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'HR'])
ax.set_yticklabels(['Speed', 'Altitude', 'HR'])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{corr_matrix_processed[i, j]:.3f}',
                ha="center", va="center", color="black", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.set_title('Processed Data Correlations', fontsize=12, fontweight='bold')

ax = axes[1, 1]
sample_indices_proc = np.random.choice(len(train_speed_orig), min(10000, len(train_speed_orig)), replace=False)
ax.scatter(train_speed_orig[sample_indices_proc], train_hr[sample_indices_proc], alpha=0.3, s=1, color='orange')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Processed: Speed vs HR (r={corr_matrix_processed[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.scatter(train_altitude_orig[sample_indices_proc], train_hr[sample_indices_proc], alpha=0.3, s=1, color='red')
ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Processed: Altitude vs HR (r={corr_matrix_processed[1,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.suptitle('RAW vs PROCESSED DATA CORRELATION COMPARISON', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "correlation_comparison_raw_vs_processed.png", dpi=150, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'correlation_comparison_raw_vs_processed.png'}")
plt.close()

# ============================================================================
# GENERATE REPORT
# ============================================================================

report = []
report.append("# RAW vs PROCESSED DATA - CORRELATION ANALYSIS\n\n")
report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
report.append(f"**Raw workouts analyzed:** {len(workouts):,}\n")
report.append(f"**Total timesteps (raw):** {len(all_speed):,}\n")
report.append(f"**Processed timesteps:** {len(train_speed):,}\n\n")

report.append("## Correlation Comparison\n\n")
report.append("| Feature Pair | Raw Data | Processed Data | Difference |\n")
report.append("|--------------|----------|----------------|------------|\n")
report.append(f"| Speed → Heart Rate | {corr_matrix_raw[0,2]:.6f} | {corr_matrix_processed[0,2]:.6f} | "
             f"{corr_matrix_processed[0,2]-corr_matrix_raw[0,2]:+.6f} |\n")
report.append(f"| Altitude → Heart Rate | {corr_matrix_raw[1,2]:.6f} | {corr_matrix_processed[1,2]:.6f} | "
             f"{corr_matrix_processed[1,2]-corr_matrix_raw[1,2]:+.6f} |\n")
report.append(f"| Speed → Altitude | {corr_matrix_raw[0,1]:.6f} | {corr_matrix_processed[0,1]:.6f} | "
             f"{corr_matrix_processed[0,1]-corr_matrix_raw[0,1]:+.6f} |\n\n")

report.append("## Key Findings\n\n")

if abs(corr_matrix_raw[0,2] - corr_matrix_processed[0,2]) < 0.05:
    report.append("✅ **Preprocessing preserves correlations well** - Difference < 0.05\n\n")
else:
    report.append("⚠️ **Preprocessing affects correlations** - Difference = "
                 f"{abs(corr_matrix_raw[0,2] - corr_matrix_processed[0,2]):.3f}\n\n")

if corr_matrix_raw[0,2] < 0.3:
    report.append(f"🔴 **Weak correlation confirmed in raw data** - Speed-HR = {corr_matrix_raw[0,2]:.3f}\n\n")
    report.append("The weak correlation is **inherent to the data**, not a preprocessing artifact.\n\n")
else:
    report.append(f"✅ **Moderate correlation in raw data** - Speed-HR = {corr_matrix_raw[0,2]:.3f}\n\n")

report.append("## Recommendations\n\n")
report.append("1. The correlation issue is **fundamental to the dataset**\n")
report.append("2. Consider feature engineering:\n")
report.append("   - Speed changes (acceleration)\n")
report.append("   - Rolling averages\n")
report.append("   - Lagged features\n")
report.append("   - User-specific embeddings\n\n")

report_path = OUTPUT_DIR / "RAW_VS_PROCESSED_CORRELATION.md"
with open(report_path, "w") as f:
    f.writelines(report)

print(f"✓ Report saved: {report_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\n" + "="*80)
