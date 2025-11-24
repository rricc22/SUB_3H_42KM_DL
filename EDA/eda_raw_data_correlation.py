#!/usr/bin/env python3
"""
EDA script for raw endomondoHR_proper.json data.
Analyzes feature-target correlations in raw data and compares with processed data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import ast
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Paths
RAW_DATA_FILE = Path("DATA/endomondoHR_proper.json")
OUTPUT_DIR = Path("EDA/EDA_Generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("RAW DATA CORRELATION ANALYSIS")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Analyzing: {RAW_DATA_FILE}")

# ============================================================================
# SECTION 1: LOAD AND PARSE RAW DATA
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: LOADING RAW DATA")
print("="*80)

def parse_workout(line):
    """Parse a single workout JSON line."""
    try:
        data = json.loads(line)
        return data
    except:
        try:
            # Try with ast.literal_eval for single quotes
            data = ast.literal_eval(line)
            return data
        except:
            return None

# Sample workouts (for faster processing)
# Use reservoir sampling to get representative sample
SAMPLE_SIZE = 140000  # Sample 50k workouts (should cover all valid ones)

print(f"\nSampling up to {SAMPLE_SIZE:,} workouts from raw JSON...")
print("(Using reservoir sampling for representative distribution)")

workouts = []
valid_count = 0
invalid_count = 0
total_lines = 0

with open(RAW_DATA_FILE, 'r') as f:
    for line in tqdm(f, desc="Sampling workouts"):
        total_lines += 1
        workout = parse_workout(line.strip())
        
        if workout is not None:
            if len(workouts) < SAMPLE_SIZE:
                workouts.append(workout)
            else:
                # Reservoir sampling: randomly replace existing item
                import random
                j = random.randint(0, valid_count)
                if j < SAMPLE_SIZE:
                    workouts[j] = workout
            valid_count += 1
        else:
            invalid_count += 1

print(f"\n✓ Sampled {len(workouts)} workouts from {total_lines:,} total lines")
print(f"  Valid workouts found: {valid_count:,}")
if invalid_count > 0:
    print(f"  (Skipped {invalid_count:,} invalid lines)")

# ============================================================================
# SECTION 2: FILTER WORKOUTS (SAME AS PREPROCESSING)
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: FILTERING WORKOUTS")
print("="*80)

def filter_workout(workout):
    """Apply same filters as preprocessing."""
    try:
        # Filter 1: Must be 'run' sport
        if workout.get('sport') != 'run':
            return False, "not_run"
        
        # Filter 2: Must have required fields
        required_fields = ['speed', 'altitude', 'heart_rate', 'gender', 'userId']
        for field in required_fields:
            if field not in workout:
                return False, f"missing_{field}"
        
        # Filter 3: Check if arrays exist and are lists
        for field in ['speed', 'altitude', 'heart_rate']:
            if not isinstance(workout[field], list):
                return False, f"invalid_{field}_type"
        
        # Filter 4: Check if arrays have same length
        length = len(workout['speed'])
        if len(workout['altitude']) != length or len(workout['heart_rate']) != length:
            return False, "length_mismatch"
        
        # Filter 5: Minimum length (50 timesteps)
        if length < 50:
            return False, "too_short"
        
        # Filter 6: Check for valid values (no NaN/Inf)
        speed = np.array(workout['speed'], dtype=float)
        altitude = np.array(workout['altitude'], dtype=float)
        hr = np.array(workout['heart_rate'], dtype=float)
        
        if not (np.isfinite(speed).all() and np.isfinite(altitude).all() and np.isfinite(hr).all()):
            return False, "invalid_values"
        
        # Filter 7: Heart rate in valid range
        if np.any(hr < 40) or np.any(hr > 220):
            return False, "hr_out_of_range"
        
        return True, "valid"
    except Exception as e:
        return False, f"error_{str(e)[:20]}"

# Apply filters
print("\nApplying filters...")
filtered_workouts = []
filter_stats = {}

for workout in tqdm(workouts, desc="Filtering"):
    is_valid, reason = filter_workout(workout)
    
    if is_valid:
        filtered_workouts.append(workout)
    
    filter_stats[reason] = filter_stats.get(reason, 0) + 1

print(f"\n✓ {filter_stats.get('valid', 0)} workouts passed all filters")
print(f"✗ {len(workouts) - filter_stats.get('valid', 0)} workouts filtered out")

print("\nFilter breakdown:")
for reason, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
    if reason != 'valid':
        print(f"  {reason:25s}: {count:6d} workouts")

# ============================================================================
# SECTION 3: EXTRACT FEATURES AND COMPUTE CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: COMPUTING CORRELATIONS ON RAW DATA")
print("="*80)

print("\nExtracting features from all valid workouts...")

# Collect all timesteps from all workouts
all_speed = []
all_altitude = []
all_hr = []

for workout in tqdm(filtered_workouts, desc="Extracting features"):
    speed = np.array(workout['speed'], dtype=float)
    altitude = np.array(workout['altitude'], dtype=float)
    hr = np.array(workout['heart_rate'], dtype=float)
    
    all_speed.extend(speed.tolist())
    all_altitude.extend(altitude.tolist())
    all_hr.extend(hr.tolist())

# Convert to numpy arrays
all_speed = np.array(all_speed)
all_altitude = np.array(all_altitude)
all_hr = np.array(all_hr)

print(f"\n✓ Extracted {len(all_speed):,} total timesteps")

# Compute statistics
print("\nRaw data statistics:")
print(f"  Speed:     mean={np.mean(all_speed):8.2f}, std={np.std(all_speed):8.2f}, "
      f"min={np.min(all_speed):8.2f}, max={np.max(all_speed):8.2f}")
print(f"  Altitude:  mean={np.mean(all_altitude):8.2f}, std={np.std(all_altitude):8.2f}, "
      f"min={np.min(all_altitude):8.2f}, max={np.max(all_altitude):8.2f}")
print(f"  Heart Rate:mean={np.mean(all_hr):8.2f}, std={np.std(all_hr):8.2f}, "
      f"min={np.min(all_hr):8.2f}, max={np.max(all_hr):8.2f}")

# Compute correlation matrix
print("\nComputing correlations...")
corr_data = np.column_stack([all_speed, all_altitude, all_hr])
corr_matrix_raw = np.corrcoef(corr_data.T)

print(f"\nRAW DATA CORRELATIONS:")
print(f"  Speed    → Heart Rate: {corr_matrix_raw[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_raw[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_raw[0, 1]:.6f}")

# ============================================================================
# SECTION 4: COMPUTE CORRELATION ON PROCESSED DATA (FOR COMPARISON)
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: LOADING PROCESSED DATA FOR COMPARISON")
print("="*80)

import torch

train_data = torch.load("DATA/processed/train.pt", weights_only=False)

train_speed = train_data['speed'].numpy().squeeze().flatten()
train_altitude = train_data['altitude'].numpy().squeeze().flatten()
train_hr = train_data['heart_rate'].numpy().squeeze().flatten()

# Note: processed data is normalized, so compute on original scale
# Load scaler params
with open("DATA/processed/scaler_params.json", 'r') as f:
    scaler = json.load(f)

# Denormalize
train_speed_orig = train_speed * scaler['speed_std'] + scaler['speed_mean']
train_altitude_orig = train_altitude * scaler['altitude_std'] + scaler['altitude_mean']

print(f"\n✓ Loaded {len(train_speed):,} timesteps from processed training data")

# Compute correlation on processed (denormalized)
corr_data_processed = np.column_stack([train_speed_orig, train_altitude_orig, train_hr])
corr_matrix_processed = np.corrcoef(corr_data_processed.T)

print(f"\nPROCESSED DATA CORRELATIONS (denormalized):")
print(f"  Speed    → Heart Rate: {corr_matrix_processed[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_processed[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_processed[0, 1]:.6f}")

# Also show normalized correlations
corr_data_norm = np.column_stack([train_speed, train_altitude, train_hr])
corr_matrix_norm = np.corrcoef(corr_data_norm.T)

print(f"\nPROCESSED DATA CORRELATIONS (normalized):")
print(f"  Speed    → Heart Rate: {corr_matrix_norm[0, 2]:.6f}")
print(f"  Altitude → Heart Rate: {corr_matrix_norm[1, 2]:.6f}")
print(f"  Speed    → Altitude:   {corr_matrix_norm[0, 1]:.6f}")

# ============================================================================
# SECTION 5: COMPARISON AND VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: CORRELATION COMPARISON")
print("="*80)

# Summary comparison
print(f"\n{'Correlation':<30} {'Raw Data':>12} {'Processed':>12} {'Difference':>12}")
print("-" * 70)
print(f"{'Speed → Heart Rate':<30} {corr_matrix_raw[0,2]:>12.6f} {corr_matrix_processed[0,2]:>12.6f} "
      f"{corr_matrix_processed[0,2]-corr_matrix_raw[0,2]:>12.6f}")
print(f"{'Altitude → Heart Rate':<30} {corr_matrix_raw[1,2]:>12.6f} {corr_matrix_processed[1,2]:>12.6f} "
      f"{corr_matrix_processed[1,2]-corr_matrix_raw[1,2]:>12.6f}")

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Row 1: Raw data correlations
# Correlation matrix - Raw
ax = axes[0, 0]
im = ax.imshow(corr_matrix_raw, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'HR'])
ax.set_yticklabels(['Speed', 'Altitude', 'HR'])
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{corr_matrix_raw[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.set_title('Raw Data Correlations', fontsize=12, fontweight='bold')

# Scatter: Speed vs HR - Raw
ax = axes[0, 1]
sample_indices = np.random.choice(len(all_speed), min(10000, len(all_speed)), replace=False)
ax.scatter(all_speed[sample_indices], all_hr[sample_indices], alpha=0.3, s=1)
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Raw: Speed vs HR (r={corr_matrix_raw[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Scatter: Altitude vs HR - Raw
ax = axes[0, 2]
ax.scatter(all_altitude[sample_indices], all_hr[sample_indices], alpha=0.3, s=1, color='green')
ax.set_xlabel('Altitude (m)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Raw: Altitude vs HR (r={corr_matrix_raw[1,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Row 2: Processed data correlations
# Correlation matrix - Processed
ax = axes[1, 0]
im = ax.imshow(corr_matrix_processed, cmap='coolwarm', vmin=-1, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Speed', 'Altitude', 'HR'])
ax.set_yticklabels(['Speed', 'Altitude', 'HR'])
for i in range(3):
    for j in range(3):
        text = ax.text(j, i, f'{corr_matrix_processed[i, j]:.3f}',
                      ha="center", va="center", color="black", fontsize=11, fontweight='bold')
plt.colorbar(im, ax=ax)
ax.set_title('Processed Data Correlations', fontsize=12, fontweight='bold')

# Scatter: Speed vs HR - Processed
ax = axes[1, 1]
sample_indices_proc = np.random.choice(len(train_speed_orig), min(10000, len(train_speed_orig)), replace=False)
ax.scatter(train_speed_orig[sample_indices_proc], train_hr[sample_indices_proc], alpha=0.3, s=1, color='orange')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Processed: Speed vs HR (r={corr_matrix_processed[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Scatter: Altitude vs HR - Processed
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
# SECTION 6: TEMPORAL CORRELATION ANALYSIS (LAG ANALYSIS)
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: TEMPORAL CORRELATION ANALYSIS (LAG)")
print("="*80)

print("\nAnalyzing temporal correlations (how speed affects HR at different time lags)...")

# Sample a subset of workouts for lag analysis
sample_workouts = np.random.choice(filtered_workouts, min(1000, len(filtered_workouts)), replace=False)

# Compute lag correlations
max_lag = 30  # Check up to 30 timesteps ahead
lag_corrs_speed = []
lag_corrs_altitude = []

for lag in tqdm(range(0, max_lag + 1), desc="Computing lag correlations"):
    speed_vals = []
    altitude_vals = []
    hr_vals = []
    
    for workout in sample_workouts:
        speed = np.array(workout['speed'], dtype=float)
        altitude = np.array(workout['altitude'], dtype=float)
        hr = np.array(workout['heart_rate'], dtype=float)
        
        if len(speed) > lag:
            speed_vals.extend(speed[:-lag if lag > 0 else None].tolist())
            altitude_vals.extend(altitude[:-lag if lag > 0 else None].tolist())
            hr_vals.extend(hr[lag:].tolist())
    
    if len(speed_vals) > 0:
        corr_speed = np.corrcoef(speed_vals, hr_vals)[0, 1]
        corr_altitude = np.corrcoef(altitude_vals, hr_vals)[0, 1]
        lag_corrs_speed.append(corr_speed)
        lag_corrs_altitude.append(corr_altitude)
    else:
        lag_corrs_speed.append(0)
        lag_corrs_altitude.append(0)

# Find best lag
best_lag_speed = np.argmax(lag_corrs_speed)
best_lag_altitude = np.argmax(lag_corrs_altitude)

print(f"\nLag analysis results:")
print(f"  Speed    → HR: Best correlation {lag_corrs_speed[best_lag_speed]:.6f} at lag={best_lag_speed}")
print(f"  Altitude → HR: Best correlation {lag_corrs_altitude[best_lag_altitude]:.6f} at lag={best_lag_altitude}")

# Visualize lag correlations
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(max_lag + 1), lag_corrs_speed, marker='o', label='Speed → HR', linewidth=2)
ax.plot(range(max_lag + 1), lag_corrs_altitude, marker='s', label='Altitude → HR', linewidth=2)
ax.axvline(best_lag_speed, color='blue', linestyle='--', alpha=0.5, label=f'Best Speed lag={best_lag_speed}')
ax.axvline(best_lag_altitude, color='orange', linestyle='--', alpha=0.5, label=f'Best Altitude lag={best_lag_altitude}')
ax.set_xlabel('Time Lag (timesteps)', fontsize=12)
ax.set_ylabel('Correlation Coefficient', fontsize=12)
ax.set_title('Temporal Correlation: Feature vs Heart Rate at Different Time Lags', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "temporal_lag_correlation.png", dpi=150, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / 'temporal_lag_correlation.png'}")
plt.close()

# ============================================================================
# SECTION 7: GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("SECTION 7: GENERATING SUMMARY REPORT")
print("="*80)

report = []
report.append("# RAW vs PROCESSED DATA - CORRELATION ANALYSIS\n\n")
report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

report.append("## Summary\n\n")
report.append(f"- **Raw workouts analyzed:** {len(filtered_workouts):,}\n")
report.append(f"- **Total timesteps (raw):** {len(all_speed):,}\n")
report.append(f"- **Processed timesteps:** {len(train_speed):,}\n\n")

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

# Check if correlations are similar
if abs(corr_matrix_raw[0,2] - corr_matrix_processed[0,2]) < 0.05:
    report.append("✅ **Preprocessing preserves correlations well** - Speed-HR correlation difference < 0.05\n\n")
else:
    report.append("⚠️ **Preprocessing may affect correlations** - Speed-HR correlation differs by "
                 f"{abs(corr_matrix_raw[0,2] - corr_matrix_processed[0,2]):.3f}\n\n")

if corr_matrix_raw[0,2] < 0.3:
    report.append("🔴 **Weak correlation in raw data confirmed** - Speed-HR correlation is {:.3f} in original data\n\n".format(
        corr_matrix_raw[0,2]))
    report.append("This confirms that the weak correlation is **inherent to the data**, not caused by preprocessing.\n\n")

report.append("## Temporal Lag Analysis\n\n")
report.append(f"- **Best Speed lag:** {best_lag_speed} timesteps (correlation: {lag_corrs_speed[best_lag_speed]:.6f})\n")
report.append(f"- **Best Altitude lag:** {best_lag_altitude} timesteps (correlation: {lag_corrs_altitude[best_lag_altitude]:.6f})\n\n")

if best_lag_speed > 0:
    report.append(f"💡 **Insight:** Heart rate responds to speed changes with a **{best_lag_speed}-timestep delay**. "
                 f"Consider adding lagged features to your model.\n\n")

report.append("## Recommendations\n\n")
report.append("1. The weak correlation is **inherent to the dataset**, not a preprocessing artifact\n")
report.append("2. Consider adding **lagged features** (speed[t-k] to predict HR[t])\n")
report.append("3. Feature engineering suggestions:\n")
report.append("   - Speed changes (acceleration/deceleration)\n")
report.append("   - Rolling averages of speed and altitude\n")
report.append("   - Cumulative elevation gain\n")
report.append("   - User-specific features (fitness level, historical average HR)\n\n")

report_path = OUTPUT_DIR / "RAW_VS_PROCESSED_CORRELATION.md"
with open(report_path, "w") as f:
    f.writelines(report)

print(f"\n✓ Report saved: {report_path}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - correlation_comparison_raw_vs_processed.png")
print("  - temporal_lag_correlation.png")
print("  - RAW_VS_PROCESSED_CORRELATION.md")
print("\n" + "="*80)
