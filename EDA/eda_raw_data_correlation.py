#!/usr/bin/env python3
"""
EDA script for raw endomondoHR_proper.json data.
Analyzes feature-target correlations in raw data and compares with processed data.
STREAMING VERSION - Low RAM usage for large datasets.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import ast
from tqdm import tqdm
import gc

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Paths
RAW_DATA_FILE = Path("DATA/endomondoHR_proper.json")
OUTPUT_DIR = Path("EDA/EDA_Generation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("RAW DATA CORRELATION ANALYSIS (STREAMING)")
print("="*80)
print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Analyzing: {RAW_DATA_FILE}")

# ============================================================================
# SECTION 1: STREAMING CORRELATION COMPUTATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 1: STREAMING THROUGH RAW DATA")
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

class StreamingCorrelation:
    """Compute correlation incrementally without storing all data."""
    def __init__(self):
        self.n = 0
        self.mean_x = 0.0
        self.mean_y = 0.0
        self.M2_x = 0.0
        self.M2_y = 0.0
        self.cov_xy = 0.0
        self.min_x = float('inf')
        self.max_x = float('-inf')
        self.min_y = float('inf')
        self.max_y = float('-inf')
    
    def update(self, x_values, y_values):
        """Update statistics with new batch of values."""
        for x, y in zip(x_values, y_values):
            self.n += 1
            delta_x = x - self.mean_x
            delta_y = y - self.mean_y
            
            self.mean_x += delta_x / self.n
            self.mean_y += delta_y / self.n
            
            delta_x2 = x - self.mean_x
            delta_y2 = y - self.mean_y
            
            self.M2_x += delta_x * delta_x2
            self.M2_y += delta_y * delta_y2
            self.cov_xy += delta_x * delta_y2
            
            self.min_x = min(self.min_x, x)
            self.max_x = max(self.max_x, x)
            self.min_y = min(self.min_y, y)
            self.max_y = max(self.max_y, y)
    
    def correlation(self):
        """Compute Pearson correlation coefficient."""
        if self.n < 2:
            return 0.0
        var_x = self.M2_x / (self.n - 1)
        var_y = self.M2_y / (self.n - 1)
        cov = self.cov_xy / (self.n - 1)
        
        if var_x == 0 or var_y == 0:
            return 0.0
        
        return cov / (np.sqrt(var_x) * np.sqrt(var_y))
    
    def std(self, is_x=True):
        """Compute standard deviation."""
        if self.n < 2:
            return 0.0
        M2 = self.M2_x if is_x else self.M2_y
        return np.sqrt(M2 / (self.n - 1))

def count_lines(filepath):
    """Count total lines for progress bar."""
    print("Counting total workouts...")
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

print("\nInitializing streaming correlation computation...")

# Streaming correlation objects
speed_hr_corr = StreamingCorrelation()
altitude_hr_corr = StreamingCorrelation()
speed_altitude_corr = StreamingCorrelation()

# Counters
valid_count = 0
invalid_count = 0
filter_stats = {}

# For scatter plot sampling (reservoir sampling)
MAX_SCATTER_SAMPLES = 10000
scatter_samples = {'speed': [], 'altitude': [], 'hr': []}
scatter_count = 0

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

# Stream through file and compute correlations
total_lines = count_lines(RAW_DATA_FILE)
print(f"Total workouts in file: {total_lines:,}")

print("\nStreaming through data and computing correlations...")
with open(RAW_DATA_FILE, 'r') as f:
    for line in tqdm(f, total=total_lines, desc="Processing"):
        workout = parse_workout(line.strip())
        
        if workout is None:
            invalid_count += 1
            continue
        
        is_valid, reason = filter_workout(workout)
        filter_stats[reason] = filter_stats.get(reason, 0) + 1
        
        if is_valid:
            valid_count += 1
            
            # Extract arrays
            speed = np.array(workout['speed'], dtype=float)
            altitude = np.array(workout['altitude'], dtype=float)
            hr = np.array(workout['heart_rate'], dtype=float)
            
            # Update streaming correlations
            speed_hr_corr.update(speed, hr)
            altitude_hr_corr.update(altitude, hr)
            speed_altitude_corr.update(speed, altitude)
            
            # Reservoir sampling for scatter plots
            for i in range(len(speed)):
                scatter_count += 1
                if len(scatter_samples['speed']) < MAX_SCATTER_SAMPLES:
                    scatter_samples['speed'].append(float(speed[i]))
                    scatter_samples['altitude'].append(float(altitude[i]))
                    scatter_samples['hr'].append(float(hr[i]))
                else:
                    # Random replacement
                    import random
                    j = random.randint(0, scatter_count - 1)
                    if j < MAX_SCATTER_SAMPLES:
                        scatter_samples['speed'][j] = float(speed[i])
                        scatter_samples['altitude'][j] = float(altitude[i])
                        scatter_samples['hr'][j] = float(hr[i])

print(f"\n✓ {valid_count} workouts passed all filters")
print(f"✗ {total_lines - valid_count} workouts filtered out")

print("\nFilter breakdown:")
for reason, count in sorted(filter_stats.items(), key=lambda x: -x[1]):
    if reason != 'valid':
        print(f"  {reason:25s}: {count:6d} workouts")

# ============================================================================
# SECTION 2: COMPUTE RAW DATA CORRELATIONS
# ============================================================================
print("\n" + "="*80)
print("SECTION 2: RAW DATA CORRELATIONS")
print("="*80)

print(f"\n✓ Analyzed {speed_hr_corr.n:,} total timesteps")

# Compute statistics
print("\nRaw data statistics:")
print(f"  Speed:     mean={speed_hr_corr.mean_x:8.2f}, std={speed_hr_corr.std(True):8.2f}, "
      f"min={speed_hr_corr.min_x:8.2f}, max={speed_hr_corr.max_x:8.2f}")
print(f"  Altitude:  mean={altitude_hr_corr.mean_x:8.2f}, std={altitude_hr_corr.std(True):8.2f}, "
      f"min={altitude_hr_corr.min_x:8.2f}, max={altitude_hr_corr.max_x:8.2f}")
print(f"  Heart Rate:mean={speed_hr_corr.mean_y:8.2f}, std={speed_hr_corr.std(False):8.2f}, "
      f"min={speed_hr_corr.min_y:8.2f}, max={speed_hr_corr.max_y:8.2f}")

# Compute correlations
corr_speed_hr = speed_hr_corr.correlation()
corr_altitude_hr = altitude_hr_corr.correlation()
corr_speed_altitude = speed_altitude_corr.correlation()

print(f"\nRAW DATA CORRELATIONS:")
print(f"  Speed    → Heart Rate: {corr_speed_hr:.6f}")
print(f"  Altitude → Heart Rate: {corr_altitude_hr:.6f}")
print(f"  Speed    → Altitude:   {corr_speed_altitude:.6f}")

# Build correlation matrix for visualization
corr_matrix_raw = np.array([
    [1.0, corr_speed_altitude, corr_speed_hr],
    [corr_speed_altitude, 1.0, corr_altitude_hr],
    [corr_speed_hr, corr_altitude_hr, 1.0]
])

# ============================================================================
# SECTION 3: COMPUTE CORRELATION ON PROCESSED DATA (FOR COMPARISON)
# ============================================================================
print("\n" + "="*80)
print("SECTION 3: LOADING PROCESSED DATA FOR COMPARISON")
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
# SECTION 4: COMPARISON AND VISUALIZATION
# ============================================================================
print("\n" + "="*80)
print("SECTION 4: CORRELATION COMPARISON")
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

# Convert scatter samples to numpy
scatter_speed = np.array(scatter_samples['speed'])
scatter_altitude = np.array(scatter_samples['altitude'])
scatter_hr = np.array(scatter_samples['hr'])

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
sample_indices = np.random.choice(len(scatter_speed), min(10000, len(scatter_speed)), replace=False)
ax.scatter(scatter_speed[sample_indices], scatter_hr[sample_indices], alpha=0.3, s=1)
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Heart Rate (BPM)')
ax.set_title(f'Raw: Speed vs HR (r={corr_matrix_raw[0,2]:.3f})', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

# Scatter: Altitude vs HR - Raw
ax = axes[0, 2]
ax.scatter(scatter_altitude[sample_indices], scatter_hr[sample_indices], alpha=0.3, s=1, color='green')
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

# Free memory
del scatter_speed, scatter_altitude, scatter_hr, scatter_samples
gc.collect()

# ============================================================================
# SECTION 5: TEMPORAL CORRELATION ANALYSIS (LAG ANALYSIS)
# ============================================================================
print("\n" + "="*80)
print("SECTION 5: TEMPORAL CORRELATION ANALYSIS (LAG)")
print("="*80)

print("\nAnalyzing temporal correlations (streaming through data again for lag analysis)...")

# Stream through file again for lag analysis (memory-efficient)
max_lag = 30
lag_corrs_speed = [StreamingCorrelation() for _ in range(max_lag + 1)]
lag_corrs_altitude = [StreamingCorrelation() for _ in range(max_lag + 1)]

# Sample workouts for lag analysis (to keep it fast)
WORKOUT_SAMPLE_RATE = 0.2  # Use 20% of workouts

with open(RAW_DATA_FILE, 'r') as f:
    for line in tqdm(f, total=total_lines, desc="Lag analysis"):
        workout = parse_workout(line.strip())
        
        if workout is None:
            continue
        
        is_valid, _ = filter_workout(workout)
        
        if is_valid:
            # Sample 20% of workouts to speed up
            import random
            if random.random() > WORKOUT_SAMPLE_RATE:
                continue
            
            speed = np.array(workout['speed'], dtype=float)
            altitude = np.array(workout['altitude'], dtype=float)
            hr = np.array(workout['heart_rate'], dtype=float)
            
            # Compute lag correlations
            for lag in range(max_lag + 1):
                if len(speed) > lag:
                    speed_vals = speed[:-lag if lag > 0 else None]
                    altitude_vals = altitude[:-lag if lag > 0 else None]
                    hr_vals = hr[lag:]
                    
                    lag_corrs_speed[lag].update(speed_vals, hr_vals)
                    lag_corrs_altitude[lag].update(altitude_vals, hr_vals)

# Extract correlation values
lag_corrs_speed_vals = [corr.correlation() for corr in lag_corrs_speed]
lag_corrs_altitude_vals = [corr.correlation() for corr in lag_corrs_altitude]

# Find best lag
best_lag_speed = int(np.argmax(lag_corrs_speed_vals))
best_lag_altitude = int(np.argmax(lag_corrs_altitude_vals))

print(f"\nLag analysis results (sampled {WORKOUT_SAMPLE_RATE*100:.0f}% of workouts):")
print(f"  Speed    → HR: Best correlation {lag_corrs_speed_vals[best_lag_speed]:.6f} at lag={best_lag_speed}")
print(f"  Altitude → HR: Best correlation {lag_corrs_altitude_vals[best_lag_altitude]:.6f} at lag={best_lag_altitude}")

# Visualize lag correlations
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(range(max_lag + 1), lag_corrs_speed_vals, marker='o', label='Speed → HR', linewidth=2)
ax.plot(range(max_lag + 1), lag_corrs_altitude_vals, marker='s', label='Altitude → HR', linewidth=2)
ax.axvline(float(best_lag_speed), color='blue', linestyle='--', alpha=0.5, label=f'Best Speed lag={best_lag_speed}')
ax.axvline(float(best_lag_altitude), color='orange', linestyle='--', alpha=0.5, label=f'Best Altitude lag={best_lag_altitude}')
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
# SECTION 6: GENERATE SUMMARY REPORT
# ============================================================================
print("\n" + "="*80)
print("SECTION 6: GENERATING SUMMARY REPORT")
print("="*80)

report = []
report.append("# RAW vs PROCESSED DATA - CORRELATION ANALYSIS\n\n")
report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

report.append("## Summary\n\n")
report.append(f"- **Raw workouts analyzed:** {valid_count:,}\n")
report.append(f"- **Total timesteps (raw):** {speed_hr_corr.n:,}\n")
report.append(f"- **Processed timesteps:** {len(train_speed):,}\n")
report.append(f"- **Analysis method:** Streaming (low RAM usage)\n\n")

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
report.append(f"- **Best Speed lag:** {best_lag_speed} timesteps (correlation: {lag_corrs_speed_vals[best_lag_speed]:.6f})\n")
report.append(f"- **Best Altitude lag:** {best_lag_altitude} timesteps (correlation: {lag_corrs_altitude_vals[best_lag_altitude]:.6f})\n\n")

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
