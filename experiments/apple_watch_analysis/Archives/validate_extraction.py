#!/usr/bin/env python3
"""
Validation Script for Apple Health Data Extraction

This script validates the extracted data by:
1. Loading parsed workouts
2. Extracting HR records for sample workouts
3. Aligning HR with GPS trackpoints
4. Generating validation plots

Author: Apple Watch Analysis Pipeline
Date: 2025-11-25
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from parse_apple_health import AppleHealthParser, GPXParser, Workout


def align_hr_to_gps(hr_records, gps_trackpoints):
    """
    Align heart rate records to GPS trackpoints using interpolation
    
    Args:
        hr_records: List of HeartRateRecord objects
        gps_trackpoints: List of GPXTrackpoint objects
    
    Returns:
        aligned_data: DataFrame with columns [timestamp, speed, elevation, heart_rate]
    """
    if not hr_records or not gps_trackpoints:
        return None
    
    # Convert to pandas for easier manipulation
    hr_df = pd.DataFrame([
        {'timestamp': hr.timestamp, 'heart_rate': hr.value}
        for hr in hr_records
    ])
    
    gps_df = pd.DataFrame([
        {
            'timestamp': tp.timestamp,
            'lat': tp.lat,
            'lon': tp.lon,
            'elevation': tp.elevation,
            'speed': tp.speed,
            'course': tp.course
        }
        for tp in gps_trackpoints
    ])
    
    # Remove timezone info for easier comparison (keep as naive datetime)
    hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp']).dt.tz_localize(None)
    gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp']).dt.tz_localize(None)
    
    # Sort by timestamp
    hr_df = hr_df.sort_values('timestamp').reset_index(drop=True)
    gps_df = gps_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n  HR records: {len(hr_df)} (from {hr_df['timestamp'].min()} to {hr_df['timestamp'].max()})")
    print(f"  GPS points: {len(gps_df)} (from {gps_df['timestamp'].min()} to {gps_df['timestamp'].max()})")
    
    # Check if there's overlap
    hr_start = hr_df['timestamp'].min()
    hr_end = hr_df['timestamp'].max()
    gps_start = gps_df['timestamp'].min()
    gps_end = gps_df['timestamp'].max()
    
    overlap_start = max(hr_start, gps_start)
    overlap_end = min(hr_end, gps_end)
    
    if overlap_start >= overlap_end:
        print("  ✗ Warning: No temporal overlap between HR and GPS data")
        return None
    
    print(f"  Overlap: {overlap_start} to {overlap_end}")
    
    # Filter to overlap period
    hr_df = hr_df[(hr_df['timestamp'] >= overlap_start) & (hr_df['timestamp'] <= overlap_end)]
    gps_df = gps_df[(gps_df['timestamp'] >= overlap_start) & (gps_df['timestamp'] <= overlap_end)]
    
    if len(hr_df) < 2:
        print("  ✗ Warning: Insufficient HR records in overlap period")
        return None
    
    # Convert timestamps to seconds from start for interpolation
    hr_seconds = (hr_df['timestamp'] - overlap_start).dt.total_seconds().values
    hr_values = hr_df['heart_rate'].values
    
    gps_seconds = (gps_df['timestamp'] - overlap_start).dt.total_seconds().values
    
    # Interpolate HR to GPS timestamps (linear interpolation)
    hr_interp = interp1d(hr_seconds, hr_values, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    
    gps_df['heart_rate'] = hr_interp(gps_seconds)
    
    # Remove any NaN or invalid values
    gps_df = gps_df.dropna(subset=['heart_rate'])
    gps_df = gps_df[gps_df['heart_rate'] > 0]
    
    print(f"  ✓ Aligned {len(gps_df)} trackpoints with HR data")
    
    return gps_df


def calculate_features(df):
    """Calculate additional features from GPS data"""
    if df is None or len(df) < 2:
        return df
    
    # Convert speed from m/s to km/h
    df['speed_kmh'] = df['speed'] * 3.6
    
    # Calculate pace (min/km) - avoid division by zero
    df['pace_min_per_km'] = np.where(
        df['speed'] > 0.5,  # Only calculate for speed > 0.5 m/s
        60 / (df['speed'] * 3.6),
        np.nan
    )
    
    # Calculate elevation change (gradient)
    df['elevation_change'] = df['elevation'].diff()
    
    # Calculate grade (%) - elevation change per distance
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    df['distance_diff'] = df['speed'] * df['time_diff']
    df['grade_percent'] = np.where(
        df['distance_diff'] > 0,
        (df['elevation_change'] / df['distance_diff']) * 100,
        0
    )
    
    # Smooth the grade using rolling average
    df['grade_smooth'] = df['grade_percent'].rolling(window=10, center=True).mean()
    
    return df


def plot_workout_validation(df, workout_info, output_path):
    """Create validation plots for a single workout"""
    if df is None or len(df) == 0:
        print("  ✗ No data to plot")
        return
    
    # Calculate time elapsed from start (in minutes)
    df['time_min'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
    # Create figure with subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle(f"Workout Validation: {workout_info['workout_id']}\n{workout_info['date']}", 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Heart Rate
    ax1 = axes[0]
    ax1.plot(df['time_min'], df['heart_rate'], 'r-', linewidth=1.5, label='Heart Rate')
    ax1.set_ylabel('Heart Rate (BPM)', fontsize=11)
    ax1.set_title('Heart Rate Time Series', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add HR statistics
    hr_mean = df['heart_rate'].mean()
    hr_min = df['heart_rate'].min()
    hr_max = df['heart_rate'].max()
    ax1.axhline(hr_mean, color='orange', linestyle='--', alpha=0.5, label=f'Mean: {hr_mean:.0f}')
    ax1.text(0.02, 0.95, f'Mean: {hr_mean:.0f} | Min: {hr_min:.0f} | Max: {hr_max:.0f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Speed
    ax2 = axes[1]
    ax2.plot(df['time_min'], df['speed_kmh'], 'b-', linewidth=1.5, label='Speed')
    ax2.set_ylabel('Speed (km/h)', fontsize=11)
    ax2.set_title('Speed Profile', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add speed statistics
    speed_mean = df['speed_kmh'].mean()
    ax2.text(0.02, 0.95, f'Average: {speed_mean:.2f} km/h | Max: {df["speed_kmh"].max():.2f} km/h',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 3: Elevation
    ax3 = axes[2]
    ax3.fill_between(df['time_min'], df['elevation'], alpha=0.3, color='green')
    ax3.plot(df['time_min'], df['elevation'], 'g-', linewidth=1.5, label='Elevation')
    ax3.set_ylabel('Elevation (m)', fontsize=11)
    ax3.set_title('Elevation Profile', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Add elevation statistics
    elev_gain = df[df['elevation_change'] > 0]['elevation_change'].sum()
    elev_loss = abs(df[df['elevation_change'] < 0]['elevation_change'].sum())
    ax3.text(0.02, 0.95, f'Gain: {elev_gain:.0f}m | Loss: {elev_loss:.0f}m',
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # Plot 4: HR vs Speed correlation
    ax4 = axes[3]
    scatter = ax4.scatter(df['speed_kmh'], df['heart_rate'], 
                          c=df['time_min'], cmap='viridis', 
                          alpha=0.5, s=20)
    ax4.set_xlabel('Speed (km/h)', fontsize=11)
    ax4.set_ylabel('Heart Rate (BPM)', fontsize=11)
    ax4.set_title('Heart Rate vs Speed (colored by time)', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Add correlation coefficient
    correlation = df[['speed_kmh', 'heart_rate']].corr().iloc[0, 1]
    ax4.text(0.02, 0.95, f'Correlation: {correlation:.3f}',
             transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Time (min)', fontsize=10)
    
    # Common x-axis label for time-series plots
    axes[2].set_xlabel('Time (minutes)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot to: {output_path}")


def main():
    """Main validation workflow"""
    print("=" * 70)
    print("APPLE HEALTH DATA VALIDATION - STEP 2: VERIFICATION")
    print("=" * 70)
    
    # Paths
    base_dir = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_health_export")
    export_xml = base_dir / "export.xml"
    workouts_json = Path("experiments/apple_watch_analysis/output/workouts_summary.json")
    plots_dir = Path("experiments/apple_watch_analysis/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parsed workouts
    print("\n1. Loading parsed workouts...")
    with open(workouts_json) as f:
        workouts_data = json.load(f)
    
    print(f"   ✓ Loaded {len(workouts_data)} workouts")
    
    # Initialize parser
    parser = AppleHealthParser(str(export_xml), str(base_dir))
    
    # Select 5 recent workouts with GPX data for validation
    print("\n2. Selecting sample workouts for validation...")
    
    sample_workouts = []
    for w in reversed(workouts_data):  # Start from most recent
        if w['gpx_file_path'] and len(sample_workouts) < 5:
            sample_workouts.append(w)
    
    print(f"   ✓ Selected {len(sample_workouts)} recent workouts with GPX data")
    
    # Process each sample workout
    print("\n3. Processing and validating sample workouts...")
    print("=" * 70)
    
    successful_validations = 0
    
    for i, workout_data in enumerate(sample_workouts, 1):
        print(f"\nWorkout {i}/{len(sample_workouts)}: {workout_data['workout_id']}")
        print(f"  Date: {workout_data['start_date']}")
        print(f"  Duration: {workout_data['duration_min']:.1f} min")
        print(f"  Distance: {workout_data['total_distance_km']:.2f} km" if workout_data['total_distance_km'] else "  Distance: N/A")
        
        try:
            # Parse GPX file
            gpx_path = base_dir / workout_data['gpx_file_path'].lstrip('/')
            
            if not gpx_path.exists():
                print(f"  ✗ GPX file not found: {gpx_path}")
                continue
            
            gpx_parser = GPXParser(str(gpx_path))
            trackpoints = gpx_parser.parse_trackpoints()
            print(f"  ✓ Loaded {len(trackpoints)} GPS trackpoints")
            
            # Parse heart rate records
            start_date = datetime.fromisoformat(workout_data['start_date'].replace('+00:00', ''))
            end_date = datetime.fromisoformat(workout_data['end_date'].replace('+00:00', ''))
            
            print(f"  Extracting HR records...")
            hr_records = parser.parse_heart_rate_records(start_date, end_date)
            print(f"  ✓ Found {len(hr_records)} HR records in time window")
            
            if len(hr_records) == 0:
                print(f"  ✗ No HR data available for this workout")
                continue
            
            # Align HR and GPS data
            print(f"  Aligning HR and GPS data...")
            aligned_df = align_hr_to_gps(hr_records, trackpoints)
            
            if aligned_df is None or len(aligned_df) < 10:
                print(f"  ✗ Failed to align data (insufficient overlap)")
                continue
            
            # Calculate additional features
            aligned_df = calculate_features(aligned_df)
            
            # Generate validation plot
            plot_path = plots_dir / f"validation_{workout_data['workout_id']}.png"
            workout_info = {
                'workout_id': workout_data['workout_id'],
                'date': workout_data['start_date'][:10]
            }
            
            plot_workout_validation(aligned_df, workout_info, plot_path)
            successful_validations += 1
            
            # Save aligned data as CSV
            csv_path = plots_dir.parent / "data_cache" / f"{workout_data['workout_id']}_aligned.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            aligned_df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved aligned data to: {csv_path.name}")
            
        except Exception as e:
            print(f"  ✗ Error processing workout: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Total workouts processed: {len(sample_workouts)}")
    print(f"Successful validations: {successful_validations}")
    print(f"Validation plots saved to: {plots_dir}")
    print("\nNext step: Review the plots to verify data quality")
    print("=" * 70)


if __name__ == "__main__":
    main()
