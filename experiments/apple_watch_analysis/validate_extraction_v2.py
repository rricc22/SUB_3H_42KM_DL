#!/usr/bin/env python3
"""
Validation Script v2 - With Timezone Fix and Speed Calculation

Improvements:
- Automatic timezone detection and alignment
- Speed calculation from GPS positions (haversine formula)
- Better overlap detection

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


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate distance between two GPS points using Haversine formula
    
    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)
    
    Returns:
        Distance in meters
    """
    # Earth radius in meters
    R = 6371000
    
    # Convert to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def calculate_speed_from_gps(df):
    """
    Calculate speed from GPS position differences
    
    Args:
        df: DataFrame with timestamp, lat, lon columns
    
    Returns:
        df with added 'speed_calculated' column in m/s
    """
    if len(df) < 2:
        df['speed_calculated'] = 0
        return df
    
    # Calculate distance between consecutive points
    distances = []
    for i in range(len(df)):
        if i == 0:
            distances.append(0)
        else:
            dist = haversine_distance(
                df.iloc[i-1]['lat'], df.iloc[i-1]['lon'],
                df.iloc[i]['lat'], df.iloc[i]['lon']
            )
            distances.append(dist)
    
    df['distance_diff'] = distances
    
    # Calculate time differences
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    
    # Calculate speed (m/s)
    df['speed_calculated'] = np.where(
        df['time_diff'] > 0,
        df['distance_diff'] / df['time_diff'],
        0
    )
    
    # Apply moving average to smooth speed (5-second window)
    df['speed_smooth'] = df['speed_calculated'].rolling(window=5, center=True, min_periods=1).mean()
    
    return df


def detect_timezone_offset(workout_start_date, gps_start_time):
    """
    Detect timezone offset between workout metadata and GPX timestamps
    
    Args:
        workout_start_date: Workout start date from metadata (datetime)
        gps_start_time: First GPS timestamp from GPX (datetime, UTC)
    
    Returns:
        Offset in hours (int)
    """
    # Remove timezone info for comparison
    workout_naive = workout_start_date.replace(tzinfo=None) if workout_start_date.tzinfo else workout_start_date
    gps_naive = gps_start_time.replace(tzinfo=None) if gps_start_time.tzinfo else gps_start_time
    
    # Calculate difference
    diff = (workout_naive - gps_naive).total_seconds() / 3600
    
    # Round to nearest hour
    offset = round(diff)
    
    return offset


def align_hr_to_gps_v2(hr_records, gps_trackpoints, workout_start_date):
    """
    Align heart rate records to GPS trackpoints with timezone correction
    
    Args:
        hr_records: List of HeartRateRecord objects
        gps_trackpoints: List of GPXTrackpoint objects
        workout_start_date: Workout start date for timezone detection
    
    Returns:
        aligned_data: DataFrame with aligned data
    """
    if not hr_records or not gps_trackpoints:
        return None
    
    # Convert to pandas
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
    
    # Remove timezone info
    hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp']).dt.tz_localize(None)
    gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp']).dt.tz_localize(None)
    
    # Detect timezone offset
    offset_hours = detect_timezone_offset(workout_start_date, gps_df['timestamp'].iloc[0])
    
    print(f"\n  Detected timezone offset: {offset_hours} hours")
    print(f"  Workout start (metadata): {workout_start_date}")
    print(f"  GPS start (original): {gps_df['timestamp'].iloc[0]}")
    
    # Apply offset to GPS timestamps
    gps_df['timestamp'] = gps_df['timestamp'] + pd.Timedelta(hours=offset_hours)
    
    print(f"  GPS start (corrected): {gps_df['timestamp'].iloc[0]}")
    
    # Sort by timestamp
    hr_df = hr_df.sort_values('timestamp').reset_index(drop=True)
    gps_df = gps_df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"\n  HR records: {len(hr_df)} (from {hr_df['timestamp'].min()} to {hr_df['timestamp'].max()})")
    print(f"  GPS points: {len(gps_df)} (from {gps_df['timestamp'].min()} to {gps_df['timestamp'].max()})")
    
    # Check overlap
    hr_start = hr_df['timestamp'].min()
    hr_end = hr_df['timestamp'].max()
    gps_start = gps_df['timestamp'].min()
    gps_end = gps_df['timestamp'].max()
    
    overlap_start = max(hr_start, gps_start)
    overlap_end = min(hr_end, gps_end)
    
    if overlap_start >= overlap_end:
        print("  ✗ Warning: No temporal overlap between HR and GPS data")
        return None
    
    overlap_duration = (overlap_end - overlap_start).total_seconds() / 60
    print(f"  ✓ Overlap: {overlap_start} to {overlap_end} ({overlap_duration:.1f} min)")
    
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
    
    # Interpolate HR to GPS timestamps
    hr_interp = interp1d(hr_seconds, hr_values, kind='linear', 
                         bounds_error=False, fill_value='extrapolate')
    
    gps_df['heart_rate'] = hr_interp(gps_seconds)
    
    # Remove invalid values
    gps_df = gps_df.dropna(subset=['heart_rate'])
    gps_df = gps_df[gps_df['heart_rate'] > 0]
    
    print(f"  ✓ Aligned {len(gps_df)} trackpoints with HR data")
    
    return gps_df


def calculate_features(df):
    """Calculate additional features from GPS data"""
    if df is None or len(df) < 2:
        return df
    
    # Calculate speed from GPS positions
    df = calculate_speed_from_gps(df)
    
    # Convert to km/h
    df['speed_kmh'] = df['speed_smooth'] * 3.6
    
    # Calculate pace (min/km)
    df['pace_min_per_km'] = np.where(
        df['speed_smooth'] > 0.5,  # Only for speed > 0.5 m/s (1.8 km/h)
        60 / (df['speed_smooth'] * 3.6),
        np.nan
    )
    
    # Calculate elevation change
    df['elevation_change'] = df['elevation'].diff()
    
    # Calculate grade (%)
    df['grade_percent'] = np.where(
        df['distance_diff'] > 0,
        (df['elevation_change'] / df['distance_diff']) * 100,
        0
    )
    
    # Smooth grade
    df['grade_smooth'] = df['grade_percent'].rolling(window=10, center=True, min_periods=1).mean()
    
    # Calculate time from start in minutes
    df['time_min'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    
    return df


def plot_workout_validation(df, workout_info, output_path):
    """Create validation plots for a single workout"""
    if df is None or len(df) == 0:
        print("  ✗ No data to plot")
        return
    
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
    
    hr_mean = df['heart_rate'].mean()
    hr_min = df['heart_rate'].min()
    hr_max = df['heart_rate'].max()
    ax1.axhline(hr_mean, color='orange', linestyle='--', alpha=0.5)
    ax1.text(0.02, 0.95, f'Mean: {hr_mean:.0f} | Min: {hr_min:.0f} | Max: {hr_max:.0f}',
             transform=ax1.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 2: Speed (calculated from GPS)
    ax2 = axes[1]
    ax2.plot(df['time_min'], df['speed_kmh'], 'b-', linewidth=1.5, label='Speed (calculated)')
    ax2.set_ylabel('Speed (km/h)', fontsize=11)
    ax2.set_title('Speed Profile (calculated from GPS positions)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    speed_mean = df['speed_kmh'].mean()
    pace_mean = df['pace_min_per_km'].median()
    ax2.text(0.02, 0.95, f'Avg Speed: {speed_mean:.2f} km/h | Avg Pace: {pace_mean:.1f} min/km',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 3: Elevation
    ax3 = axes[2]
    ax3.fill_between(df['time_min'], df['elevation'], alpha=0.3, color='green')
    ax3.plot(df['time_min'], df['elevation'], 'g-', linewidth=1.5, label='Elevation')
    ax3.set_ylabel('Elevation (m)', fontsize=11)
    ax3.set_xlabel('Time (minutes)', fontsize=11)
    ax3.set_title('Elevation Profile', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
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
    
    # Filter out zero speeds for correlation
    valid_data = df[df['speed_kmh'] > 1.0][['speed_kmh', 'heart_rate']].dropna()
    if len(valid_data) > 10:
        correlation = valid_data.corr().iloc[0, 1]
        ax4.text(0.02, 0.95, f'Correlation: {correlation:.3f}',
                 transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Time (min)', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved plot to: {output_path}")
    
    return {
        'hr_mean': hr_mean,
        'hr_min': hr_min,
        'hr_max': hr_max,
        'speed_mean': speed_mean,
        'duration_min': df['time_min'].max()
    }


def main():
    """Main validation workflow"""
    print("=" * 70)
    print("APPLE HEALTH DATA VALIDATION v2 - WITH TIMEZONE FIX")
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
    
    # Select 10 workouts: 2 from each year, plus 2 most recent
    print("\n2. Selecting sample workouts for validation...")
    
    sample_workouts = []
    
    # Get 2 most recent
    for w in reversed(workouts_data):
        if w['gpx_file_path'] and len(sample_workouts) < 2:
            sample_workouts.append(w)
    
    # Get 2 from each year (2025, 2024, 2021, 2020, 2019)
    for year in [2025, 2024, 2021, 2020, 2019]:
        year_workouts = [w for w in workouts_data 
                         if w['gpx_file_path'] and 
                         datetime.fromisoformat(w['start_date'].replace('+00:00', '')).year == year]
        if year_workouts and len(sample_workouts) < 10:
            # Get first and middle workout from year
            sample_workouts.append(year_workouts[0])
            if len(year_workouts) > 1 and len(sample_workouts) < 10:
                sample_workouts.append(year_workouts[len(year_workouts)//2])
    
    # Remove duplicates
    sample_workouts = list({w['workout_id']: w for w in sample_workouts}.values())[:10]
    
    print(f"   ✓ Selected {len(sample_workouts)} workouts for validation")
    
    # Process each sample workout
    print("\n3. Processing and validating sample workouts...")
    print("=" * 70)
    
    results = []
    successful = 0
    
    for i, workout_data in enumerate(sample_workouts, 1):
        print(f"\nWorkout {i}/{len(sample_workouts)}: {workout_data['workout_id']}")
        print(f"  Date: {workout_data['start_date']}")
        print(f"  Duration: {workout_data['duration_min']:.1f} min")
        
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
            
            hr_records = parser.parse_heart_rate_records(start_date, end_date)
            print(f"  ✓ Found {len(hr_records)} HR records in time window")
            
            if len(hr_records) == 0:
                print(f"  ✗ No HR data available")
                continue
            
            # Align with timezone correction
            aligned_df = align_hr_to_gps_v2(hr_records, trackpoints, start_date)
            
            if aligned_df is None or len(aligned_df) < 60:  # At least 1 minute
                print(f"  ✗ Failed to align data (insufficient overlap)")
                continue
            
            # Calculate features
            aligned_df = calculate_features(aligned_df)
            
            # Generate plot
            plot_path = plots_dir / f"validation_v2_{workout_data['workout_id']}.png"
            workout_info = {
                'workout_id': workout_data['workout_id'],
                'date': workout_data['start_date'][:10]
            }
            
            stats = plot_workout_validation(aligned_df, workout_info, plot_path)
            
            # Save aligned data
            csv_path = plots_dir.parent / "data_cache" / f"{workout_data['workout_id']}_aligned_v2.csv"
            aligned_df.to_csv(csv_path, index=False)
            print(f"  ✓ Saved aligned data to: {csv_path.name}")
            
            successful += 1
            results.append({
                'workout_id': workout_data['workout_id'],
                'date': workout_data['start_date'][:10],
                'success': True,
                'aligned_points': len(aligned_df),
                **stats
            })
            
        except Exception as e:
            print(f"  ✗ Error processing workout: {e}")
            results.append({
                'workout_id': workout_data['workout_id'],
                'date': workout_data['start_date'][:10],
                'success': False,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY v2")
    print("=" * 70)
    print(f"Total workouts processed: {len(sample_workouts)}")
    print(f"Successful validations: {successful}/{len(sample_workouts)} ({100*successful/len(sample_workouts):.0f}%)")
    print(f"\nValidation plots saved to: {plots_dir}")
    
    # Save results
    results_path = plots_dir.parent / "output" / "validation_results_v2.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
