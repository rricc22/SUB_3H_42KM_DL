#!/usr/bin/env python3
"""
Process All Workouts - Full Dataset Generation

Processes all 285 running workouts with:
- Timezone correction
- Speed calculation from GPS
- Heart rate alignment
- Feature engineering

Output: Individual CSV files for each workout with aligned time-series data

Author: Apple Watch Analysis Pipeline
Date: 2025-11-25
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
from tqdm import tqdm
from scipy.interpolate import interp1d

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from parse_apple_health import AppleHealthParser, GPXParser


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two GPS points using Haversine formula"""
    R = 6371000  # Earth radius in meters
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def calculate_speed_from_gps(df):
    """Calculate speed from GPS position differences"""
    if len(df) < 2:
        df['speed_calculated'] = 0
        return df
    
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
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds()
    df['speed_calculated'] = np.where(df['time_diff'] > 0, df['distance_diff'] / df['time_diff'], 0)
    df['speed_smooth'] = df['speed_calculated'].rolling(window=5, center=True, min_periods=1).mean()
    return df


def detect_timezone_offset(workout_start_date, gps_start_time):
    """Detect timezone offset between workout metadata and GPX timestamps"""
    workout_naive = workout_start_date.replace(tzinfo=None) if workout_start_date.tzinfo else workout_start_date
    gps_naive = gps_start_time.replace(tzinfo=None) if gps_start_time.tzinfo else gps_start_time
    diff = (workout_naive - gps_naive).total_seconds() / 3600
    return round(diff)


def align_hr_to_gps_v2(hr_records, gps_trackpoints, workout_start_date):
    """Align heart rate records to GPS trackpoints with timezone correction"""
    if not hr_records or not gps_trackpoints:
        return None
    
    hr_df = pd.DataFrame([{'timestamp': hr.timestamp, 'heart_rate': hr.value} for hr in hr_records])
    gps_df = pd.DataFrame([
        {'timestamp': tp.timestamp, 'lat': tp.lat, 'lon': tp.lon,
         'elevation': tp.elevation, 'speed': tp.speed, 'course': tp.course}
        for tp in gps_trackpoints
    ])
    
    hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp']).dt.tz_localize(None)
    gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp']).dt.tz_localize(None)
    
    offset_hours = detect_timezone_offset(workout_start_date, gps_df['timestamp'].iloc[0])
    gps_df['timestamp'] = gps_df['timestamp'] + pd.Timedelta(hours=offset_hours)
    
    hr_df = hr_df.sort_values('timestamp').reset_index(drop=True)
    gps_df = gps_df.sort_values('timestamp').reset_index(drop=True)
    
    hr_start, hr_end = hr_df['timestamp'].min(), hr_df['timestamp'].max()
    gps_start, gps_end = gps_df['timestamp'].min(), gps_df['timestamp'].max()
    overlap_start = max(hr_start, gps_start)
    overlap_end = min(hr_end, gps_end)
    
    if overlap_start >= overlap_end:
        return None
    
    hr_df = hr_df[(hr_df['timestamp'] >= overlap_start) & (hr_df['timestamp'] <= overlap_end)]
    gps_df = gps_df[(gps_df['timestamp'] >= overlap_start) & (gps_df['timestamp'] <= overlap_end)]
    
    if len(hr_df) < 2:
        return None
    
    hr_seconds = (hr_df['timestamp'] - overlap_start).dt.total_seconds().values
    hr_values = hr_df['heart_rate'].values
    gps_seconds = (gps_df['timestamp'] - overlap_start).dt.total_seconds().values
    
    hr_interp = interp1d(hr_seconds, hr_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    gps_df['heart_rate'] = hr_interp(gps_seconds)
    gps_df = gps_df.dropna(subset=['heart_rate'])
    gps_df = gps_df[gps_df['heart_rate'] > 0]
    
    return gps_df


def calculate_features(df):
    """Calculate additional features from GPS data"""
    if df is None or len(df) < 2:
        return df
    
    df = calculate_speed_from_gps(df)
    df['speed_kmh'] = df['speed_smooth'] * 3.6
    df['pace_min_per_km'] = np.where(df['speed_smooth'] > 0.5, 60 / (df['speed_smooth'] * 3.6), np.nan)
    df['elevation_change'] = df['elevation'].diff()
    df['grade_percent'] = np.where(df['distance_diff'] > 0, (df['elevation_change'] / df['distance_diff']) * 100, 0)
    df['grade_smooth'] = df['grade_percent'].rolling(window=10, center=True, min_periods=1).mean()
    df['time_min'] = (df['timestamp'] - df['timestamp'].min()).dt.total_seconds() / 60
    return df


def calculate_hr_quality(hr_records, duration_min):
    """
    Calculate heart rate data quality metrics
    
    Args:
        hr_records: List of heart rate records
        duration_min: Workout duration in minutes
    
    Returns:
        dict with quality metrics
    """
    if not hr_records or duration_min == 0:
        return {
            'hr_sample_count': 0,
            'hr_samples_per_min': 0,
            'quality': 'NONE'
        }
    
    samples_per_min = len(hr_records) / duration_min
    
    if samples_per_min >= 5:
        quality = 'GOOD'
    elif samples_per_min >= 1:
        quality = 'MEDIUM'
    else:
        quality = 'SPARSE'
    
    return {
        'hr_sample_count': len(hr_records),
        'hr_samples_per_min': samples_per_min,
        'quality': quality
    }


def process_single_workout(workout_data, parser, base_dir, output_dir):
    """
    Process a single workout and save aligned data
    
    Args:
        workout_data: Workout metadata dict
        parser: AppleHealthParser instance
        base_dir: Base directory for Apple Health export
        output_dir: Output directory for processed files
    
    Returns:
        dict with processing results
    """
    workout_id = workout_data['workout_id']
    
    try:
        # Parse GPX file
        gpx_path = base_dir / workout_data['gpx_file_path'].lstrip('/')
        
        if not gpx_path.exists():
            return {
                'workout_id': workout_id,
                'success': False,
                'error': 'GPX file not found'
            }
        
        gpx_parser = GPXParser(str(gpx_path))
        trackpoints = gpx_parser.parse_trackpoints()
        
        if len(trackpoints) < 60:  # Less than 1 minute of data
            return {
                'workout_id': workout_id,
                'success': False,
                'error': 'Insufficient GPS trackpoints'
            }
        
        # Parse heart rate records
        start_date = datetime.fromisoformat(workout_data['start_date'].replace('+00:00', ''))
        end_date = datetime.fromisoformat(workout_data['end_date'].replace('+00:00', ''))
        
        hr_records = parser.parse_heart_rate_records(start_date, end_date)
        
        if len(hr_records) == 0:
            return {
                'workout_id': workout_id,
                'success': False,
                'error': 'No heart rate data'
            }
        
        # Calculate HR quality
        hr_quality = calculate_hr_quality(hr_records, workout_data['duration_min'])
        
        # Align HR to GPS with timezone correction
        aligned_df = align_hr_to_gps_v2(hr_records, trackpoints, start_date)
        
        if aligned_df is None or len(aligned_df) < 60:
            return {
                'workout_id': workout_id,
                'success': False,
                'error': 'Failed to align data',
                **hr_quality
            }
        
        # Calculate features
        aligned_df = calculate_features(aligned_df)
        
        # Add metadata columns
        aligned_df['workout_id'] = workout_id
        aligned_df['date'] = start_date.date()
        
        # Save to CSV
        csv_path = output_dir / f"{workout_id}_processed.csv"
        
        # Select relevant columns
        output_columns = [
            'workout_id',
            'date',
            'timestamp',
            'time_min',
            'lat',
            'lon',
            'elevation',
            'speed_kmh',
            'pace_min_per_km',
            'grade_smooth',
            'heart_rate'
        ]
        
        aligned_df[output_columns].to_csv(csv_path, index=False)
        
        # Calculate statistics
        duration = aligned_df['time_min'].max()
        distance_km = (aligned_df['distance_diff'].sum()) / 1000
        avg_speed = aligned_df['speed_kmh'].mean()
        avg_hr = aligned_df['heart_rate'].mean()
        
        return {
            'workout_id': workout_id,
            'success': True,
            'date': start_date.strftime('%Y-%m-%d'),
            'aligned_points': len(aligned_df),
            'duration_min': duration,
            'distance_km': distance_km,
            'avg_speed_kmh': avg_speed,
            'avg_heart_rate': avg_hr,
            **hr_quality,
            'output_file': csv_path.name
        }
        
    except Exception as e:
        return {
            'workout_id': workout_id,
            'success': False,
            'error': str(e)
        }


def main():
    """Main processing workflow"""
    print("=" * 80)
    print("PROCESSING ALL APPLE WATCH WORKOUTS")
    print("=" * 80)
    
    # Paths
    base_dir = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_health_export")
    export_xml = base_dir / "export.xml"
    workouts_json = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_watch_analysis/output/workouts_summary.json")
    output_dir = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_watch_analysis/processed_workouts")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load parsed workouts
    print("\n1. Loading workout metadata...")
    with open(workouts_json) as f:
        workouts_data = json.load(f)
    
    print(f"   ✓ Loaded {len(workouts_data)} workouts")
    
    # Filter workouts with GPX files
    workouts_with_gpx = [w for w in workouts_data if w['gpx_file_path']]
    print(f"   ✓ {len(workouts_with_gpx)} workouts have GPX files")
    
    # Initialize parser
    print("\n2. Initializing Apple Health parser...")
    parser = AppleHealthParser(str(export_xml), str(base_dir))
    print("   ✓ Parser ready")
    
    # Process all workouts
    print(f"\n3. Processing {len(workouts_with_gpx)} workouts...")
    print("   (This may take 30-60 minutes)")
    print("=" * 80)
    
    results = []
    successful = 0
    failed = 0
    
    # Process with progress bar
    for workout_data in tqdm(workouts_with_gpx, desc="Processing workouts", unit="workout"):
        result = process_single_workout(workout_data, parser, base_dir, output_dir)
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    # Calculate statistics
    print("\n" + "=" * 80)
    print("PROCESSING SUMMARY")
    print("=" * 80)
    
    successful_results = [r for r in results if r['success']]
    
    print(f"\nTotal workouts: {len(workouts_with_gpx)}")
    print(f"Successful: {successful} ({100*successful/len(workouts_with_gpx):.1f}%)")
    print(f"Failed: {failed} ({100*failed/len(workouts_with_gpx):.1f}%)")
    
    if successful_results:
        # Calculate aggregate statistics
        total_distance = sum(r['distance_km'] for r in successful_results)
        total_duration = sum(r['duration_min'] for r in successful_results)
        total_points = sum(r['aligned_points'] for r in successful_results)
        
        print(f"\n--- Aggregate Statistics ---")
        print(f"Total distance: {total_distance:.1f} km")
        print(f"Total duration: {total_duration/60:.1f} hours")
        print(f"Total aligned data points: {total_points:,}")
        print(f"Average points per workout: {total_points/successful:.0f}")
        
        # HR quality breakdown
        quality_counts = {}
        for r in successful_results:
            q = r.get('quality', 'UNKNOWN')
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        print(f"\n--- Heart Rate Quality ---")
        for quality in ['GOOD', 'MEDIUM', 'SPARSE']:
            count = quality_counts.get(quality, 0)
            pct = 100 * count / successful if successful > 0 else 0
            print(f"{quality:8s}: {count:3d} workouts ({pct:.1f}%)")
        
        # Temporal distribution
        years = {}
        for r in successful_results:
            year = r['date'][:4]
            years[year] = years.get(year, 0) + 1
        
        print(f"\n--- Temporal Distribution ---")
        for year in sorted(years.keys()):
            print(f"{year}: {years[year]:3d} workouts")
    
    # Save results
    results_path = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_watch_analysis/output/processing_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n--- Output ---")
    print(f"Processed CSV files: {output_dir}")
    print(f"Processing results: {results_path}")
    print("=" * 80)
    
    # Print failure summary if any
    if failed > 0:
        print(f"\n--- Failure Summary ---")
        failure_reasons = {}
        for r in results:
            if not r['success']:
                error = r.get('error', 'Unknown')
                failure_reasons[error] = failure_reasons.get(error, 0) + 1
        
        for reason, count in sorted(failure_reasons.items(), key=lambda x: -x[1]):
            print(f"{reason}: {count} workouts")
        print("=" * 80)


if __name__ == "__main__":
    main()
