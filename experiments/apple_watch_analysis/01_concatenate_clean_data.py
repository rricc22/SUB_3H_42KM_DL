#!/usr/bin/env python3
"""
Step 1: Concatenate & Clean Apple Watch Data (HYBRID VERSION)

Processes data from multiple users:
- User1: Loads existing processed CSV files (FAST)
- User2: Processes from scratch (XML/GPX parsing - SLOW)

Output: DATA/apple_watch_clean/workouts_all_users.csv

Author: Apple Watch Processing Pipeline
Date: 2025-11-25
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
import ast
from scipy.interpolate import interp1d

# Add parser to path
sys.path.append(str(Path(__file__).parent))
from parse_apple_health import AppleHealthParser, GPXParser

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROCESSED_DIR = Path(__file__).parent / 'processed_workouts'
DATA_DIR = PROJECT_ROOT / 'DATA' / 'CUSTOM_DATA'
OUTPUT_DIR = PROJECT_ROOT / 'DATA' / 'apple_watch_clean'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# User configuration
USERS = {
    'User1': {
        'base_dir': None,  # Uses processed CSVs
        'userId': 0,
        'gender': 1,  # 0 = female, 1 = male
        'use_processed_csv': True
    },
    'User2': {
        'base_dir': DATA_DIR / 'apple_health_export_User2',
        'userId': 1,
        'gender': 1,  # Adjust if known
        'use_processed_csv': False
    }
}

MIN_SEQUENCE_LENGTH = 60  # Minimum 60 points (~1 minute)
HR_QUALITY_THRESHOLD = 5.0  # Minimum HR samples per minute for "GOOD" quality

# ═══════════════════════════════════════════════════════════════════════════
# USER1: LOAD FROM PROCESSED CSV (FAST)
# ═══════════════════════════════════════════════════════════════════════════

def load_processed_workout(csv_path, user_name, user_id, gender):
    """
    Load a single processed workout CSV and extract sequences
    
    Returns:
        dict with workout data or None if too short
    """
    try:
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Filter out invalid rows
        df = df.dropna(subset=['speed_kmh', 'elevation', 'heart_rate'])
        df = df[df['heart_rate'] > 0]
        df = df[np.isfinite(df['heart_rate'])]
        df = df[np.isfinite(df['speed_kmh'])]
        df = df[np.isfinite(df['elevation'])]
        
        if len(df) < MIN_SEQUENCE_LENGTH:
            return None
        
        # Extract sequences as lists
        speed_sequence = df['speed_kmh'].values.tolist()
        altitude_sequence = df['elevation'].values.tolist()
        hr_sequence = df['heart_rate'].values.tolist()
        
        # Extract timestamps
        if 'time_min' in df.columns:
            timestamps_sequence = (df['time_min'].values * 60).tolist()
        else:
            timestamps_sequence = list(range(len(df)))
        
        # Extract metadata
        workout_id = df.iloc[0]['workout_id']
        date = df.iloc[0]['date']
        
        # Calculate statistics
        duration_min = len(df) / 60.0
        avg_hr = df['heart_rate'].mean()
        distance_km = (df['speed_kmh'] * (1.0 / 3600)).sum()
        elevation_diff = df['elevation'].diff()
        elevation_gain_m = elevation_diff[elevation_diff > 0].sum()
        
        # Create workout dictionary
        workout_dict = {
            'workout_id': workout_id,
            'user': user_name,
            'userId': user_id,
            'gender': gender,
            'date': date,
            'duration_min': duration_min,
            'distance_km': distance_km,
            'elevation_m': elevation_gain_m,
            'hr_avg': avg_hr,
            'hr_min': df['heart_rate'].min(),
            'hr_max': df['heart_rate'].max(),
            'speed_kmh': str(speed_sequence),
            'altitude': str(altitude_sequence),
            'heart_rate': str(hr_sequence),
            'timestamps': str(timestamps_sequence),
            'seq_length': len(speed_sequence),
        }
        
        return workout_dict
        
    except Exception as e:
        print(f"  ✗ Error loading {csv_path.name}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# USER2: PROCESS FROM SCRATCH (SLOW)
# ═══════════════════════════════════════════════════════════════════════════

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


def align_hr_to_gps(hr_records, gps_trackpoints, workout_start_date):
    """Align heart rate records to GPS trackpoints with timezone correction"""
    if not hr_records or not gps_trackpoints:
        return None
    
    # Convert to pandas DataFrames
    hr_df = pd.DataFrame([{'timestamp': hr.timestamp, 'heart_rate': hr.value} for hr in hr_records])
    gps_df = pd.DataFrame([
        {'timestamp': tp.timestamp, 'lat': tp.lat, 'lon': tp.lon,
         'elevation': tp.elevation, 'speed': tp.speed}
        for tp in gps_trackpoints
    ])
    
    # Remove timezone info
    hr_df['timestamp'] = pd.to_datetime(hr_df['timestamp']).dt.tz_localize(None)
    gps_df['timestamp'] = pd.to_datetime(gps_df['timestamp']).dt.tz_localize(None)
    
    # Detect and apply timezone offset
    offset_hours = detect_timezone_offset(workout_start_date, gps_df['timestamp'].iloc[0])
    gps_df['timestamp'] = gps_df['timestamp'] + pd.Timedelta(hours=offset_hours)
    
    # Sort by timestamp
    hr_df = hr_df.sort_values('timestamp').reset_index(drop=True)
    gps_df = gps_df.sort_values('timestamp').reset_index(drop=True)
    
    # Find overlap period
    hr_start, hr_end = hr_df['timestamp'].min(), hr_df['timestamp'].max()
    gps_start, gps_end = gps_df['timestamp'].min(), gps_df['timestamp'].max()
    overlap_start = max(hr_start, gps_start)
    overlap_end = min(hr_end, gps_end)
    
    if overlap_start >= overlap_end:
        return None
    
    # Filter to overlap period
    hr_df = hr_df[(hr_df['timestamp'] >= overlap_start) & (hr_df['timestamp'] <= overlap_end)]
    gps_df = gps_df[(gps_df['timestamp'] >= overlap_start) & (gps_df['timestamp'] <= overlap_end)]
    
    if len(hr_df) < 2:
        return None
    
    # Interpolate HR to GPS timestamps
    hr_seconds = (hr_df['timestamp'] - overlap_start).dt.total_seconds().values
    hr_values = hr_df['heart_rate'].values
    gps_seconds = (gps_df['timestamp'] - overlap_start).dt.total_seconds().values
    
    hr_interp = interp1d(hr_seconds, hr_values, kind='linear', bounds_error=False, fill_value='extrapolate')
    gps_df['heart_rate'] = hr_interp(gps_seconds)
    
    # Remove invalid values
    gps_df = gps_df.dropna(subset=['heart_rate'])
    gps_df = gps_df[gps_df['heart_rate'] > 0]
    
    return gps_df


def calculate_hr_quality(hr_records, duration_min):
    """Calculate HR quality metrics"""
    if not hr_records or duration_min == 0:
        return {'quality': 'NONE', 'samples_per_min': 0}
    
    samples_per_min = len(hr_records) / duration_min
    
    if samples_per_min >= HR_QUALITY_THRESHOLD:
        quality = 'GOOD'
    elif samples_per_min >= 1:
        quality = 'MEDIUM'
    else:
        quality = 'SPARSE'
    
    return {'quality': quality, 'samples_per_min': samples_per_min}


def process_workout_from_scratch(workout_data, parser, base_dir, user_name, user_id, gender):
    """
    Process a single workout from XML/GPX files
    
    Returns:
        dict with workout data or None if quality filters not met
    """
    try:
        # Check if has GPX file
        if not workout_data.get('gpx_file_path'):
            return None
        
        gpx_path = base_dir / workout_data['gpx_file_path'].lstrip('/')
        if not gpx_path.exists():
            return None
        
        # Parse GPX trackpoints
        gpx_parser = GPXParser(str(gpx_path))
        trackpoints = gpx_parser.parse_trackpoints()
        
        if len(trackpoints) < MIN_SEQUENCE_LENGTH:
            return None
        
        # Parse heart rate records
        start_date = datetime.fromisoformat(workout_data['start_date'].replace('+00:00', ''))
        end_date = datetime.fromisoformat(workout_data['end_date'].replace('+00:00', ''))
        duration_min = workout_data['duration_min']
        
        hr_records = parser.parse_heart_rate_records(start_date, end_date)
        
        # Calculate HR quality
        hr_quality = calculate_hr_quality(hr_records, duration_min)
        
        # Filter: Only GOOD quality HR data
        if hr_quality['quality'] != 'GOOD':
            return None
        
        # Align HR to GPS with timezone correction
        aligned_df = align_hr_to_gps(hr_records, trackpoints, start_date)
        
        if aligned_df is None or len(aligned_df) < MIN_SEQUENCE_LENGTH:
            return None
        
        # Calculate speed from GPS positions
        aligned_df = calculate_speed_from_gps(aligned_df)
        
        # Convert speed to km/h
        aligned_df['speed_kmh'] = aligned_df['speed_smooth'] * 3.6
        
        # Extract sequences as lists
        speed_sequence = aligned_df['speed_kmh'].values.tolist()
        altitude_sequence = aligned_df['elevation'].values.tolist()
        hr_sequence = aligned_df['heart_rate'].values.tolist()
        timestamps_sequence = [(t - aligned_df['timestamp'].min()).total_seconds() 
                               for t in aligned_df['timestamp']]
        
        # Calculate summary statistics
        total_distance_km = aligned_df['distance_diff'].sum() / 1000
        avg_hr = aligned_df['heart_rate'].mean()
        
        # Create workout dictionary
        workout_dict = {
            'workout_id': workout_data['workout_id'],
            'user': user_name,
            'userId': user_id,
            'gender': gender,
            'date': start_date.strftime('%Y-%m-%d %H:%M:%S'),
            'duration_min': duration_min,
            'distance_km': total_distance_km,
            'elevation_m': workout_data.get('elevation_ascended_m', 0),
            'hr_avg': avg_hr,
            'hr_min': aligned_df['heart_rate'].min(),
            'hr_max': aligned_df['heart_rate'].max(),
            'speed_kmh': str(speed_sequence),
            'altitude': str(altitude_sequence),
            'heart_rate': str(hr_sequence),
            'timestamps': str(timestamps_sequence),
            'seq_length': len(speed_sequence),
        }
        
        return workout_dict
        
    except Exception as e:
        print(f"  ✗ Error processing {workout_data.get('workout_id', 'unknown')}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def process_user1_from_csv(user_config):
    """Process User1 from existing CSV files (FAST)"""
    user_name = 'User1'
    user_id = user_config['userId']
    gender = user_config['gender']
    
    print(f"\n{'='*80}")
    print(f"Processing {user_name} (FROM EXISTING CSV - FAST)")
    print(f"{'='*80}")
    
    csv_files = sorted(PROCESSED_DIR.glob('workout_*_processed.csv'))
    print(f"✓ Found {len(csv_files)} processed CSV files")
    
    workouts = []
    for csv_path in tqdm(csv_files, desc=f"  {user_name}"):
        workout_dict = load_processed_workout(csv_path, user_name, user_id, gender)
        if workout_dict is not None:
            workouts.append(workout_dict)
    
    print(f"✓ Loaded {len(workouts)} valid workouts (filtered from {len(csv_files)})")
    return workouts


def process_user2_from_scratch(user_config):
    """Process User2 from XML/GPX files (SLOW) - saves CSVs for future fast loading"""
    user_name = 'User2'
    user_id = user_config['userId']
    gender = user_config['gender']
    base_dir = user_config['base_dir']
    
    # Create User2 processed workouts directory
    user2_processed_dir = Path(__file__).parent / 'processed_workouts_user2'
    user2_processed_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Processing {user_name} (FROM XML/GPX - SLOW)")
    print(f"{'='*80}")
    
    # Check if User2 CSVs already exist
    existing_csvs = list(user2_processed_dir.glob('workout_*_processed.csv'))
    if len(existing_csvs) > 0:
        print(f"✓ Found {len(existing_csvs)} existing User2 CSV files - LOADING FROM DISK (FAST)")
        workouts = []
        for csv_path in tqdm(existing_csvs, desc=f"  {user_name}"):
            workout_dict = load_processed_workout(csv_path, user_name, user_id, gender)
            if workout_dict is not None:
                workouts.append(workout_dict)
        print(f"✓ Loaded {len(workouts)} valid workouts from existing CSVs")
        return workouts
    
    # If no CSVs exist, process from scratch
    print(f"No existing CSV files found - processing from scratch (will save for future runs)")
    
    if not base_dir.exists():
        print(f"⚠️  Warning: {user_name} data not found at {base_dir}")
        return []
    
    # Initialize parser
    export_xml = base_dir / "export.xml"
    parser = AppleHealthParser(str(export_xml), str(base_dir))
    
    # Parse all workouts
    print(f"\n1. Parsing workouts from {export_xml.name}...")
    workouts = parser.parse_workouts()
    print(f"   ✓ Found {len(workouts)} running workouts")
    
    # Filter workouts with GPX
    workouts_with_gpx = [w.to_dict() for w in workouts if w.gpx_file_path]
    print(f"   ✓ {len(workouts_with_gpx)} have GPX files")
    
    # Process each workout
    print(f"\n2. Processing workouts (filtering for GOOD quality)...")
    print(f"   Saving processed CSVs to: {user2_processed_dir}")
    
    valid_workouts = []
    saved_count = 0
    
    for workout_data in tqdm(workouts_with_gpx, desc=f"  {user_name}"):
        result = process_workout_from_scratch(workout_data, parser, base_dir, user_name, user_id, gender)
        if result is not None:
            valid_workouts.append(result)
            
            # Save this workout as CSV for future fast loading
            workout_id = result['workout_id']
            csv_filename = f"{workout_id}_processed.csv"
            csv_path = user2_processed_dir / csv_filename
            
            # Create DataFrame for this single workout
            # Convert sequences back to lists
            speed_list = ast.literal_eval(result['speed_kmh'])
            altitude_list = ast.literal_eval(result['altitude'])
            hr_list = ast.literal_eval(result['heart_rate'])
            
            # Create per-point DataFrame
            workout_df = pd.DataFrame({
                'workout_id': [workout_id] * len(speed_list),
                'date': [result['date']] * len(speed_list),
                'speed_kmh': speed_list,
                'elevation': altitude_list,
                'heart_rate': hr_list
            })
            
            # Save to CSV
            workout_df.to_csv(csv_path, index=False)
            saved_count += 1
    
    print(f"\n✓ Processed {len(valid_workouts)} GOOD quality workouts (filtered from {len(workouts_with_gpx)})")
    print(f"✓ Saved {saved_count} CSV files to {user2_processed_dir}")
    
    return valid_workouts


def main():
    print("="*80)
    print("STEP 1: CONCATENATE & CLEAN APPLE WATCH DATA (HYBRID VERSION)")
    print("="*80)
    print(f"\nFilters applied:")
    print(f"  • HR Quality: GOOD (≥{HR_QUALITY_THRESHOLD} samples/min)")
    print(f"  • GPS: Must have GPX file")
    print(f"  • Minimum sequence length: {MIN_SEQUENCE_LENGTH} points")
    
    # Process all users
    all_workouts = []
    
    # User1: FAST (from CSV)
    if 'User1' in USERS and USERS['User1']['use_processed_csv']:
        user1_workouts = process_user1_from_csv(USERS['User1'])
        all_workouts.extend(user1_workouts)
    
    # User2: SLOW (from scratch)
    if 'User2' in USERS and not USERS['User2']['use_processed_csv']:
        user2_workouts = process_user2_from_scratch(USERS['User2'])
        all_workouts.extend(user2_workouts)
    
    # Create DataFrame
    print(f"\n{'='*80}")
    print("CREATING FINAL DATASET")
    print(f"{'='*80}")
    
    df = pd.DataFrame(all_workouts)
    
    # Sort by date (handle mixed formats from User1 CSV and User2 XML)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.sort_values('date').reset_index(drop=True)
    
    # Summary statistics
    print(f"\nDataset Summary:")
    print(f"  Total workouts: {len(df)}")
    print(f"  Users: {df['user'].value_counts().to_dict()}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Total distance: {df['distance_km'].sum():.1f} km")
    print(f"  Total duration: {df['duration_min'].sum()/60:.1f} hours")
    print(f"  Avg sequence length: {df['seq_length'].mean():.0f} points")
    print(f"  Avg HR: {df['hr_avg'].mean():.1f} bpm (min={df['hr_min'].min():.0f}, max={df['hr_max'].max():.0f})")
    
    # Save main CSV
    output_path = OUTPUT_DIR / 'workouts_all_users.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved clean data to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    # Save metadata summary
    metadata = {
        'total_workouts': len(df),
        'users': df['user'].value_counts().to_dict(),
        'date_range': {
            'start': df['date'].min().isoformat(),
            'end': df['date'].max().isoformat()
        },
        'total_distance_km': float(df['distance_km'].sum()),
        'total_duration_hours': float(df['duration_min'].sum() / 60),
        'avg_sequence_length': float(df['seq_length'].mean()),
        'avg_hr': float(df['hr_avg'].mean()),
        'filters_applied': {
            'hr_quality': 'GOOD',
            'hr_samples_per_min_threshold': HR_QUALITY_THRESHOLD,
            'min_sequence_length': MIN_SEQUENCE_LENGTH,
            'has_gpx': True
        }
    }
    
    metadata_path = OUTPUT_DIR / 'metadata_summary.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved metadata to: {metadata_path}")
    
    print(f"\n{'='*80}")
    print("STEP 1 COMPLETE!")
    print(f"{'='*80}")
    print(f"\nNext step: Run 02_prepare_training_data.py to create PyTorch tensors")
    print(f"\nOutput files:")
    print(f"  • {output_path}")
    print(f"  • {metadata_path}")


if __name__ == "__main__":
    main()
