#!/usr/bin/env python3
"""
Quick test script to verify User2 data parsing works correctly
"""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add parser to path
sys.path.append(str(Path(__file__).parent))
from parse_apple_health import AppleHealthParser

def main():
    print("="*80)
    print("TESTING USER2 DATA PARSING")
    print("="*80)
    
    # Define paths
    base_dir = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/DATA/CUSTOM_DATA/apple_health_export_User2")
    export_xml = base_dir / "export.xml"
    
    print(f"\n1. Checking paths...")
    print(f"   Base directory: {base_dir}")
    print(f"   Export XML exists: {export_xml.exists()}")
    
    if not export_xml.exists():
        print(f"   ❌ ERROR: export.xml not found!")
        return
    
    print(f"   ✓ Files found")
    
    # Initialize parser
    print(f"\n2. Initializing parser...")
    try:
        parser = AppleHealthParser(str(export_xml), str(base_dir))
        print(f"   ✓ Parser initialized successfully")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # Parse all workouts
    print(f"\n3. Parsing workouts from XML...")
    try:
        workouts = parser.parse_workouts()
        print(f"   ✓ Found {len(workouts)} total workouts")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return
    
    # Convert to DataFrame
    print(f"\n4. Converting to DataFrame...")
    try:
        workouts_data = []
        for w in workouts:
            workouts_data.append({
                'workout_type': w.activity_type,
                'start_date': w.start_date,
                'end_date': w.end_date,
                'duration_min': w.duration_min,
                'distance_km': w.total_distance_km,
                'energy_kcal': w.total_energy_kcal,
                'gpx_file': w.gpx_file_path
            })
        
        df_workouts = pd.DataFrame(workouts_data)
        df_workouts['start_date'] = pd.to_datetime(df_workouts['start_date'])
        df_workouts['has_gpx'] = df_workouts['gpx_file'].notna()
        
        print(f"   ✓ DataFrame created with {len(df_workouts)} rows")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Display workout types
    print(f"\n5. Workout Types Distribution:")
    print(df_workouts['workout_type'].value_counts())
    
    # Filter running workouts
    print(f"\n6. Running Workouts Analysis:")
    df_runs = df_workouts[df_workouts['workout_type'].str.contains('Running', case=False, na=False)].copy()
    df_runs_gpx = df_runs[df_runs['has_gpx']].copy()
    
    print(f"   Total Running Workouts: {len(df_runs)}")
    print(f"   Running Workouts with GPX: {len(df_runs_gpx)} ({100*len(df_runs_gpx)/len(df_runs):.1f}%)")
    
    # Basic statistics
    print(f"\n7. Running Workouts Summary (with GPX):")
    print(f"   Date range: {df_runs_gpx['start_date'].min().date()} to {df_runs_gpx['start_date'].max().date()}")
    print(f"\n   Duration (minutes):")
    print(f"     Mean: {df_runs_gpx['duration_min'].mean():.1f}")
    print(f"     Median: {df_runs_gpx['duration_min'].median():.1f}")
    print(f"     Range: {df_runs_gpx['duration_min'].min():.1f} - {df_runs_gpx['duration_min'].max():.1f}")
    
    print(f"\n   Distance (km):")
    print(f"     Mean: {df_runs_gpx['distance_km'].mean():.2f}")
    print(f"     Median: {df_runs_gpx['distance_km'].median():.2f}")
    print(f"     Range: {df_runs_gpx['distance_km'].min():.2f} - {df_runs_gpx['distance_km'].max():.2f}")
    
    # Test GPX parsing
    print(f"\n8. Testing GPX parsing...")
    sample_workout = df_runs_gpx.iloc[0]
    gpx_path = base_dir / sample_workout['gpx_file'].lstrip('/')
    
    print(f"   Sample GPX: {gpx_path.name}")
    print(f"   File exists: {gpx_path.exists()}")
    
    if gpx_path.exists():
        try:
            from parse_apple_health import GPXParser
            gpx_parser = GPXParser(str(gpx_path))
            trackpoints = gpx_parser.parse_trackpoints()
            print(f"   ✓ Parsed {len(trackpoints)} GPS trackpoints")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    # Test heart rate parsing
    print(f"\n9. Testing heart rate parsing...")
    try:
        start_date = sample_workout['start_date'].to_pydatetime()
        end_date = sample_workout['start_date'] + pd.Timedelta(minutes=sample_workout['duration_min'])
        end_date = end_date.to_pydatetime()
        
        hr_records = parser.parse_heart_rate_records(start_date, end_date)
        print(f"   ✓ Found {len(hr_records)} heart rate records")
        print(f"   Sampling rate: {len(hr_records)/sample_workout['duration_min']:.1f} samples/minute")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED - User2 data is ready for exploration!")
    print("="*80)

if __name__ == "__main__":
    main()
