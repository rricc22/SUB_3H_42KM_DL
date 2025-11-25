#!/usr/bin/env python3
"""
Analyze HR Data Quality Across All Workouts

This script checks HR measurement density for all 285 workouts
to identify which ones have sufficient HR data for model training.

Author: Apple Watch Analysis Pipeline  
Date: 2025-11-25
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.append(str(Path(__file__).parent))
from parse_apple_health import AppleHealthParser


def main():
    print("=" * 70)
    print("HR DATA QUALITY ANALYSIS - ALL 285 WORKOUTS")
    print("=" * 70)
    
    # Paths
    base_dir = Path("/home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL/experiments/apple_health_export")
    export_xml = base_dir / "export.xml"
    workouts_json = Path("experiments/apple_watch_analysis/output/workouts_summary.json")
    
    # Load workouts
    print("\n1. Loading workouts...")
    with open(workouts_json) as f:
        workouts = json.load(f)
    
    print(f"   ✓ Loaded {len(workouts)} workouts")
    
    # Initialize parser
    parser = AppleHealthParser(str(export_xml), str(base_dir))
    
    # Analyze HR data for all workouts
    print("\n2. Analyzing HR data quality (this may take a few minutes)...")
    
    hr_quality = []
    
    for i, workout in enumerate(workouts, 1):
        if i % 50 == 0:
            print(f"   Processed {i}/{len(workouts)} workouts...")
        
        start_date = datetime.fromisoformat(workout['start_date'].replace('+00:00', ''))
        end_date = datetime.fromisoformat(workout['end_date'].replace('+00:00', ''))
        duration_min = workout['duration_min']
        
        # Count HR records
        try:
            hr_records = parser.parse_heart_rate_records(start_date, end_date)
            hr_count = len(hr_records)
            samples_per_min = hr_count / duration_min if duration_min > 0 else 0
            
            # Categorize quality
            if samples_per_min >= 5:
                quality = 'GOOD'
            elif samples_per_min >= 1:
                quality = 'ACCEPTABLE'
            else:
                quality = 'SPARSE'
            
            hr_quality.append({
                'workout_id': workout['workout_id'],
                'date': workout['start_date'][:10],
                'duration_min': duration_min,
                'hr_count': hr_count,
                'samples_per_min': samples_per_min,
                'quality': quality,
                'has_gpx': workout['gpx_file_path'] is not None
            })
            
        except Exception as e:
            hr_quality.append({
                'workout_id': workout['workout_id'],
                'date': workout['start_date'][:10],
                'duration_min': duration_min,
                'hr_count': 0,
                'samples_per_min': 0,
                'quality': 'ERROR',
                'has_gpx': workout['gpx_file_path'] is not None,
                'error': str(e)
            })
    
    print(f"   ✓ Analyzed all {len(workouts)} workouts")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY BY QUALITY")
    print("=" * 70)
    
    good = [w for w in hr_quality if w['quality'] == 'GOOD']
    acceptable = [w for w in hr_quality if w['quality'] == 'ACCEPTABLE']
    sparse = [w for w in hr_quality if w['quality'] == 'SPARSE']
    errors = [w for w in hr_quality if w['quality'] == 'ERROR']
    
    print(f"\nGOOD (≥5 samples/min):       {len(good):>3} workouts ({100*len(good)/len(workouts):.1f}%)")
    print(f"ACCEPTABLE (≥1 sample/min):  {len(acceptable):>3} workouts ({100*len(acceptable)/len(workouts):.1f}%)")
    print(f"SPARSE (<1 sample/min):      {len(sparse):>3} workouts ({100*len(sparse)/len(workouts):.1f}%)")
    print(f"ERRORS:                      {len(errors):>3} workouts ({100*len(errors)/len(workouts):.1f}%)")
    
    # Summary by year
    print("\n" + "=" * 70)
    print("SUMMARY BY YEAR")
    print("=" * 70)
    
    years = {}
    for w in hr_quality:
        year = w['date'][:4]
        if year not in years:
            years[year] = {'GOOD': 0, 'ACCEPTABLE': 0, 'SPARSE': 0, 'total': 0}
        years[year][w['quality']] = years[year].get(w['quality'], 0) + 1
        years[year]['total'] += 1
    
    print(f"\n{'Year':<6} {'Total':>7} {'Good':>7} {'Accept':>7} {'Sparse':>7} {'Good %':>8}")
    print("-" * 70)
    for year in sorted(years.keys()):
        stats = years[year]
        good_pct = 100 * stats['GOOD'] / stats['total']
        print(f"{year:<6} {stats['total']:>7} {stats['GOOD']:>7} {stats.get('ACCEPTABLE', 0):>7} {stats.get('SPARSE', 0):>7} {good_pct:>7.1f}%")
    
    # Show top 20 workouts with best HR data
    print("\n" + "=" * 70)
    print("TOP 20 WORKOUTS BY HR QUALITY")
    print("=" * 70)
    
    best = sorted([w for w in hr_quality if w['quality'] != 'ERROR'], 
                  key=lambda x: x['samples_per_min'], reverse=True)[:20]
    
    print(f"\n{'Workout ID':<30} {'Date':<12} {'HR/min':>8} {'Quality':>10}")
    print("-" * 70)
    for w in best:
        print(f"{w['workout_id']:<30} {w['date']:<12} {w['samples_per_min']:>8.1f} {w['quality']:>10}")
    
    # Save results
    output_path = Path("experiments/apple_watch_analysis/output/hr_quality_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(hr_quality, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    usable = len(good) + len(acceptable)
    print(f"\n1. USABLE WORKOUTS: {usable} ({100*usable/len(workouts):.0f}%)")
    print(f"   - {len(good)} with GOOD HR quality (≥5 samples/min)")
    print(f"   - {len(acceptable)} with ACCEPTABLE HR quality (≥1 sample/min)")
    
    print(f"\n2. FOCUS ON RECENT DATA:")
    recent_good = len([w for w in good if w['date'] >= '2025'])
    recent_accept = len([w for w in acceptable if w['date'] >= '2025'])
    print(f"   - 2025 workouts: {recent_good} GOOD, {recent_accept} ACCEPTABLE")
    
    print(f"\n3. SPARSE HR DATA: {len(sparse)} workouts")
    print(f"   - Can still be used with interpolation")
    print(f"   - Less accurate but provides training signal")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
