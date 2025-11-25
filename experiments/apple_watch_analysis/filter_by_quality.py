#!/usr/bin/env python3
"""
Filter Processed Workouts by HR Quality

Separates workouts into quality tiers based on heart rate sampling density:
- GOOD: ≥5 samples/min (detailed HR curves)
- MEDIUM: 1-5 samples/min (moderate quality)
- SPARSE: <1 sample/min (heavily interpolated)

Author: Apple Watch Analysis Pipeline
Date: 2025-11-25
"""

import json
import pandas as pd
from pathlib import Path


def main():
    """Filter and categorize workouts by HR quality"""
    print("=" * 70)
    print("FILTERING WORKOUTS BY HR QUALITY")
    print("=" * 70)
    
    # Load processing results
    results_path = Path("experiments/apple_watch_analysis/output/processing_results.json")
    
    if not results_path.exists():
        print(f"\n✗ Error: {results_path} not found")
        print("  Run process_all_workouts.py first")
        return
    
    print("\n1. Loading processing results...")
    with open(results_path) as f:
        results = json.load(f)
    
    successful = [r for r in results if r['success']]
    print(f"   ✓ Loaded {len(successful)} successful workouts")
    
    # Categorize by quality
    print("\n2. Categorizing by quality...")
    
    categories = {
        'GOOD': [],
        'MEDIUM': [],
        'SPARSE': []
    }
    
    for r in successful:
        quality = r.get('quality', 'UNKNOWN')
        if quality in categories:
            categories[quality].append(r)
    
    # Print summary
    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)
    
    for quality in ['GOOD', 'MEDIUM', 'SPARSE']:
        workouts = categories[quality]
        count = len(workouts)
        pct = 100 * count / len(successful) if successful else 0
        
        print(f"\n{quality} Quality:")
        print(f"  Count: {count} workouts ({pct:.1f}%)")
        
        if workouts:
            total_distance = sum(w['distance_km'] for w in workouts)
            total_duration = sum(w['duration_min'] for w in workouts)
            total_points = sum(w['aligned_points'] for w in workouts)
            avg_samples_per_min = sum(w['hr_samples_per_min'] for w in workouts) / count
            
            print(f"  Total distance: {total_distance:.1f} km")
            print(f"  Total duration: {total_duration/60:.1f} hours")
            print(f"  Total data points: {total_points:,}")
            print(f"  Avg HR samples/min: {avg_samples_per_min:.2f}")
    
    # Save filtered lists
    print("\n" + "=" * 70)
    print("SAVING FILTERED LISTS")
    print("=" * 70)
    
    output_dir = Path("experiments/apple_watch_analysis/output")
    
    for quality, workouts in categories.items():
        if workouts:
            output_path = output_dir / f"workouts_{quality.lower()}_quality.json"
            with open(output_path, 'w') as f:
                json.dump(workouts, f, indent=2)
            print(f"  ✓ {quality}: {output_path.name} ({len(workouts)} workouts)")
    
    # Create combined high-quality list (GOOD + MEDIUM)
    high_quality = categories['GOOD'] + categories['MEDIUM']
    if high_quality:
        output_path = output_dir / "workouts_high_quality.json"
        with open(output_path, 'w') as f:
            json.dump(high_quality, f, indent=2)
        print(f"  ✓ HIGH (GOOD+MEDIUM): {output_path.name} ({len(high_quality)} workouts)")
    
    print("=" * 70)
    
    # Print recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 70)
    print("For training:")
    print("  • Option A: Use all workouts (maximize data)")
    print("  • Option B: Use GOOD+MEDIUM only (better quality)")
    print("  • Option C: Use GOOD only (best quality, less data)")
    print("\nFor validation/testing:")
    print("  • Recommend using GOOD quality workouts only")
    print("=" * 70)


if __name__ == "__main__":
    main()
