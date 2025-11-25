#!/usr/bin/env python3
"""
Create Training Dataset from Apple Watch Workouts

Converts processed Apple Watch workouts into the format expected by LSTM/PatchTST models:
- Sequences padded/truncated to 500 timesteps
- Speed and altitude normalized (Z-score)
- Train/Val/Test splits (temporal: 70/15/15)
- Compatible with existing training scripts

Output: PyTorch format (.pt files) compatible with Model/train.py

Author: Apple Watch Analysis Pipeline
Date: 2025-11-25
"""

import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler


# Constants (matching AGENTS.md specifications)
SEQUENCE_LENGTH = 500
RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def load_processed_workouts(quality_filter='all'):
    """
    Load processed workout metadata
    
    Args:
        quality_filter: 'all', 'good', 'high', or 'medium'
    
    Returns:
        list of workout metadata dicts
    """
    output_dir = Path("experiments/apple_watch_analysis/output")
    
    if quality_filter == 'all':
        results_path = output_dir / "processing_results.json"
        with open(results_path) as f:
            results = json.load(f)
        workouts = [r for r in results if r['success']]
    elif quality_filter == 'good':
        with open(output_dir / "workouts_good_quality.json") as f:
            workouts = json.load(f)
    elif quality_filter == 'high':
        with open(output_dir / "workouts_high_quality.json") as f:
            workouts = json.load(f)
    elif quality_filter == 'medium':
        with open(output_dir / "workouts_medium_quality.json") as f:
            workouts = json.load(f)
    else:
        raise ValueError(f"Invalid quality filter: {quality_filter}")
    
    return workouts


def load_workout_csv(workout_id):
    """Load processed CSV file for a single workout"""
    csv_path = Path(f"experiments/apple_watch_analysis/processed_workouts/{workout_id}_processed.csv")
    
    if not csv_path.exists():
        return None
    
    df = pd.read_csv(csv_path)
    return df


def resample_to_fixed_length(sequence, target_length=SEQUENCE_LENGTH):
    """
    Resample sequence to fixed length using linear interpolation
    
    Args:
        sequence: numpy array of shape (seq_len,)
        target_length: target sequence length
    
    Returns:
        resampled sequence of shape (target_length,)
    """
    if len(sequence) == 0:
        return np.zeros(target_length)
    
    if len(sequence) == target_length:
        return sequence
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(sequence))
    x_new = np.linspace(0, 1, target_length)
    resampled = np.interp(x_new, x_old, sequence)
    
    return resampled


def create_splits(workouts, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Create temporal train/val/test splits
    
    Args:
        workouts: list of workout metadata
        train_ratio: proportion for training
        val_ratio: proportion for validation
    
    Returns:
        train_workouts, val_workouts, test_workouts
    """
    # Sort by date (temporal split)
    workouts_sorted = sorted(workouts, key=lambda x: x['date'])
    
    n_total = len(workouts_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_workouts = workouts_sorted[:n_train]
    val_workouts = workouts_sorted[n_train:n_train+n_val]
    test_workouts = workouts_sorted[n_train+n_val:]
    
    return train_workouts, val_workouts, test_workouts


def process_workout_to_sequence(df):
    """
    Extract and resample sequences from workout dataframe
    
    Args:
        df: workout dataframe
    
    Returns:
        dict with resampled sequences
    """
    if df is None or len(df) < 10:
        return None
    
    # Extract sequences
    speed = df['speed_kmh'].values  # km/h
    altitude = df['elevation'].values  # meters
    heart_rate = df['heart_rate'].values  # BPM
    
    # Handle NaN/Inf
    speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)
    altitude = np.nan_to_num(altitude, nan=0.0, posinf=0.0, neginf=0.0)
    heart_rate = np.nan_to_num(heart_rate, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Resample to fixed length
    speed_resampled = resample_to_fixed_length(speed, SEQUENCE_LENGTH)
    altitude_resampled = resample_to_fixed_length(altitude, SEQUENCE_LENGTH)
    hr_resampled = resample_to_fixed_length(heart_rate, SEQUENCE_LENGTH)
    
    return {
        'speed': speed_resampled,
        'altitude': altitude_resampled,
        'heart_rate': hr_resampled
    }


def create_dataset(workouts, data_dir, normalize=False, scaler_speed=None, scaler_altitude=None):
    """
    Create dataset from list of workouts
    
    Args:
        workouts: list of workout metadata
        data_dir: directory containing processed CSV files
        normalize: whether to normalize features
        scaler_speed: fitted scaler for speed (if normalizing)
        scaler_altitude: fitted scaler for altitude (if normalizing)
    
    Returns:
        dict with arrays and scalers
    """
    sequences = []
    failed = 0
    
    for workout_meta in workouts:
        workout_id = workout_meta['workout_id']
        df = load_workout_csv(workout_id)
        
        seq = process_workout_to_sequence(df)
        if seq is None:
            failed += 1
            continue
        
        sequences.append(seq)
    
    if len(sequences) == 0:
        raise ValueError(f"No valid sequences found! Failed: {failed}")
    
    # Stack into arrays
    speed_array = np.stack([s['speed'] for s in sequences])  # (N, 500)
    altitude_array = np.stack([s['altitude'] for s in sequences])  # (N, 500)
    hr_array = np.stack([s['heart_rate'] for s in sequences])  # (N, 500)
    
    # Normalize if requested
    if normalize:
        # Fit scalers on training data (first split only)
        if scaler_speed is None:
            scaler_speed = StandardScaler()
            speed_flat = speed_array.flatten().reshape(-1, 1)
            scaler_speed.fit(speed_flat)
        
        if scaler_altitude is None:
            scaler_altitude = StandardScaler()
            altitude_flat = altitude_array.flatten().reshape(-1, 1)
            scaler_altitude.fit(altitude_flat)
        
        # Transform
        speed_array = scaler_speed.transform(speed_array.flatten().reshape(-1, 1)).reshape(speed_array.shape)
        altitude_array = scaler_altitude.transform(altitude_array.flatten().reshape(-1, 1)).reshape(altitude_array.shape)
    
    print(f"  ✓ Processed {len(sequences)} workouts ({failed} failed)")
    print(f"    Speed shape: {speed_array.shape}, range: [{speed_array.min():.2f}, {speed_array.max():.2f}]")
    print(f"    Altitude shape: {altitude_array.shape}, range: [{altitude_array.min():.2f}, {altitude_array.max():.2f}]")
    print(f"    HR shape: {hr_array.shape}, range: [{hr_array.min():.0f}, {hr_array.max():.0f}]")
    
    return {
        'speed': speed_array,
        'altitude': altitude_array,
        'heart_rate': hr_array,
        'scaler_speed': scaler_speed,
        'scaler_altitude': scaler_altitude,
        'n_sequences': len(sequences),
        'n_failed': failed
    }


def save_pytorch_format(train_data, val_data, test_data, output_dir):
    """
    Save datasets in PyTorch format (compatible with existing training scripts)
    
    Args:
        train_data, val_data, test_data: dataset dicts
        output_dir: output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to PyTorch tensors
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        dataset = {
            'speed': torch.FloatTensor(data['speed']),
            'altitude': torch.FloatTensor(data['altitude']),
            'heart_rate': torch.FloatTensor(data['heart_rate']),
            'userId': torch.zeros(data['n_sequences'], dtype=torch.long),  # Single user (Apple Watch owner)
            'gender': torch.ones(data['n_sequences'], dtype=torch.long),  # Placeholder (0 or 1)
        }
        
        output_path = output_dir / f"{split_name}.pt"
        torch.save(dataset, output_path)
        print(f"  ✓ Saved {split_name}.pt ({data['n_sequences']} sequences)")
    
    # Save normalization parameters
    scaler_info = {
        'speed_mean': float(train_data['scaler_speed'].mean_[0]),
        'speed_std': float(train_data['scaler_speed'].scale_[0]),
        'altitude_mean': float(train_data['scaler_altitude'].mean_[0]),
        'altitude_std': float(train_data['scaler_altitude'].scale_[0]),
    }
    
    with open(output_dir / 'scaler_params.json', 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    print(f"  ✓ Saved scaler_params.json")


def main():
    """Main workflow"""
    print("=" * 80)
    print("CREATING TRAINING DATASET FROM APPLE WATCH WORKOUTS")
    print("=" * 80)
    
    # Configuration
    quality_filter = 'all'  # Options: 'all', 'good', 'high', 'medium'
    output_dir = Path("DATA/apple_watch_processed")
    
    print(f"\nConfiguration:")
    print(f"  Quality filter: {quality_filter}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Train/Val/Test: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%}")
    print(f"  Output directory: {output_dir}")
    
    # Load workouts
    print("\n1. Loading processed workouts...")
    workouts = load_processed_workouts(quality_filter)
    print(f"   ✓ Loaded {len(workouts)} workouts")
    
    # Create splits
    print("\n2. Creating temporal splits...")
    train_workouts, val_workouts, test_workouts = create_splits(workouts)
    print(f"   ✓ Train: {len(train_workouts)} workouts")
    print(f"   ✓ Val:   {len(val_workouts)} workouts")
    print(f"   ✓ Test:  {len(test_workouts)} workouts")
    
    # Process training set (fit scalers)
    print("\n3. Processing training set...")
    train_data = create_dataset(
        train_workouts,
        data_dir="experiments/apple_watch_analysis/processed_workouts",
        normalize=True
    )
    
    # Process validation set (use train scalers)
    print("\n4. Processing validation set...")
    val_data = create_dataset(
        val_workouts,
        data_dir="experiments/apple_watch_analysis/processed_workouts",
        normalize=True,
        scaler_speed=train_data['scaler_speed'],
        scaler_altitude=train_data['scaler_altitude']
    )
    
    # Process test set (use train scalers)
    print("\n5. Processing test set...")
    test_data = create_dataset(
        test_workouts,
        data_dir="experiments/apple_watch_analysis/processed_workouts",
        normalize=True,
        scaler_speed=train_data['scaler_speed'],
        scaler_altitude=train_data['scaler_altitude']
    )
    
    # Save in PyTorch format
    print("\n6. Saving datasets...")
    save_pytorch_format(train_data, val_data, test_data, output_dir)
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET CREATION SUMMARY")
    print("=" * 80)
    print(f"Total workouts: {len(workouts)}")
    print(f"Train: {train_data['n_sequences']} sequences ({train_data['n_failed']} failed)")
    print(f"Val:   {val_data['n_sequences']} sequences ({val_data['n_failed']} failed)")
    print(f"Test:  {test_data['n_sequences']} sequences ({test_data['n_failed']} failed)")
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  • train.pt ({train_data['n_sequences']} sequences × {SEQUENCE_LENGTH} timesteps)")
    print(f"  • val.pt ({val_data['n_sequences']} sequences × {SEQUENCE_LENGTH} timesteps)")
    print(f"  • test.pt ({test_data['n_sequences']} sequences × {SEQUENCE_LENGTH} timesteps)")
    print(f"  • scaler_params.json")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("Train LSTM model:")
    print(f"  python3 Model/train.py --model lstm --epochs 100 --batch_size 32 \\")
    print(f"    --data_dir DATA/apple_watch_processed")
    print("\nTrain PatchTST model (convert format first):")
    print(f"  python3 Preprocessing/convert_pt_to_hf.py --input_dir DATA/apple_watch_processed")
    print(f"  python3 Model/train_patchtst.py --epochs 50 --batch_size 32")
    print("=" * 80)


if __name__ == "__main__":
    main()
