#!/usr/bin/env python3
"""
Preprocess running workout data for heart rate prediction.

This script:
1. Loads running workouts from endomondoHR.json
2. Filters for complete sequences (speed, altitude, HR)
3. Pads/truncates to fixed length (300 timesteps)
4. Normalizes features
5. Splits into train/val/test sets by userId
6. Saves as PyTorch tensors
"""

import numpy as np
import pandas as pd
import torch
import ast
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'DATA' / 'endomondoHR.json'
OUTPUT_DIR = PROJECT_ROOT / 'DATA' / 'processed'
SEQUENCE_LENGTH = 300
MAX_SAMPLES = 10000  # Load first 10k workouts
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def load_workouts(filepath, n_samples=MAX_SAMPLES):
    """Load workouts from JSON file."""
    workouts = []
    print(f"Loading workouts from {filepath}...")
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading", total=n_samples)):
            if i >= n_samples:
                break
            try:
                workout = ast.literal_eval(line.strip())
                workouts.append(workout)
            except Exception as e:
                continue
    
    return workouts

def filter_valid_workouts(workouts):
    """Filter for genuine running workouts with complete data."""
    valid = []
    
    print("\nFiltering valid running workouts...")
    for w in tqdm(workouts, desc="Filtering"):
        # Must be a running workout
        if 'sport' not in w or w['sport'].lower() != 'run':
            continue
        
        # Must have all required fields
        required = ['speed', 'altitude', 'heart_rate', 'gender', 'userId']
        if not all(field in w and w[field] is not None for field in required):
            continue
        
        # Convert to arrays
        try:
            speed = np.array(w['speed'], dtype=float)
            altitude = np.array(w['altitude'], dtype=float)
            hr = np.array(w['heart_rate'], dtype=float)
        except:
            continue
        
        # Check for minimum length and valid values
        min_length = 50  # At least 50 timesteps
        if len(speed) < min_length or len(altitude) < min_length or len(hr) < min_length:
            continue
        
        # All sequences must have same length
        if not (len(speed) == len(altitude) == len(hr)):
            continue
        
        # Remove NaN/Inf values
        if np.any(np.isnan(speed)) or np.any(np.isnan(altitude)) or np.any(np.isnan(hr)):
            continue
        if np.any(np.isinf(speed)) or np.any(np.isinf(altitude)) or np.any(np.isinf(hr)):
            continue
        
        # Valid heart rate range (50-220 BPM)
        if np.any(hr < 50) or np.any(hr > 220):
            continue
        
        valid.append({
            'speed': speed,
            'altitude': altitude,
            'heart_rate': hr,
            'gender': w['gender'],
            'userId': w['userId'],
            'original_length': len(speed)
        })
    
    print(f"✓ Found {len(valid)} valid workouts")
    return valid

def pad_or_truncate(sequence, target_length):
    """Pad or truncate sequence to target length."""
    current_length = len(sequence)
    
    if current_length >= target_length:
        # Truncate: take first target_length elements
        return sequence[:target_length]
    else:
        # Pad: repeat last value
        padding = np.full(target_length - current_length, sequence[-1])
        return np.concatenate([sequence, padding])

def preprocess_sequences(workouts, seq_length=SEQUENCE_LENGTH):
    """Pad/truncate all sequences to fixed length."""
    print(f"\nPadding/truncating sequences to {seq_length} timesteps...")
    
    processed = []
    for w in tqdm(workouts, desc="Processing"):
        processed.append({
            'speed': pad_or_truncate(w['speed'], seq_length),
            'altitude': pad_or_truncate(w['altitude'], seq_length),
            'heart_rate': pad_or_truncate(w['heart_rate'], seq_length),
            'gender': w['gender'],
            'userId': w['userId'],
            'original_length': w['original_length']
        })
    
    return processed

def normalize_features(workouts_train, workouts_val, workouts_test):
    """Normalize speed and altitude using training set statistics."""
    print("\nNormalizing features...")
    
    # Collect all training sequences
    train_speed = np.concatenate([w['speed'] for w in workouts_train])
    train_altitude = np.concatenate([w['altitude'] for w in workouts_train])
    
    # Compute statistics on training data only
    speed_scaler = StandardScaler()
    altitude_scaler = StandardScaler()
    
    speed_scaler.fit(train_speed.reshape(-1, 1))
    altitude_scaler.fit(train_altitude.reshape(-1, 1))
    
    # Apply to all sets
    def apply_scaling(workouts):
        scaled = []
        for w in workouts:
            scaled.append({
                'speed': speed_scaler.transform(w['speed'].reshape(-1, 1)).flatten(),
                'altitude': altitude_scaler.transform(w['altitude'].reshape(-1, 1)).flatten(),
                'heart_rate': w['heart_rate'],  # Keep HR unnormalized for easier interpretation
                'gender': w['gender'],
                'userId': w['userId'],
                'original_length': w['original_length']
            })
        return scaled
    
    workouts_train = apply_scaling(workouts_train)
    workouts_val = apply_scaling(workouts_val)
    workouts_test = apply_scaling(workouts_test)
    
    # Save scalers for later use
    scaler_params = {
        'speed_mean': float(speed_scaler.mean_[0]),
        'speed_std': float(speed_scaler.scale_[0]),
        'altitude_mean': float(altitude_scaler.mean_[0]),
        'altitude_std': float(altitude_scaler.scale_[0])
    }
    
    print(f"  Speed: mean={scaler_params['speed_mean']:.2f}, std={scaler_params['speed_std']:.2f}")
    print(f"  Altitude: mean={scaler_params['altitude_mean']:.2f}, std={scaler_params['altitude_std']:.2f}")
    
    return workouts_train, workouts_val, workouts_test, scaler_params

def split_by_user(workouts, test_size=0.15, val_size=0.15):
    """Split data by userId to avoid data leakage."""
    print("\nSplitting by userId...")
    
    # Group by userId
    user_workouts = {}
    for w in workouts:
        uid = w['userId']
        if uid not in user_workouts:
            user_workouts[uid] = []
        user_workouts[uid].append(w)
    
    print(f"  Total users: {len(user_workouts)}")
    print(f"  Total workouts: {len(workouts)}")
    
    # Split users
    user_ids = list(user_workouts.keys())
    
    # First split: train+val vs test
    train_val_users, test_users = train_test_split(
        user_ids, test_size=test_size, random_state=RANDOM_SEED
    )
    
    # Second split: train vs val
    val_size_adjusted = val_size / (1 - test_size)
    train_users, val_users = train_test_split(
        train_val_users, test_size=val_size_adjusted, random_state=RANDOM_SEED
    )
    
    # Collect workouts
    train_workouts = [w for uid in train_users for w in user_workouts[uid]]
    val_workouts = [w for uid in val_users for w in user_workouts[uid]]
    test_workouts = [w for uid in test_users for w in user_workouts[uid]]
    
    print(f"  Train: {len(train_workouts)} workouts ({len(train_users)} users)")
    print(f"  Val: {len(val_workouts)} workouts ({len(val_users)} users)")
    print(f"  Test: {len(test_workouts)} workouts ({len(test_users)} users)")
    
    return train_workouts, val_workouts, test_workouts

def convert_to_tensors(workouts):
    """Convert workouts to PyTorch tensors."""
    n = len(workouts)
    seq_len = len(workouts[0]['speed'])
    
    # Initialize tensors
    speed = torch.zeros(n, seq_len, 1)
    altitude = torch.zeros(n, seq_len, 1)
    heart_rate = torch.zeros(n, seq_len, 1)
    gender = torch.zeros(n, 1)
    userId = torch.zeros(n, 1, dtype=torch.long)
    original_lengths = torch.zeros(n, 1, dtype=torch.long)
    
    # Fill tensors
    for i, w in enumerate(workouts):
        speed[i, :, 0] = torch.FloatTensor(w['speed'])
        altitude[i, :, 0] = torch.FloatTensor(w['altitude'])
        heart_rate[i, :, 0] = torch.FloatTensor(w['heart_rate'])
        gender[i, 0] = 1.0 if w['gender'] == 'male' else 0.0
        userId[i, 0] = w['userId']
        original_lengths[i, 0] = w['original_length']
    
    return {
        'speed': speed,
        'altitude': altitude,
        'heart_rate': heart_rate,
        'gender': gender,
        'userId': userId,
        'original_lengths': original_lengths
    }

def main():
    """Main preprocessing pipeline."""
    print("="*80)
    print("HEART RATE PREDICTION - DATA PREPROCESSING")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    workouts = load_workouts(str(DATA_PATH), n_samples=MAX_SAMPLES)
    
    # Filter valid workouts
    workouts = filter_valid_workouts(workouts)
    
    if len(workouts) == 0:
        print("ERROR: No valid workouts found!")
        return
    
    # Pad/truncate sequences
    workouts = preprocess_sequences(workouts, SEQUENCE_LENGTH)
    
    # Split by user
    train_workouts, val_workouts, test_workouts = split_by_user(workouts)
    
    # Normalize features
    train_workouts, val_workouts, test_workouts, scaler_params = normalize_features(
        train_workouts, val_workouts, test_workouts
    )
    
    # Convert to tensors
    print("\nConverting to PyTorch tensors...")
    train_data = convert_to_tensors(train_workouts)
    val_data = convert_to_tensors(val_workouts)
    test_data = convert_to_tensors(test_workouts)
    
    # Save data
    print("\nSaving preprocessed data...")
    torch.save(train_data, str(OUTPUT_DIR / 'train.pt'))
    torch.save(val_data, str(OUTPUT_DIR / 'val.pt'))
    torch.save(test_data, str(OUTPUT_DIR / 'test.pt'))
    
    # Save scaler parameters
    with open(str(OUTPUT_DIR / 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    # Save metadata
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_train': len(train_workouts),
        'num_val': len(val_workouts),
        'num_test': len(test_workouts),
        'random_seed': RANDOM_SEED
    }
    
    with open(str(OUTPUT_DIR / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Files saved to: {OUTPUT_DIR}/")
    print(f"  - train.pt ({len(train_workouts)} samples)")
    print(f"  - val.pt ({len(val_workouts)} samples)")
    print(f"  - test.pt ({len(test_workouts)} samples)")
    print(f"  - scaler_params.json")
    print(f"  - metadata.json")
    print("\nSequence shape: [batch_size, {SEQUENCE_LENGTH}, 1]")
    print("\nReady for model training!")

if __name__ == '__main__':
    main()
