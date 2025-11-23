#!/usr/bin/env python3
"""
Streaming preprocessing for heart rate prediction - Low RAM usage
Processes 253K workouts with constant ~500 MB RAM
"""

import numpy as np
import torch
import h5py
import ast
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'DATA' / 'endomondoHR.json'
OUTPUT_DIR = PROJECT_ROOT / 'DATA' / 'processed'
TEMP_DIR = PROJECT_ROOT / 'DATA' / 'temp'
SEQUENCE_LENGTH = 500
BATCH_SIZE = 5000  # Save every 500 valid workouts
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def count_lines(filepath):
    """Count total lines for progress bar."""
    with open(filepath, 'r') as f:
        return sum(1 for _ in f)

def validate_workout(w):
    """Check if workout is valid. Returns (is_valid, processed_data or None)."""
    # Filter: must be 'run'
    if 'sport' not in w or w['sport'].lower() != 'run':
        return False, None
    
    # Filter: required fields
    required = ['speed', 'altitude', 'heart_rate', 'gender', 'userId']
    if not all(field in w and w[field] is not None for field in required):
        return False, None
    
    # Convert to arrays
    try:
        speed = np.array(w['speed'], dtype=float)
        altitude = np.array(w['altitude'], dtype=float)
        hr = np.array(w['heart_rate'], dtype=float)
        timestamp = np.array(w['timestamp'], dtype=float) if 'timestamp' in w else None
    except:
        return False, None
    
    # Filter: minimum length
    if len(speed) < 50 or len(altitude) < 50 or len(hr) < 50:
        return False, None
    
    # Filter: same length
    if not (len(speed) == len(altitude) == len(hr)):
        return False, None
    
    # Filter: no NaN/Inf
    if np.any(np.isnan(speed)) or np.any(np.isnan(altitude)) or np.any(np.isnan(hr)):
        return False, None
    if np.any(np.isinf(speed)) or np.any(np.isinf(altitude)) or np.any(np.isinf(hr)):
        return False, None
    
    # Filter: valid HR range
    if np.any(hr < 50) or np.any(hr > 220):
        return False, None
    
    return True, {
        'speed': speed,
        'altitude': altitude,
        'heart_rate': hr,
        'timestamp': timestamp,
        'gender': w['gender'],
        'userId': w['userId'],
        'original_length': len(speed)
    }

def pad_or_truncate(sequence, target_length):
    """Pad or truncate sequence."""
    if len(sequence) >= target_length:
        return sequence[:target_length]
    else:
        padding = np.full(target_length - len(sequence), sequence[-1])
        return np.concatenate([sequence, padding])

def save_batch(workouts, batch_num, temp_dir):
    """Save batch to temporary file."""
    filepath = temp_dir / f"batch_{batch_num:04d}.pt"
    torch.save(workouts, str(filepath))

def stream_filter_and_process(filepath, temp_dir):
    """Stream through file, filter, process, save in batches."""
    print("=" * 80)
    print("PHASE 1: STREAMING FILTER & PREPROCESS")
    print("=" * 80)
    
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    total_lines = count_lines(str(filepath))
    print(f"Total workouts in file: {total_lines:,}")
    
    batch = []
    batch_num = 0
    total_valid = 0
    
    with open(str(filepath), 'r') as f:
        for line in tqdm(f, total=total_lines, desc="Processing"):
            try:
                w = ast.literal_eval(line.strip())
                is_valid, data = validate_workout(w)
                
                if is_valid:
                    # Pad/truncate sequences
                    processed = {
                        'speed': pad_or_truncate(data['speed'], SEQUENCE_LENGTH),
                        'altitude': pad_or_truncate(data['altitude'], SEQUENCE_LENGTH),
                        'heart_rate': pad_or_truncate(data['heart_rate'], SEQUENCE_LENGTH),
                        'timestamp': pad_or_truncate(data['timestamp'], SEQUENCE_LENGTH) if data['timestamp'] is not None else None,
                        'gender': data['gender'],
                        'userId': data['userId'],
                        'original_length': data['original_length']
                    }
                    batch.append(processed)
                    total_valid += 1
                    
                    # Save batch when full
                    if len(batch) >= BATCH_SIZE:
                        save_batch(batch, batch_num, temp_dir)
                        batch_num += 1
                        batch = []
                        gc.collect()  # Force garbage collection
            except:
                continue
    
    # Save remaining workouts
    if batch:
        save_batch(batch, batch_num, temp_dir)
        batch_num += 1
    
    print(f"\n✓ Found {total_valid} valid workouts")
    print(f"✓ Saved in {batch_num} batches")
    
    return batch_num

def load_all_batches(temp_dir, num_batches):
    """Load all batches from temp files."""
    print("\nLoading all batches...")
    all_workouts = []
    
    for i in tqdm(range(num_batches), desc="Loading batches"):
        filepath = temp_dir / f"batch_{i:04d}.pt"
        batch = torch.load(str(filepath), weights_only=False)
        all_workouts.extend(batch)
    
    return all_workouts

def split_by_user(workouts, test_size=0.15, val_size=0.15):
    """Split by userId."""
    print("\nSplitting by userId...")
    
    user_workouts = {}
    for w in workouts:
        uid = w['userId']
        if uid not in user_workouts:
            user_workouts[uid] = []
        user_workouts[uid].append(w)
    
    user_ids = list(user_workouts.keys())
    
    train_val_users, test_users = train_test_split(
        user_ids, test_size=test_size, random_state=RANDOM_SEED
    )
    
    val_size_adjusted = val_size / (1 - test_size)
    train_users, val_users = train_test_split(
        train_val_users, test_size=val_size_adjusted, random_state=RANDOM_SEED
    )
    
    train_workouts = [w for uid in train_users for w in user_workouts[uid]]
    val_workouts = [w for uid in val_users for w in user_workouts[uid]]
    test_workouts = [w for uid in test_users for w in user_workouts[uid]]
    
    print(f"  Train: {len(train_workouts)} workouts ({len(train_users)} users)")
    print(f"  Val: {len(val_workouts)} workouts ({len(val_users)} users)")
    print(f"  Test: {len(test_workouts)} workouts ({len(test_users)} users)")
    
    return train_workouts, val_workouts, test_workouts

def normalize_features(workouts_train, workouts_val, workouts_test):
    """Normalize speed and altitude."""
    print("\nNormalizing features...")
    
    train_speed = np.concatenate([w['speed'] for w in workouts_train])
    train_altitude = np.concatenate([w['altitude'] for w in workouts_train])
    
    speed_scaler = StandardScaler()
    altitude_scaler = StandardScaler()
    
    speed_scaler.fit(train_speed.reshape(-1, 1))
    altitude_scaler.fit(train_altitude.reshape(-1, 1))
    
    def apply_scaling(workouts):
        scaled = []
        for w in workouts:
            scaled_workout = {
                'speed': speed_scaler.transform(w['speed'].reshape(-1, 1)).flatten(),
                'altitude': altitude_scaler.transform(w['altitude'].reshape(-1, 1)).flatten(),
                'heart_rate': w['heart_rate'],
                'timestamp': w['timestamp'],
                'gender': w['gender'],
                'userId': w['userId'],
                'original_length': w['original_length']
            }
            scaled.append(scaled_workout)
        return scaled
    
    workouts_train = apply_scaling(workouts_train)
    workouts_val = apply_scaling(workouts_val)
    workouts_test = apply_scaling(workouts_test)
    
    scaler_params = {
        'speed_mean': float(speed_scaler.mean_[0]),
        'speed_std': float(speed_scaler.scale_[0]),
        'altitude_mean': float(altitude_scaler.mean_[0]),
        'altitude_std': float(altitude_scaler.scale_[0])
    }
    
    print(f"  Speed: mean={scaler_params['speed_mean']:.2f}, std={scaler_params['speed_std']:.2f}")
    print(f"  Altitude: mean={scaler_params['altitude_mean']:.2f}, std={scaler_params['altitude_std']:.2f}")
    
    return workouts_train, workouts_val, workouts_test, scaler_params

def convert_to_tensors(workouts):
    """Convert to PyTorch tensors."""
    n = len(workouts)
    seq_len = SEQUENCE_LENGTH
    
    speed = torch.zeros(n, seq_len, 1)
    altitude = torch.zeros(n, seq_len, 1)
    heart_rate = torch.zeros(n, seq_len, 1)
    timestamps = torch.zeros(n, seq_len, 1)
    gender = torch.zeros(n, 1)
    userId = torch.zeros(n, 1, dtype=torch.long)
    original_lengths = torch.zeros(n, 1, dtype=torch.long)
    
    for i, w in enumerate(workouts):
        speed[i, :, 0] = torch.FloatTensor(w['speed'])
        altitude[i, :, 0] = torch.FloatTensor(w['altitude'])
        heart_rate[i, :, 0] = torch.FloatTensor(w['heart_rate'])
        
        if w['timestamp'] is not None:
            timestamps[i, :, 0] = torch.FloatTensor(w['timestamp'])
        
        gender[i, 0] = 1.0 if w['gender'].lower() == 'male' else 0.0
        userId[i, 0] = w['userId']
        original_lengths[i, 0] = w['original_length']
    
    return {
        'speed': speed,
        'altitude': altitude,
        'heart_rate': heart_rate,
        'timestamps': timestamps,
        'gender': gender,
        'userId': userId,
        'original_lengths': original_lengths
    }

def save_as_pt(data, filepath):
    """Save as PyTorch format."""
    torch.save(data, str(filepath))

def save_as_h5(data, filepath):
    """Save as HDF5 format."""
    with h5py.File(str(filepath), 'w') as f:
        for key, tensor in data.items():
            f.create_dataset(key, data=tensor.numpy(), compression='gzip', compression_opts=4)

def cleanup_temp_files(temp_dir):
    """Remove temporary batch files."""
    print("\nCleaning up temporary files...")
    import shutil
    if temp_dir.exists():
        shutil.rmtree(str(temp_dir))
    print("✓ Cleanup complete")

def main():
    print("=" * 80)
    print("STREAMING PREPROCESSING - LOW RAM USAGE")
    print("=" * 80)
    print(f"Dataset: {DATA_PATH}")
    print(f"Batch size: {BATCH_SIZE} workouts")
    print(f"Output: Both .pt and .h5 formats")
    print("=" * 80)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Stream filter and preprocess
    num_batches = stream_filter_and_process(DATA_PATH, TEMP_DIR)
    
    # Phase 2: Load all batches
    print("\n" + "=" * 80)
    print("PHASE 2: LOADING & SPLITTING")
    print("=" * 80)
    all_workouts = load_all_batches(TEMP_DIR, num_batches)
    
    if len(all_workouts) == 0:
        print("ERROR: No valid workouts found!")
        return
    
    # Split by user
    train_workouts, val_workouts, test_workouts = split_by_user(all_workouts)
    
    # Free memory
    del all_workouts
    gc.collect()
    
    # Normalize
    print("\n" + "=" * 80)
    print("PHASE 3: NORMALIZATION & CONVERSION")
    print("=" * 80)
    train_workouts, val_workouts, test_workouts, scaler_params = normalize_features(
        train_workouts, val_workouts, test_workouts
    )
    
    # Convert to tensors
    print("\nConverting to tensors...")
    train_data = convert_to_tensors(train_workouts)
    val_data = convert_to_tensors(val_workouts)
    test_data = convert_to_tensors(test_workouts)
    
    # Save in both formats
    print("\n" + "=" * 80)
    print("PHASE 4: SAVING")
    print("=" * 80)
    
    print("\nSaving PyTorch format (.pt)...")
    save_as_pt(train_data, OUTPUT_DIR / 'train.pt')
    save_as_pt(val_data, OUTPUT_DIR / 'val.pt')
    save_as_pt(test_data, OUTPUT_DIR / 'test.pt')
    
    print("Saving HDF5 format (.h5)...")
    save_as_h5(train_data, OUTPUT_DIR / 'train.h5')
    save_as_h5(val_data, OUTPUT_DIR / 'val.h5')
    save_as_h5(test_data, OUTPUT_DIR / 'test.h5')
    
    # Save metadata
    with open(str(OUTPUT_DIR / 'scaler_params.json'), 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_train': len(train_workouts),
        'num_val': len(val_workouts),
        'num_test': len(test_workouts),
        'random_seed': RANDOM_SEED,
        'version': 'streaming',
        'formats': ['pt', 'h5']
    }
    
    with open(str(OUTPUT_DIR / 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Cleanup
    cleanup_temp_files(TEMP_DIR)
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"\nPyTorch format (.pt):")
    print(f"  - train.pt ({len(train_workouts)} samples)")
    print(f"  - val.pt ({len(val_workouts)} samples)")
    print(f"  - test.pt ({len(test_workouts)} samples)")
    print(f"\nHDF5 format (.h5):")
    print(f"  - train.h5 (compressed)")
    print(f"  - val.h5 (compressed)")
    print(f"  - test.h5 (compressed)")
    print(f"\nMetadata:")
    print(f"  - scaler_params.json")
    print(f"  - metadata.json")
    print("\n✓ Ready for training!")

if __name__ == '__main__':
    main()
