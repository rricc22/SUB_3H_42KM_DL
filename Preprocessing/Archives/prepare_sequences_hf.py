#!/usr/bin/env python3
"""
Preprocess running workout data for heart rate prediction - HUGGINGFACE FORMAT

This script creates HuggingFace Dataset format compatible with:
- HuggingFace Trainer API
- PatchTST fine-tuning
- Lazy loading for large datasets

Key differences from prepare_sequences_v2.py:
1. Output: HuggingFace Dataset (instead of PyTorch .pt files)
2. Data structure: Dict per sample (instead of batched tensors)
3. Format: Compatible with transformers.Trainer

This script:
1. Loads running workouts from endomondoHR.json
2. Filters for complete sequences (speed, altitude, HR)
3. Pads/truncates to fixed length (500 timesteps)
4. Normalizes features (speed, altitude only)
5. Splits into train/val/test sets by userId
6. Saves as HuggingFace Dataset (Arrow format)
"""

import numpy as np
import pandas as pd
import ast
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datasets import Dataset, DatasetDict

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / 'DATA' / 'endomondoHR.json'
OUTPUT_DIR = PROJECT_ROOT / 'DATA' / 'processed_hf'
SEQUENCE_LENGTH = 500
MAX_SAMPLES = None  # Load first 100k workouts (change to None for ALL data)
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
def load_workouts(filepath, n_samples=MAX_SAMPLES):
    """
    Load workouts from JSON file.
    
    Args:
        filepath: Path to endomondoHR.json
        n_samples: Number of workouts to load (None = all)
    
    Returns:
        List of workout dictionaries
    """
    workouts = []
    print(f"Loading workouts from {filepath}...")
    
    with open(filepath, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Loading", total=n_samples)):
            if n_samples and i >= n_samples:
                break
            try:
                workout = ast.literal_eval(line.strip())
                workouts.append(workout)
            except Exception:
                continue
    
    return workouts


# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: FILTER VALID WORKOUTS
# ═══════════════════════════════════════════════════════════════════════════
def filter_valid_workouts(workouts):
    """
    Filter for genuine running workouts with complete data.
    
    Filters applied:
    1. Sport must be 'run'
    2. Must have all required fields (speed, altitude, HR, gender, userId)
    3. Arrays must be convertible to float
    4. Minimum length >= 50 timesteps
    5. All sequences must have same length
    6. No NaN or Inf values
    7. Valid heart rate range (50-220 BPM)
    
    Returns:
        List of valid workout dictionaries with numpy arrays
    """
    valid = []
    
    print("\nFiltering valid running workouts...")
    for w in tqdm(workouts, desc="Filtering"):
        # Filter 1: Must be a running workout
        if 'sport' not in w or w['sport'].lower() != 'run':
            continue
        
        # Filter 2: Must have all required fields
        required = ['speed', 'altitude', 'heart_rate', 'gender', 'userId']
        if not all(field in w and w[field] is not None for field in required):
            continue
        
        # Filter 3: Convert to arrays
        try:
            speed = np.array(w['speed'], dtype=float)
            altitude = np.array(w['altitude'], dtype=float)
            hr = np.array(w['heart_rate'], dtype=float)
        except:
            continue
        
        # Filter 4: Check for minimum length (at least 50 timesteps)
        min_length = 50
        if len(speed) < min_length or len(altitude) < min_length or len(hr) < min_length:
            continue
        
        # Filter 5: All sequences must have same length
        if not (len(speed) == len(altitude) == len(hr)):
            continue
        
        # Filter 6: Remove NaN/Inf values
        if np.any(np.isnan(speed)) or np.any(np.isnan(altitude)) or np.any(np.isnan(hr)):
            continue
        if np.any(np.isinf(speed)) or np.any(np.isinf(altitude)) or np.any(np.isinf(hr)):
            continue
        
        # Filter 7: Valid heart rate range (50-220 BPM)
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


# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: PAD OR TRUNCATE SEQUENCES
# ═══════════════════════════════════════════════════════════════════════════
def pad_or_truncate(sequence, target_length):
    """
    Pad or truncate sequence to target length.
    
    Strategy:
    - If sequence > target_length: Take first target_length elements
    - If sequence < target_length: Repeat last value to fill
    
    Args:
        sequence: Numpy array of any length
        target_length: Desired output length
    
    Returns:
        Numpy array of length target_length
    """
    current_length = len(sequence)
    
    if current_length >= target_length:
        # Truncate: take first target_length elements
        return sequence[:target_length]
    else:
        # Pad: repeat last value
        padding = np.full(target_length - current_length, sequence[-1])
        return np.concatenate([sequence, padding])


def preprocess_sequences(workouts, seq_length=SEQUENCE_LENGTH):
    """
    Pad/truncate all sequences to fixed length.
    
    Args:
        workouts: List of workout dictionaries
        seq_length: Target sequence length (default: 500)
    
    Returns:
        List of processed workout dictionaries
    """
    print(f"\nPadding/truncating sequences to {seq_length} timesteps...")
    
    processed = []
    for w in tqdm(workouts, desc="Processing"):
        processed_workout = {
            'speed': pad_or_truncate(w['speed'], seq_length),
            'altitude': pad_or_truncate(w['altitude'], seq_length),
            'heart_rate': pad_or_truncate(w['heart_rate'], seq_length),
            'gender': w['gender'],
            'userId': w['userId'],
            'original_length': w['original_length']
        }
        processed.append(processed_workout)
    
    return processed


# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: SPLIT BY USER (AVOID DATA LEAKAGE)
# ═══════════════════════════════════════════════════════════════════════════
def split_by_user(workouts, test_size=0.15, val_size=0.15):
    """
    Split data by userId to avoid data leakage.
    
    IMPORTANT: All workouts from the same user go into the SAME split.
    
    Args:
        workouts: List of workout dictionaries
        test_size: Fraction of users for test set (default: 0.15)
        val_size: Fraction of users for validation set (default: 0.15)
    
    Returns:
        train_workouts, val_workouts, test_workouts
    """
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
    
    # Split users (NOT workouts!)
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
    
    # Collect workouts for each split
    train_workouts = [w for uid in train_users for w in user_workouts[uid]]
    val_workouts = [w for uid in val_users for w in user_workouts[uid]]
    test_workouts = [w for uid in test_users for w in user_workouts[uid]]
    
    print(f"  Train: {len(train_workouts)} workouts ({len(train_users)} users)")
    print(f"  Val: {len(val_workouts)} workouts ({len(val_users)} users)")
    print(f"  Test: {len(test_workouts)} workouts ({len(test_users)} users)")
    
    return train_workouts, val_workouts, test_workouts


# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: NORMALIZE FEATURES
# ═══════════════════════════════════════════════════════════════════════════
def normalize_features(workouts_train, workouts_val, workouts_test):
    """
    Normalize speed and altitude using training set statistics.
    
    IMPORTANT: Fit scaler ONLY on training data (prevents data leakage)
    Keep HR unnormalized for easy interpretation.
    
    Args:
        workouts_train, workouts_val, workouts_test: Lists of workout dictionaries
    
    Returns:
        Normalized workouts and scaler parameters
    """
    print("\nNormalizing features...")
    
    # Collect all training sequences (flatten to 1D)
    train_speed = np.concatenate([w['speed'] for w in workouts_train])
    train_altitude = np.concatenate([w['altitude'] for w in workouts_train])
    
    # Compute statistics on training data ONLY
    speed_scaler = StandardScaler()
    altitude_scaler = StandardScaler()
    
    speed_scaler.fit(train_speed.reshape(-1, 1))
    altitude_scaler.fit(train_altitude.reshape(-1, 1))
    
    # Apply scaling to all sets
    def apply_scaling(workouts):
        scaled = []
        for w in workouts:
            scaled_workout = {
                'speed': speed_scaler.transform(w['speed'].reshape(-1, 1)).flatten(),
                'altitude': altitude_scaler.transform(w['altitude'].reshape(-1, 1)).flatten(),
                'heart_rate': w['heart_rate'],  # Keep HR unnormalized
                'gender': w['gender'],
                'userId': w['userId'],
                'original_length': w['original_length']
            }
            scaled.append(scaled_workout)
        return scaled
    
    workouts_train = apply_scaling(workouts_train)
    workouts_val = apply_scaling(workouts_val)
    workouts_test = apply_scaling(workouts_test)
    
    # Save scaler parameters
    scaler_params = {
        'speed_mean': float(speed_scaler.mean_[0]),
        'speed_std': float(speed_scaler.scale_[0]),
        'altitude_mean': float(altitude_scaler.mean_[0]),
        'altitude_std': float(altitude_scaler.scale_[0])
    }
    
    print(f"  Speed: mean={scaler_params['speed_mean']:.2f}, std={scaler_params['speed_std']:.2f}")
    print(f"  Altitude: mean={scaler_params['altitude_mean']:.2f}, std={scaler_params['altitude_std']:.2f}")
    
    return workouts_train, workouts_val, workouts_test, scaler_params


# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: CONVERT TO HUGGINGFACE DATASET
# ═══════════════════════════════════════════════════════════════════════════
def convert_to_hf_dataset(workouts):
    """
    Convert workouts to HuggingFace Dataset format.
    
    Each sample is a dictionary with:
    - speed: array of shape [seq_len]
    - altitude: array of shape [seq_len]
    - heart_rate: array of shape [seq_len] (TARGET)
    - gender: float (1.0=male, 0.0=female)
    - userId: int
    - original_length: int
    
    Args:
        workouts: List of workout dictionaries
    
    Returns:
        HuggingFace Dataset
    """
    # Convert to pandas DataFrame (HuggingFace prefers this format)
    data = {
        'speed': [],
        'altitude': [],
        'heart_rate': [],
        'gender': [],
        'userId': [],
        'original_length': []
    }
    
    for w in workouts:
        data['speed'].append(w['speed'].tolist())
        data['altitude'].append(w['altitude'].tolist())
        data['heart_rate'].append(w['heart_rate'].tolist())
        data['gender'].append(1.0 if w['gender'].lower() == 'male' else 0.0)
        data['userId'].append(int(w['userId']))
        data['original_length'].append(int(w['original_length']))
    
    # Create HuggingFace Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    return dataset


# ═══════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════
def main():
    """Main preprocessing pipeline for HuggingFace format."""
    print("="*80)
    print("HEART RATE PREDICTION - DATA PREPROCESSING (HUGGINGFACE FORMAT)")
    print("="*80)
    print(f"\nOutput format: HuggingFace Dataset (Arrow format)")
    print(f"Compatible with: transformers.Trainer, PatchTST fine-tuning")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    workouts = load_workouts(str(DATA_PATH), n_samples=MAX_SAMPLES)
    
    # Step 2: Filter valid workouts
    workouts = filter_valid_workouts(workouts)
    
    if len(workouts) == 0:
        print("ERROR: No valid workouts found!")
        return
    
    # Step 3: Pad/truncate sequences
    workouts = preprocess_sequences(workouts, SEQUENCE_LENGTH)
    
    # Step 4: Split by user
    train_workouts, val_workouts, test_workouts = split_by_user(workouts)
    
    # Step 5: Normalize features
    train_workouts, val_workouts, test_workouts, scaler_params = normalize_features(
        train_workouts, val_workouts, test_workouts
    )
    
    # Step 6: Convert to HuggingFace Dataset
    print("\nConverting to HuggingFace Dataset format...")
    train_dataset = convert_to_hf_dataset(train_workouts)
    val_dataset = convert_to_hf_dataset(val_workouts)
    test_dataset = convert_to_hf_dataset(test_workouts)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save dataset
    print("\nSaving HuggingFace Dataset...")
    dataset_dict.save_to_disk(str(OUTPUT_DIR))
    
    # Save scaler parameters
    scaler_path = OUTPUT_DIR / 'scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)
    
    # Save metadata
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_train': len(train_workouts),
        'num_val': len(val_workouts),
        'num_test': len(test_workouts),
        'num_users': len(set(w['userId'] for w in train_workouts + val_workouts + test_workouts)),
        'random_seed': RANDOM_SEED,
        'format': 'huggingface_dataset',
        'features': ['speed', 'altitude', 'heart_rate', 'gender', 'userId', 'original_length'],
        'target': 'heart_rate'
    }
    
    metadata_path = OUTPUT_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("PREPROCESSING COMPLETE!")
    print("="*80)
    print(f"Dataset saved to: {OUTPUT_DIR}/")
    print(f"  - train: {len(train_workouts)} samples")
    print(f"  - validation: {len(val_workouts)} samples")
    print(f"  - test: {len(test_workouts)} samples")
    print(f"  - scaler_params.json")
    print(f"  - metadata.json")
    print(f"\nSequence shape: [batch_size, {SEQUENCE_LENGTH}]")
    print("\nLoad with:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{OUTPUT_DIR}')")
    print("\nReady for PatchTST fine-tuning!")


if __name__ == '__main__':
    main()
