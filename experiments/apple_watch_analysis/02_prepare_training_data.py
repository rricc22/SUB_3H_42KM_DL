#!/usr/bin/env python3
"""
Step 2: Prepare Training Data from Clean Apple Watch CSV

Converts clean CSV from Step 1 into PyTorch tensors for training:
- Resamples sequences to 500 timesteps (fixed length)
- Normalizes speed and altitude (Z-score on training set)
- Creates train/val/test splits (70/15/15 temporal split)
- Saves in PyTorch format compatible with Model/train.py

Input: DATA/apple_watch_clean/workouts_all_users.csv
Output: DATA/apple_watch_processed/{train,val,test}.pt

Author: Apple Watch Processing Pipeline
Date: 2025-11-25
"""

import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import ast

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent.parent
INPUT_FILE = PROJECT_ROOT / 'DATA' / 'apple_watch_clean' / 'workouts_all_users.csv'
OUTPUT_DIR = PROJECT_ROOT / 'DATA' / 'apple_watch_processed'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEQUENCE_LENGTH = 500
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def parse_sequence(seq_str):
    """Convert string representation of list to numpy array"""
    try:
        seq_list = ast.literal_eval(seq_str)
        return np.array(seq_list, dtype=np.float32)
    except:
        return np.array([], dtype=np.float32)


def resample_to_fixed_length(sequence, target_length=SEQUENCE_LENGTH):
    """
    Resample sequence to fixed length using linear interpolation
    
    Args:
        sequence: numpy array of shape (seq_len,)
        target_length: target sequence length (default 500)
    
    Returns:
        resampled sequence of shape (target_length,)
    """
    if len(sequence) == 0:
        return np.zeros(target_length, dtype=np.float32)
    
    if len(sequence) == target_length:
        return sequence.astype(np.float32)
    
    # Linear interpolation
    x_old = np.linspace(0, 1, len(sequence))
    x_new = np.linspace(0, 1, target_length)
    resampled = np.interp(x_new, x_old, sequence)
    
    return resampled.astype(np.float32)


def create_temporal_splits(df, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO):
    """
    Create temporal train/val/test splits
    
    Args:
        df: DataFrame with workouts (must have 'date' column)
        train_ratio: proportion for training (0.70)
        val_ratio: proportion for validation (0.15)
    
    Returns:
        train_df, val_df, test_df
    """
    # Sort by date (temporal split - oldest first)
    df_sorted = df.sort_values('date').reset_index(drop=True)
    
    n_total = len(df_sorted)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train:n_train+n_val]
    test_df = df_sorted.iloc[n_train+n_val:]
    
    return train_df, val_df, test_df


def process_split(df, split_name, scaler_speed=None, scaler_altitude=None):
    """
    Process a data split (train/val/test)
    
    Args:
        df: DataFrame with workouts
        split_name: 'train', 'val', or 'test'
        scaler_speed: fitted scaler for speed (None = fit new scaler)
        scaler_altitude: fitted scaler for altitude (None = fit new scaler)
    
    Returns:
        dict with tensors and scalers
    """
    print(f"\nProcessing {split_name} split ({len(df)} workouts)...")
    
    speed_list = []
    altitude_list = []
    hr_list = []
    timestamp_list = []
    userId_list = []
    gender_list = []
    original_length_list = []
    
    failed = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"  {split_name}"):
        # Parse sequences from string
        speed_seq = parse_sequence(row['speed_kmh'])
        altitude_seq = parse_sequence(row['altitude'])
        hr_seq = parse_sequence(row['heart_rate'])
        timestamp_seq = parse_sequence(row['timestamps'])
        
        # Track original length before resampling
        original_length = len(speed_seq)
        
        # Validate sequences
        if len(speed_seq) < 10 or len(altitude_seq) < 10 or len(hr_seq) < 10:
            failed += 1
            continue
        
        # Resample to fixed length (500 timesteps)
        speed_resampled = resample_to_fixed_length(speed_seq)
        altitude_resampled = resample_to_fixed_length(altitude_seq)
        hr_resampled = resample_to_fixed_length(hr_seq)
        timestamp_resampled = resample_to_fixed_length(timestamp_seq)
        
        # Handle NaN/Inf (should be rare after Step 1 filtering)
        speed_resampled = np.nan_to_num(speed_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        altitude_resampled = np.nan_to_num(altitude_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        hr_resampled = np.nan_to_num(hr_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        timestamp_resampled = np.nan_to_num(timestamp_resampled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Append to lists
        speed_list.append(speed_resampled)
        altitude_list.append(altitude_resampled)
        hr_list.append(hr_resampled)
        timestamp_list.append(timestamp_resampled)
        userId_list.append(row['userId'])
        gender_list.append(row['gender'])
        original_length_list.append(original_length)
    
    if len(speed_list) == 0:
        raise ValueError(f"No valid sequences in {split_name} split! Failed: {failed}")
    
    # Stack into arrays (N, 500)
    speed_array = np.stack(speed_list, axis=0)
    altitude_array = np.stack(altitude_list, axis=0)
    hr_array = np.stack(hr_list, axis=0)
    timestamp_array = np.stack(timestamp_list, axis=0)
    userId_array = np.array(userId_list, dtype=np.int64)
    gender_array = np.array(gender_list, dtype=np.float32)  # Changed to float32
    original_length_array = np.array(original_length_list, dtype=np.int64)
    
    print(f"  ✓ Loaded {len(speed_list)} sequences ({failed} failed)")
    
    # Normalize speed and altitude (Z-score on training set)
    if split_name == 'train':
        # Fit scalers on training data
        scaler_speed = StandardScaler()
        scaler_altitude = StandardScaler()
        
        speed_flat = speed_array.flatten().reshape(-1, 1)
        altitude_flat = altitude_array.flatten().reshape(-1, 1)
        
        scaler_speed.fit(speed_flat)
        scaler_altitude.fit(altitude_flat)
        
        print(f"  ✓ Fitted scalers on training data")
        print(f"    Speed: mean={scaler_speed.mean_[0]:.2f}, std={scaler_speed.scale_[0]:.2f}")
        print(f"    Altitude: mean={scaler_altitude.mean_[0]:.2f}, std={scaler_altitude.scale_[0]:.2f}")
    
    # Transform using scalers
    speed_normalized = scaler_speed.transform(speed_array.flatten().reshape(-1, 1)).reshape(speed_array.shape)
    altitude_normalized = scaler_altitude.transform(altitude_array.flatten().reshape(-1, 1)).reshape(altitude_array.shape)
    
    print(f"  ✓ Normalized features")
    print(f"    Speed: range=[{speed_normalized.min():.2f}, {speed_normalized.max():.2f}]")
    print(f"    Altitude: range=[{altitude_normalized.min():.2f}, {altitude_normalized.max():.2f}]")
    print(f"    HR: range=[{hr_array.min():.0f}, {hr_array.max():.0f}] bpm (not normalized)")
    print(f"    Timestamps: range=[{timestamp_array.min():.0f}, {timestamp_array.max():.0f}]")
    print(f"    Original lengths: range=[{original_length_array.min()}, {original_length_array.max()}]")
    
    return {
        'speed': speed_normalized.astype(np.float32),
        'altitude': altitude_normalized.astype(np.float32),
        'heart_rate': hr_array.astype(np.float32),  # Keep unnormalized (raw BPM)
        'timestamps': timestamp_array.astype(np.float32),
        'userId': userId_array,
        'gender': gender_array,
        'original_lengths': original_length_array,
        'scaler_speed': scaler_speed,
        'scaler_altitude': scaler_altitude,
        'n_sequences': len(speed_list),
        'n_failed': failed
    }


def save_pytorch_format(train_data, val_data, test_data, output_dir, df):
    """
    Save datasets in PyTorch format (compatible with Model/train.py)
    
    Args:
        train_data, val_data, test_data: processed data dicts
        output_dir: output directory path
        df: original DataFrame (for metadata)
    """
    print(f"\nSaving PyTorch datasets to: {output_dir}")
    
    for split_name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        # Convert to tensors and add 3rd dimension: [N, 500] → [N, 500, 1]
        dataset = {
            'speed': torch.from_numpy(data['speed']).unsqueeze(-1),           # [N, 500, 1]
            'altitude': torch.from_numpy(data['altitude']).unsqueeze(-1),     # [N, 500, 1]
            'heart_rate': torch.from_numpy(data['heart_rate']).unsqueeze(-1), # [N, 500, 1]
            'timestamps': torch.from_numpy(data['timestamps']).unsqueeze(-1), # [N, 500, 1]
            'userId': torch.from_numpy(data['userId']).unsqueeze(-1),         # [N, 1]
            'gender': torch.from_numpy(data['gender']).unsqueeze(-1),         # [N, 1]
            'original_lengths': torch.from_numpy(data['original_lengths']).unsqueeze(-1), # [N, 1]
        }
        
        output_path = output_dir / f"{split_name}.pt"
        torch.save(dataset, output_path)
        
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"  ✓ Saved {split_name}.pt ({data['n_sequences']} sequences, {file_size_mb:.2f} MB)")
    
    # Save normalization parameters (for inference)
    scaler_info = {
        'speed_mean': float(train_data['scaler_speed'].mean_[0]),
        'speed_std': float(train_data['scaler_speed'].scale_[0]),
        'altitude_mean': float(train_data['scaler_altitude'].mean_[0]),
        'altitude_std': float(train_data['scaler_altitude'].scale_[0]),
        'sequence_length': SEQUENCE_LENGTH,
    }
    
    scaler_path = output_dir / 'scaler_params.json'
    with open(scaler_path, 'w') as f:
        json.dump(scaler_info, f, indent=2)
    
    print(f"  ✓ Saved scaler_params.json")
    
    # Save metadata
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'num_train': train_data['n_sequences'],
        'num_val': val_data['n_sequences'],
        'num_test': test_data['n_sequences'],
        'random_seed': RANDOM_SEED,
        'version': 'apple_watch_v2',
        'source': 'apple_watch_clean',
        'num_users': int(df['userId'].nunique()),
        'formats': ['pt']
    }
    
    metadata_path = output_dir / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved metadata.json")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SCRIPT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("="*80)
    print("STEP 2: PREPARE TRAINING DATA FROM CLEAN CSV")
    print("="*80)
    
    print(f"\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Train/Val/Test split: {TRAIN_RATIO:.0%}/{VAL_RATIO:.0%}/{TEST_RATIO:.0%} (temporal)")
    print(f"  Random seed: {RANDOM_SEED}")
    
    # Load clean CSV from Step 1
    print(f"\n{'='*80}")
    print("1. LOADING CLEAN CSV")
    print(f"{'='*80}")
    
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}\nRun Step 1 first (01_concatenate_clean_data.py)")
    
    df = pd.read_csv(INPUT_FILE)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} workouts from {INPUT_FILE.name}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Users: {df['user'].unique()}")
    
    # Create temporal splits
    print(f"\n{'='*80}")
    print("2. CREATING TEMPORAL SPLITS")
    print(f"{'='*80}")
    
    train_df, val_df, test_df = create_temporal_splits(df)
    
    print(f"✓ Created splits:")
    print(f"  Train: {len(train_df)} workouts ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"  Val:   {len(val_df)} workouts ({val_df['date'].min()} to {val_df['date'].max()})")
    print(f"  Test:  {len(test_df)} workouts ({test_df['date'].min()} to {test_df['date'].max()})")
    
    # Process training set (fit scalers)
    print(f"\n{'='*80}")
    print("3. PROCESSING TRAINING SET")
    print(f"{'='*80}")
    train_data = process_split(train_df, 'train')
    
    # Process validation set (use train scalers)
    print(f"\n{'='*80}")
    print("4. PROCESSING VALIDATION SET")
    print(f"{'='*80}")
    val_data = process_split(
        val_df, 'val',
        scaler_speed=train_data['scaler_speed'],
        scaler_altitude=train_data['scaler_altitude']
    )
    
    # Process test set (use train scalers)
    print(f"\n{'='*80}")
    print("5. PROCESSING TEST SET")
    print(f"{'='*80}")
    test_data = process_split(
        test_df, 'test',
        scaler_speed=train_data['scaler_speed'],
        scaler_altitude=train_data['scaler_altitude']
    )
    
    # Save PyTorch datasets
    print(f"\n{'='*80}")
    print("6. SAVING PYTORCH DATASETS")
    print(f"{'='*80}")
    save_pytorch_format(train_data, val_data, test_data, OUTPUT_DIR, df)
    
    # Final summary
    print(f"\n{'='*80}")
    print("STEP 2 COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\nDataset Summary:")
    print(f"  Train: {train_data['n_sequences']} sequences ({train_data['n_failed']} failed)")
    print(f"  Val:   {val_data['n_sequences']} sequences ({val_data['n_failed']} failed)")
    print(f"  Test:  {test_data['n_sequences']} sequences ({test_data['n_failed']} failed)")
    print(f"  Total: {train_data['n_sequences'] + val_data['n_sequences'] + test_data['n_sequences']} sequences")
    
    print(f"\nOutput Files:")
    print(f"  • {OUTPUT_DIR / 'train.pt'}")
    print(f"  • {OUTPUT_DIR / 'val.pt'}")
    print(f"  • {OUTPUT_DIR / 'test.pt'}")
    print(f"  • {OUTPUT_DIR / 'scaler_params.json'}")
    print(f"  • {OUTPUT_DIR / 'metadata.json'}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("Train LSTM model:")
    print(f"  python3 Model/train.py --model lstm --epochs 100 --batch_size 32 \\")
    print(f"    --data_dir DATA/apple_watch_processed")
    
    print("\nTrain GRU model:")
    print(f"  python3 Model/train.py --model gru --epochs 100 --batch_size 32 \\")
    print(f"    --data_dir DATA/apple_watch_processed")
    
    print("\nTrain LSTM with embeddings:")
    print(f"  python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32 \\")
    print(f"    --data_dir DATA/apple_watch_processed")
    print(f"{'='*80}")


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    main()
