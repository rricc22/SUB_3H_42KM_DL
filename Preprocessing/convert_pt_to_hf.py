#!/usr/bin/env python3
"""
Convert existing PyTorch .pt files to HuggingFace Dataset format.

This script is MUCH faster than prepare_sequences_hf.py because it uses
the already preprocessed data from DATA/processed/*.pt files.

Usage:
    python3 Preprocessing/convert_pt_to_hf.py

Input:  DATA/processed/train.pt, val.pt, test.pt
Output: DATA/processed_hf/ (HuggingFace Dataset format)
"""

import torch
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
from datasets import Dataset, DatasetDict

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / 'DATA' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'DATA' / 'processed_hf'


def convert_pt_to_hf(pt_data):
    """
    Convert PyTorch tensors to HuggingFace Dataset format.
    
    Args:
        pt_data: Dictionary with PyTorch tensors from .pt file
            Expected keys: 'speed', 'altitude', 'heart_rate', 'gender', 
                          'userId', 'original_lengths'
    
    Returns:
        HuggingFace Dataset
    """
    # Extract tensors
    speed = pt_data['speed']  # [N, 500, 1]
    altitude = pt_data['altitude']  # [N, 500, 1]
    heart_rate = pt_data['heart_rate']  # [N, 500, 1]
    gender = pt_data['gender']  # [N, 1]
    userId = pt_data['userId']  # [N, 1]
    original_lengths = pt_data['original_lengths']  # [N, 1]
    
    num_samples = speed.shape[0]
    
    # Convert to lists (HuggingFace format)
    data = {
        'speed': [],
        'altitude': [],
        'heart_rate': [],
        'gender': [],
        'userId': [],
        'original_length': []
    }
    
    print(f"  Converting {num_samples} samples...")
    for i in tqdm(range(num_samples), desc="  Processing"):
        # Squeeze out the last dimension and convert to list
        data['speed'].append(speed[i, :, 0].numpy().tolist())
        data['altitude'].append(altitude[i, :, 0].numpy().tolist())
        data['heart_rate'].append(heart_rate[i, :, 0].numpy().tolist())
        data['gender'].append(float(gender[i, 0].item()))
        data['userId'].append(int(userId[i, 0].item()))
        data['original_length'].append(int(original_lengths[i, 0].item()))
    
    # Create HuggingFace Dataset
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df, preserve_index=False)
    
    return dataset


def main():
    """Main conversion pipeline."""
    print("="*80)
    print("CONVERT PYTORCH .PT FILES TO HUGGINGFACE FORMAT")
    print("="*80)
    print(f"\nInput:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    
    # Check if input files exist
    train_path = INPUT_DIR / 'train.pt'
    val_path = INPUT_DIR / 'val.pt'
    test_path = INPUT_DIR / 'test.pt'
    
    if not all([train_path.exists(), val_path.exists(), test_path.exists()]):
        print("\n❌ ERROR: PyTorch .pt files not found!")
        print("Please run preprocessing first:")
        print("  python3 Preprocessing/prepare_sequences_v2.py")
        return
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch tensors
    print("\n1. Loading PyTorch tensors...")
    train_data = torch.load(train_path, weights_only=False)
    val_data = torch.load(val_path, weights_only=False)
    test_data = torch.load(test_path, weights_only=False)
    print(f"   ✓ Train: {train_data['speed'].shape[0]} samples")
    print(f"   ✓ Val:   {val_data['speed'].shape[0]} samples")
    print(f"   ✓ Test:  {test_data['speed'].shape[0]} samples")
    
    # Convert to HuggingFace format
    print("\n2. Converting to HuggingFace Dataset...")
    print("   Train:")
    train_dataset = convert_pt_to_hf(train_data)
    print("   Validation:")
    val_dataset = convert_pt_to_hf(val_data)
    print("   Test:")
    test_dataset = convert_pt_to_hf(test_data)
    
    # Create DatasetDict
    print("\n3. Creating DatasetDict...")
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # Save dataset
    print("\n4. Saving HuggingFace Dataset...")
    dataset_dict.save_to_disk(str(OUTPUT_DIR))
    print(f"   ✓ Saved to {OUTPUT_DIR}")
    
    # Copy scaler parameters
    print("\n5. Copying scaler parameters...")
    scaler_src = INPUT_DIR / 'scaler_params.json'
    scaler_dst = OUTPUT_DIR / 'scaler_params.json'
    
    if scaler_src.exists():
        with open(scaler_src, 'r') as f:
            scaler_params = json.load(f)
        with open(scaler_dst, 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print(f"   ✓ Copied scaler_params.json")
    else:
        print("   ⚠ scaler_params.json not found (skipping)")
    
    # Create metadata
    print("\n6. Creating metadata...")
    metadata_src = INPUT_DIR / 'metadata.json'
    
    if metadata_src.exists():
        with open(metadata_src, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    # Update metadata for HuggingFace format
    metadata.update({
        'num_train': len(train_dataset),
        'num_val': len(val_dataset),
        'num_test': len(test_dataset),
        'format': 'huggingface_dataset',
        'features': ['speed', 'altitude', 'heart_rate', 'gender', 'userId', 'original_length'],
        'target': 'heart_rate',
        'converted_from': 'pytorch_pt_files'
    })
    
    metadata_path = OUTPUT_DIR / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"   ✓ Saved metadata.json")
    
    print("\n" + "="*80)
    print("CONVERSION COMPLETE!")
    print("="*80)
    print(f"Dataset saved to: {OUTPUT_DIR}/")
    print(f"  - train:      {len(train_dataset)} samples")
    print(f"  - validation: {len(val_dataset)} samples")
    print(f"  - test:       {len(test_dataset)} samples")
    print(f"  - scaler_params.json")
    print(f"  - metadata.json")
    print(f"\nSequence length: {metadata.get('sequence_length', 500)}")
    print("\nLoad with:")
    print(f"  from datasets import load_from_disk")
    print(f"  dataset = load_from_disk('{OUTPUT_DIR}')")
    print("\n✅ Ready for PatchTST training!")
    print(f"   python3 Model/train_patchtst.py --epochs 50 --batch_size 32")


if __name__ == '__main__':
    main()
