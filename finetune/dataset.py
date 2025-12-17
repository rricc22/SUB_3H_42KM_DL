"""
PyTorch Dataset class for preprocessed workout data
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, Tuple


class WorkoutDataset(Dataset):
    """
    Dataset for workout sequences with heart rate, speed, and altitude data.
    
    Data format (from preprocessed .pt files):
    {
        'speed': (N, 500, 1),
        'altitude': (N, 500, 1),
        'heart_rate': (N, 500, 1),
        'timestamps': (N, 500, 1),
        'gender': (N, 1),
        'userId': (N, 1),
        'original_lengths': (N, 1)
    }
    """
    
    def __init__(self, data_path: str, use_mask: bool = True):
        """
        Initialize dataset
        
        Args:
            data_path: Path to .pt file (train.pt, val.pt, or test.pt)
            use_mask: Whether to use original_lengths for masking padded values
        """
        self.data_path = Path(data_path)
        self.use_mask = use_mask
        
        # Load data
        self.data = torch.load(self.data_path, weights_only=False)
        
        # Extract features
        self.speed = self.data['speed']  # (N, 500, 1)
        self.altitude = self.data['altitude']  # (N, 500, 1)
        self.heart_rate = self.data['heart_rate']  # (N, 500, 1)
        self.timestamps = self.data['timestamps']  # (N, 500, 1)
        self.original_lengths = self.data['original_lengths']  # (N, 1)
        
        self.n_samples = self.speed.shape[0]
        self.seq_length = self.speed.shape[1]
        
        print(f"Loaded dataset from: {self.data_path}")
        print(f"  Samples: {self.n_samples}")
        print(f"  Sequence length: {self.seq_length}")
        print(f"  Features: speed, altitude, heart_rate")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample
        
        Returns:
            features: (seq_len, 3) - [speed, altitude, heart_rate]
            target: (seq_len, 1) - heart_rate (same as input for reconstruction)
            mask: (seq_len, 1) - mask for original length (1=valid, 0=padding)
        """
        # Stack features: (seq_len, 3)
        features = torch.cat([
            self.speed[idx],      # (500, 1)
            self.altitude[idx],   # (500, 1)
            self.heart_rate[idx]  # (500, 1)
        ], dim=1)  # (500, 3)
        
        # Target is heart rate
        target = self.heart_rate[idx]  # (500, 1)
        
        # Create mask for original length
        if self.use_mask:
            orig_len = int(self.original_lengths[idx].item())
            mask = torch.zeros(self.seq_length, 1)
            mask[:orig_len] = 1.0
        else:
            mask = torch.ones(self.seq_length, 1)
        
        return features, target, mask
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        stats = {
            'n_samples': self.n_samples,
            'seq_length': self.seq_length,
            'hr_mean': self.heart_rate.mean().item(),
            'hr_std': self.heart_rate.std().item(),
            'hr_min': self.heart_rate.min().item(),
            'hr_max': self.heart_rate.max().item(),
            'speed_mean': self.speed.mean().item(),
            'speed_std': self.speed.std().item(),
            'altitude_mean': self.altitude.mean().item(),
            'altitude_std': self.altitude.std().item(),
            'avg_original_length': self.original_lengths.float().mean().item(),
        }
        return stats
    
    def print_statistics(self):
        """Print dataset statistics"""
        stats = self.get_statistics()
        print("\n=== Dataset Statistics ===")
        print(f"Samples: {stats['n_samples']}")
        print(f"Sequence length: {stats['seq_length']}")
        print(f"Average original length: {stats['avg_original_length']:.1f}")
        print(f"\nHeart Rate:")
        print(f"  Mean: {stats['hr_mean']:.1f} bpm")
        print(f"  Std: {stats['hr_std']:.1f} bpm")
        print(f"  Range: [{stats['hr_min']:.1f}, {stats['hr_max']:.1f}]")
        print(f"\nSpeed (normalized):")
        print(f"  Mean: {stats['speed_mean']:.4f}")
        print(f"  Std: {stats['speed_std']:.4f}")
        print(f"\nAltitude (normalized):")
        print(f"  Mean: {stats['altitude_mean']:.4f}")
        print(f"  Std: {stats['altitude_std']:.4f}")


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    use_mask: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders
    
    Args:
        data_dir: Directory containing train.pt, val.pt, test.pt
        batch_size: Batch size
        num_workers: Number of worker processes for data loading
        use_mask: Whether to use masking for padded sequences
    
    Returns:
        train_loader, val_loader, test_loader
    """
    data_dir_path = Path(data_dir)
    
    # Create datasets
    train_dataset = WorkoutDataset(str(data_dir_path / 'train.pt'), use_mask=use_mask)
    val_dataset = WorkoutDataset(str(data_dir_path / 'val.pt'), use_mask=use_mask)
    test_dataset = WorkoutDataset(str(data_dir_path / 'test.pt'), use_mask=use_mask)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print("\n=== DataLoaders Created ===")
    print(f"Train batches: {len(train_loader)} (batch_size={batch_size})")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    print("Testing WorkoutDataset...")
    
    data_path = "DATA/Private_runs_processed/train.pt"
    dataset = WorkoutDataset(data_path)
    dataset.print_statistics()
    
    # Test __getitem__
    features, target, mask = dataset[0]
    print(f"\nSample shapes:")
    print(f"  Features: {features.shape}")
    print(f"  Target: {target.shape}")
    print(f"  Mask: {mask.shape}")
    print(f"  Valid timesteps: {mask.sum().item():.0f}")
    
    # Test dataloader
    print("\n\nTesting DataLoader...")
    train_loader, val_loader, test_loader = create_dataloaders(
        "DATA/Private_runs_processed",
        batch_size=8
    )
    
    # Get a batch
    features_batch, target_batch, mask_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Features: {features_batch.shape}")
    print(f"  Target: {target_batch.shape}")
    print(f"  Mask: {mask_batch.shape}")
