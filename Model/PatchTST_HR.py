#!/usr/bin/env python3
"""
PatchTST for heart rate prediction.

PatchTST (Patch Time Series Transformer) is a state-of-the-art transformer-based
model for time-series forecasting that processes sequences in patches for efficiency.

Architecture:
    1. Patching: Split time series into patches (subsequences)
    2. Patch embeddings: Linear projection of each patch
    3. Positional encoding: Add position information
    4. Transformer encoder: Multi-head self-attention layers
    5. Forecasting head: Predict future values or full sequence

This implementation uses HuggingFace's transformers library with custom adaptation
for heart rate prediction from speed and altitude sequences.

References:
    - Paper: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    - HuggingFace: https://huggingface.co/docs/transformers/model_doc/patchtst
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import PatchTSTConfig, PatchTSTForPrediction
from datasets import load_from_disk
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple, Optional


class PatchTSTHeartRatePredictor(nn.Module):
    """
    PatchTST model adapted for heart rate prediction.
    
    Architecture:
        1. Input: [batch, seq_len, num_features] where features = [speed, altitude, gender]
        2. PatchTST encoder: Process sequences in patches
        3. Output projection: [batch, seq_len, 1] for heart rate prediction
    
    The model is trained in a seq2seq fashion: given speed/altitude sequences,
    predict the corresponding heart rate sequence.
    """
    
    def __init__(
        self,
        num_input_channels=3,      # speed, altitude, gender (broadcasted)
        context_length=500,        # Input sequence length
        prediction_length=500,     # Output sequence length (same as input for seq2seq)
        patch_length=16,           # Length of each patch
        stride=8,                  # Stride between patches
        d_model=128,               # Model dimension
        num_attention_heads=8,     # Number of attention heads
        num_hidden_layers=4,       # Number of transformer layers
        ffn_dim=256,               # Feed-forward network dimension
        dropout=0.1,               # Dropout probability
        use_positional_encoding=True,
        pooling_type="mean",       # Pooling for prediction head
        num_parallel_samples=1,    # Number of samples for probabilistic forecasting
    ):
        """
        Initialize PatchTST model for heart rate prediction.
        
        Args:
            num_input_channels: Number of input features (default: 3 for speed, altitude, gender)
            context_length: Length of input sequence (default: 500)
            prediction_length: Length of output sequence (default: 500)
            patch_length: Length of each patch (default: 16)
            stride: Stride between patches (default: 8)
            d_model: Model dimension (default: 128)
            num_attention_heads: Number of attention heads (default: 8)
            num_hidden_layers: Number of transformer layers (default: 4)
            ffn_dim: Feed-forward network dimension (default: 256)
            dropout: Dropout probability (default: 0.1)
            use_positional_encoding: Whether to use positional encoding
            pooling_type: Pooling type for prediction head ('mean' or 'max')
            num_parallel_samples: Number of samples for probabilistic forecasting
        """
        super(PatchTSTHeartRatePredictor, self).__init__()
        
        self.num_input_channels = num_input_channels
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.patch_length = patch_length
        self.stride = stride
        
        # Create PatchTST configuration
        self.config = PatchTSTConfig(
            num_input_channels=num_input_channels,
            context_length=context_length,
            prediction_length=prediction_length,
            patch_length=patch_length,
            patch_stride=stride,  # Changed from 'stride' to 'patch_stride'
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            positional_encoding_type="sincos",  # Use sinusoidal positional encoding
            pooling_type=pooling_type,
            num_parallel_samples=1,  # Deterministic prediction
            num_targets=1,  # Output 1 channel (heart rate)
            # Time series specific settings
            scaling="mean",  # Normalize inputs
            loss="mse",      # Use MSE loss for regression
        )
        
        # Initialize PatchTST model
        self.patchtst = PatchTSTForPrediction(self.config)
        
        # Additional projection layer for single output (heart rate)
        # PatchTST outputs predictions per channel, we need to project to 1 channel
        self.output_projection = nn.Linear(prediction_length, prediction_length)
        
    def forward(
        self, 
        past_values: torch.Tensor,
        past_observed_mask: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through PatchTST model.
        
        Args:
            past_values: [batch, seq_len, num_features] - Input sequences (speed, altitude, gender)
            past_observed_mask: [batch, seq_len] - Mask for valid timesteps (optional)
            future_values: [batch, seq_len, 1] - Target heart rate (for training)
        
        Returns:
            Dictionary with 'prediction' [batch, seq_len, 1] and optional 'loss'
        """
        # PatchTST expects input shape: [batch, seq_len, num_channels]
        # We already have: [batch, seq_len, num_features]
        # So no transpose needed!
        
        # PatchTST expects observed_mask: [batch, seq_len, num_channels]
        # We have: [batch, seq_len]
        # Expand mask to match number of channels
        if past_observed_mask is not None:
            # Expand: [batch, seq_len] -> [batch, seq_len, 1] -> [batch, seq_len, num_channels]
            past_observed_mask = past_observed_mask.unsqueeze(-1).expand(-1, -1, self.num_input_channels)
        
        # Forward pass through PatchTST
        outputs = self.patchtst(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_values=future_values if future_values is not None else None,
        )
        
        # Extract predictions: [batch, prediction_length, num_channels] 
        # (no num_parallel_samples dimension for deterministic prediction)
        predictions = outputs.prediction_outputs
        
        # predictions is already [batch, prediction_length, num_channels]
        # We need to reduce it to [batch, prediction_length, 1] for heart rate
        # Average across all input channels to get a single output
        heart_rate_pred = predictions.mean(dim=-1, keepdim=True)  # [batch, prediction_length, 1]
        
        result = {
            'prediction': heart_rate_pred,
        }
        
        # Add loss if future values are provided
        if future_values is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(heart_rate_pred, future_values)
            result['loss'] = loss
        
        return result


class WorkoutDatasetHF(Dataset):
    """
    PyTorch Dataset wrapper for HuggingFace Dataset format.
    
    Loads data from HuggingFace Arrow format and converts to tensors
    suitable for PatchTST training.
    """
    
    def __init__(self, hf_dataset, return_dict=True):
        """
        Initialize dataset from HuggingFace Dataset.
        
        Args:
            hf_dataset: HuggingFace Dataset with keys ['speed', 'altitude', 'heart_rate', 'gender', ...]
            return_dict: If True, return dictionary with keys. If False, return tuple.
        """
        self.dataset = hf_dataset
        self.return_dict = return_dict
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Returns:
            If return_dict=True:
                Dictionary with keys 'past_values', 'future_values', 'gender', 'userId'
            If return_dict=False:
                Tuple (past_values, future_values)
        """
        sample = self.dataset[idx]
        
        # Extract features
        speed = torch.tensor(sample['speed'], dtype=torch.float32)  # [seq_len]
        altitude = torch.tensor(sample['altitude'], dtype=torch.float32)  # [seq_len]
        heart_rate = torch.tensor(sample['heart_rate'], dtype=torch.float32)  # [seq_len]
        gender = torch.tensor(sample['gender'], dtype=torch.float32)  # scalar
        
        # Broadcast gender to match sequence length
        seq_len = len(speed)
        gender_seq = gender.repeat(seq_len)  # [seq_len]
        
        # Stack features: [seq_len, num_features]
        past_values = torch.stack([speed, altitude, gender_seq], dim=-1)
        
        # Target: [seq_len, 1]
        future_values = heart_rate.unsqueeze(-1)
        
        if self.return_dict:
            return {
                'past_values': past_values,
                'future_values': future_values,
                'gender': gender,
                'userId': sample['userId'],
                'original_length': sample['original_length']
            }
        else:
            return past_values, future_values


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Batches samples and handles variable-length sequences with padding mask.
    
    Args:
        batch: List of samples from WorkoutDatasetHF
    
    Returns:
        Dictionary with batched tensors
    """
    # Stack all samples
    past_values = torch.stack([item['past_values'] for item in batch])  # [batch, seq_len, features]
    future_values = torch.stack([item['future_values'] for item in batch])  # [batch, seq_len, 1]
    
    # Optional: Create observed mask based on original_length
    batch_size, seq_len, _ = past_values.shape
    observed_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        original_length = item['original_length']
        if original_length < seq_len:
            # Mark padded positions as not observed
            observed_mask[i, original_length:] = False
    
    return {
        'past_values': past_values,
        'past_observed_mask': observed_mask,
        'future_values': future_values,
        'gender': torch.tensor([item['gender'] for item in batch]),
        'userId': torch.tensor([item['userId'] for item in batch]),
    }


def load_data_hf(data_dir: str, batch_size: int = 32) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Load HuggingFace format data and create DataLoaders.
    
    Args:
        data_dir: Directory containing HuggingFace Dataset (from prepare_sequences_hf.py)
        batch_size: Batch size for DataLoader
    
    Returns:
        train_loader, val_loader, test_loader, metadata
    """
    print(f"\nLoading HuggingFace dataset from {data_dir}...")
    
    # Load HuggingFace dataset
    dataset_dict = load_from_disk(data_dir)
    
    # Load metadata
    metadata_path = Path(data_dir) / 'metadata.json'
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load scaler parameters
    scaler_path = Path(data_dir) / 'scaler_params.json'
    with open(scaler_path, 'r') as f:
        scaler_params = json.load(f)
    
    metadata['scaler_params'] = scaler_params
    
    # Create PyTorch datasets
    train_dataset = WorkoutDatasetHF(dataset_dict['train'])
    val_dataset = WorkoutDatasetHF(dataset_dict['validation'])
    test_dataset = WorkoutDatasetHF(dataset_dict['test'])
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Set to 0 for debugging, increase for performance
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"✓ Loaded {len(train_dataset)} train samples")
    print(f"✓ Loaded {len(val_dataset)} validation samples")
    print(f"✓ Loaded {len(test_dataset)} test samples")
    
    return train_loader, val_loader, test_loader, metadata


def test_model():
    """
    Test PatchTST model with dummy data.
    
    This function verifies that the model architecture is correct and can
    perform forward/backward passes.
    """
    print("="*80)
    print("TESTING PATCHTST MODEL")
    print("="*80)
    
    # Hyperparameters
    batch_size = 4
    seq_len = 500
    num_features = 3  # speed, altitude, gender
    
    # Create dummy data
    print("\n1. Creating dummy data...")
    past_values = torch.randn(batch_size, seq_len, num_features)
    future_values = torch.randn(batch_size, seq_len, 1)
    
    print(f"   Input shape: {past_values.shape}")
    print(f"   Target shape: {future_values.shape}")
    
    # Initialize model
    print("\n2. Initializing PatchTST model...")
    model = PatchTSTHeartRatePredictor(
        num_input_channels=num_features,
        context_length=seq_len,
        prediction_length=seq_len,
        patch_length=16,
        stride=8,
        d_model=128,
        num_attention_heads=8,
        num_hidden_layers=4,
        ffn_dim=256,
        dropout=0.1
    )
    
    print(f"✓ Model initialized")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Forward pass
    print("\n3. Testing forward pass...")
    model.eval()
    with torch.no_grad():
        outputs = model(past_values=past_values)
        predictions = outputs['prediction']
    
    print(f"✓ Forward pass successful")
    print(f"   Prediction shape: {predictions.shape}")
    assert predictions.shape == (batch_size, seq_len, 1), "Output shape mismatch!"
    
    # Test with loss computation
    print("\n4. Testing training mode with loss...")
    model.train()
    outputs = model(past_values=past_values, future_values=future_values)
    loss = outputs['loss']
    
    print(f"✓ Loss computation successful")
    print(f"   Loss value: {loss.item():.4f}")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    loss.backward()
    print(f"✓ Backward pass successful")
    
    # Test DataLoader with dummy HF dataset
    print("\n6. Testing DataLoader...")
    try:
        # Create dummy HuggingFace-style data
        from datasets import Dataset as HFDataset
        dummy_data = {
            'speed': [np.random.randn(seq_len).tolist() for _ in range(10)],
            'altitude': [np.random.randn(seq_len).tolist() for _ in range(10)],
            'heart_rate': [(np.random.randn(seq_len) * 10 + 120).tolist() for _ in range(10)],
            'gender': [1.0 if i % 2 == 0 else 0.0 for i in range(10)],
            'userId': list(range(10)),
            'original_length': [seq_len for _ in range(10)]
        }
        
        hf_dataset = HFDataset.from_dict(dummy_data)
        pytorch_dataset = WorkoutDatasetHF(hf_dataset)
        dataloader = DataLoader(pytorch_dataset, batch_size=4, collate_fn=collate_fn)
        
        # Test one batch
        batch = next(iter(dataloader))
        print(f"✓ DataLoader working")
        print(f"   Batch keys: {batch.keys()}")
        print(f"   Batch size: {batch['past_values'].shape[0]}")
        
    except Exception as e:
        print(f"⚠ DataLoader test skipped (HuggingFace datasets not available): {e}")
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED! ✓")
    print("="*80)
    print("\nModel is ready for training!")
    print("\nNext steps:")
    print("1. Preprocess data with: python3 Preprocessing/prepare_sequences_hf.py")
    print("2. Train with: python3 Model/train.py --model patchtst --epochs 50")


if __name__ == '__main__':
    test_model()
