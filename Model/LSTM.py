#!/usr/bin/env python3
"""
Basic LSTM model for heart rate prediction from running workout sequences.

Architecture:
    Input: Concatenate[speed, altitude] + gender (repeated across timesteps)
    LSTM: 2-layer bidirectional LSTM
    Output: Heart rate sequence [batch, 500, 1]

Model does NOT include user embeddings (see LSTM_with_embeddings.py for that variant).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class HeartRateLSTM(nn.Module):
    """
    Basic LSTM model for heart rate prediction.
    
    Architecture:
        1. Concatenate speed + altitude + gender → [batch, seq_len, 3]
        2. LSTM layers with dropout
        3. Fully connected layer → [batch, seq_len, 1]
    """
    
    def __init__(
        self, 
        input_size=3,           # speed + altitude + gender
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2,
        bidirectional=False
    ):
        """
        Initialize LSTM model.
        
        Args:
            input_size: Number of input features (default: 3)
            hidden_size: LSTM hidden dimension (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.2)
            bidirectional: Use bidirectional LSTM (default: False)
        """
        super(HeartRateLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        
        # Fully connected output layer
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(lstm_output_size, 1)
        
    def forward(self, speed, altitude, gender, original_lengths=None):
        """
        Forward pass.
        
        Args:
            speed: [batch, seq_len, 1] - Normalized speed sequences
            altitude: [batch, seq_len, 1] - Normalized altitude sequences
            gender: [batch, 1] - Binary gender (1.0=male, 0.0=female)
            original_lengths: [batch, 1] - Original sequence lengths (for masking)
        
        Returns:
            heart_rate_pred: [batch, seq_len, 1] - Predicted heart rate in BPM
        """
        batch_size, seq_len, _ = speed.shape
        
        # Expand gender to match sequence length: [batch, 1] → [batch, seq_len, 1]
        gender_expanded = gender.unsqueeze(1).expand(batch_size, seq_len, 1)
        
        # Concatenate features: [batch, seq_len, 3]
        x = torch.cat([speed, altitude, gender_expanded], dim=2)
        
        # LSTM forward pass
        # lstm_out: [batch, seq_len, hidden_size * num_directions]
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)
        
        # Fully connected layer: [batch, seq_len, 1]
        heart_rate_pred = self.fc(lstm_out)
        
        return heart_rate_pred
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class WorkoutDataset(Dataset):
    """
    PyTorch Dataset for workout data.
    
    Loads preprocessed tensors and returns individual samples.
    """
    
    def __init__(self, data_dict):
        """
        Initialize dataset from preprocessed tensor dictionary.
        
        Args:
            data_dict: Dictionary with keys:
                - 'speed': [n, seq_len, 1]
                - 'altitude': [n, seq_len, 1]
                - 'heart_rate': [n, seq_len, 1]
                - 'gender': [n, 1]
                - 'userId': [n, 1]
                - 'original_lengths': [n, 1]
        """
        self.speed = data_dict['speed']
        self.altitude = data_dict['altitude']
        self.heart_rate = data_dict['heart_rate']
        self.gender = data_dict['gender']
        self.userId = data_dict['userId']
        self.original_lengths = data_dict['original_lengths']
        
        self.n_samples = len(self.speed)
        
    def __len__(self):
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (speed, altitude, gender, heart_rate, original_length)
        """
        return (
            self.speed[idx],              # [seq_len, 1]
            self.altitude[idx],            # [seq_len, 1]
            self.gender[idx],              # [1]
            self.heart_rate[idx],          # [seq_len, 1] - TARGET
            self.original_lengths[idx]     # [1]
        )


# Example usage and model info
if __name__ == '__main__':
    print("="*80)
    print("BASIC LSTM MODEL FOR HEART RATE PREDICTION")
    print("="*80)
    
    # Create model
    model = HeartRateLSTM(
        input_size=3,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 500
    
    dummy_speed = torch.randn(batch_size, seq_len, 1)
    dummy_altitude = torch.randn(batch_size, seq_len, 1)
    dummy_gender = torch.randint(0, 2, (batch_size, 1)).float()
    dummy_lengths = torch.randint(50, seq_len, (batch_size, 1))
    
    print(f"\n" + "="*80)
    print("TEST FORWARD PASS")
    print("="*80)
    print(f"Input shapes:")
    print(f"  speed:       {dummy_speed.shape}")
    print(f"  altitude:    {dummy_altitude.shape}")
    print(f"  gender:      {dummy_gender.shape}")
    print(f"  lengths:     {dummy_lengths.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_speed, dummy_altitude, dummy_gender, dummy_lengths)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print("\n✓ Model initialized successfully!")
    print("\nTo train this model, use: python3 Model/train.py --model lstm")
