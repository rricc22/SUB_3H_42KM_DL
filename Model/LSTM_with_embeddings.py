#!/usr/bin/env python3
"""
LSTM with user embeddings for heart rate prediction.

Architecture:
    Input: Concatenate[speed, altitude, gender, userId_embedding]
    Embedding: userId → embedding_dim (default: 16)
    LSTM: 2-layer bidirectional LSTM
    Output: Heart rate sequence [batch, 500, 1]

This model includes user ID embeddings for personalized predictions.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class HeartRateLSTMWithEmbeddings(nn.Module):
    """
    LSTM model with user embeddings for heart rate prediction.
    
    Architecture:
        1. Embed userId → [batch, embedding_dim]
        2. Concatenate speed + altitude + gender + userId_embedding → [batch, seq_len, 3+embedding_dim]
        3. LSTM layers with dropout
        4. Fully connected layer → [batch, seq_len, 1]
    """
    
    def __init__(
        self, 
        num_users,                    # Number of unique users
        embedding_dim=16,             # User embedding dimension
        hidden_size=64, 
        num_layers=2, 
        dropout=0.2,
        bidirectional=False
    ):
        """
        Initialize LSTM model with user embeddings.
        
        Args:
            num_users: Number of unique users in dataset
            embedding_dim: Dimension of user embeddings (default: 16)
            hidden_size: LSTM hidden dimension (default: 64)
            num_layers: Number of LSTM layers (default: 2)
            dropout: Dropout probability (default: 0.2)
            bidirectional: Use bidirectional LSTM (default: False)
        """
        super(HeartRateLSTMWithEmbeddings, self).__init__()
        
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        # User embedding layer
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,
            embedding_dim=embedding_dim,
            padding_idx=None
        )
        
        # Input size: speed(1) + altitude(1) + gender(1) + user_embedding(embedding_dim)
        input_size = 3 + embedding_dim
        
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
        
    def forward(self, speed, altitude, gender, userId, original_lengths=None):
        """
        Forward pass.
        
        Args:
            speed: [batch, seq_len, 1] - Normalized speed sequences
            altitude: [batch, seq_len, 1] - Normalized altitude sequences
            gender: [batch, 1] - Binary gender (1.0=male, 0.0=female)
            userId: [batch, 1] - User IDs (integers)
            original_lengths: [batch, 1] - Original sequence lengths (for masking)
        
        Returns:
            heart_rate_pred: [batch, seq_len, 1] - Predicted heart rate in BPM
        """
        batch_size, seq_len, _ = speed.shape
        
        # Embed userId: [batch, 1] → [batch, embedding_dim]
        userId_squeezed = userId.squeeze(1)  # [batch]
        user_emb = self.user_embedding(userId_squeezed)  # [batch, embedding_dim]
        
        # Expand user embedding to match sequence length: [batch, seq_len, embedding_dim]
        user_emb_expanded = user_emb.unsqueeze(1).expand(batch_size, seq_len, self.embedding_dim)
        
        # Expand gender to match sequence length: [batch, 1] → [batch, seq_len, 1]
        gender_expanded = gender.unsqueeze(1).expand(batch_size, seq_len, 1)
        
        # Concatenate all features: [batch, seq_len, 3 + embedding_dim]
        x = torch.cat([speed, altitude, gender_expanded, user_emb_expanded], dim=2)
        
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
    PyTorch Dataset for workout data with user IDs.
    
    Loads preprocessed tensors and returns individual samples including userId.
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
        
        # Get unique user IDs and create mapping
        self.unique_users = torch.unique(self.userId).tolist()
        self.num_users = len(self.unique_users)
        
        # Create user_id → index mapping
        self.user_to_idx = {user_id: idx for idx, user_id in enumerate(self.unique_users)}
        
        # Remap userId to continuous indices [0, num_users-1]
        self.userId_mapped = self._remap_user_ids()
        
    def _remap_user_ids(self):
        """Remap userId to continuous indices for embedding layer."""
        userId_mapped = torch.zeros_like(self.userId)
        for i in range(len(self.userId)):
            original_id = self.userId[i].item()
            userId_mapped[i] = self.user_to_idx[original_id]
        return userId_mapped.long()
    
    def __len__(self):
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx: Sample index
        
        Returns:
            Tuple of (speed, altitude, gender, userId, heart_rate, original_length)
        """
        return (
            self.speed[idx],              # [seq_len, 1]
            self.altitude[idx],            # [seq_len, 1]
            self.gender[idx],              # [1]
            self.userId_mapped[idx],       # [1] - Remapped to [0, num_users-1]
            self.heart_rate[idx],          # [seq_len, 1] - TARGET
            self.original_lengths[idx]     # [1]
        )


# Example usage and model info
if __name__ == '__main__':
    print("="*80)
    print("LSTM WITH USER EMBEDDINGS FOR HEART RATE PREDICTION")
    print("="*80)
    
    # Create model
    num_users = 100  # Example: 100 unique users
    model = HeartRateLSTMWithEmbeddings(
        num_users=num_users,
        embedding_dim=16,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=False
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    print(f"Number of users: {num_users}")
    print(f"Embedding dimension: 16")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 500
    
    dummy_speed = torch.randn(batch_size, seq_len, 1)
    dummy_altitude = torch.randn(batch_size, seq_len, 1)
    dummy_gender = torch.randint(0, 2, (batch_size, 1)).float()
    dummy_userId = torch.randint(0, num_users, (batch_size, 1)).long()
    dummy_lengths = torch.randint(50, seq_len, (batch_size, 1))
    
    print(f"\n" + "="*80)
    print("TEST FORWARD PASS")
    print("="*80)
    print(f"Input shapes:")
    print(f"  speed:       {dummy_speed.shape}")
    print(f"  altitude:    {dummy_altitude.shape}")
    print(f"  gender:      {dummy_gender.shape}")
    print(f"  userId:      {dummy_userId.shape}")
    print(f"  lengths:     {dummy_lengths.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_speed, dummy_altitude, dummy_gender, dummy_userId, dummy_lengths)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print("\n✓ Model initialized successfully!")
    print("\nTo train this model, use: python3 Model/train.py --model lstm_embeddings")
