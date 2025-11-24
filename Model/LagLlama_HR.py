#!/usr/bin/env python3
"""
Lag-Llama model adapter for heart rate prediction.

Architecture:
    Input: Concatenate[speed, altitude, gender, user_embedding] as multivariate time series
    Pretrained Model: Lag-Llama (time-series foundation model)
    Output: Heart rate sequence [batch, 500, 1]

This uses transfer learning from Lag-Llama's pretraining on diverse time-series data.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class LagLlamaHRPredictor(nn.Module):
    """
    Lag-Llama-inspired model for heart rate prediction.
    
    Architecture:
        1. User embeddings (optional)
        2. Combine speed + altitude + gender + user_emb → multivariate input
        3. Transformer encoder layers (Lag-Llama style)
        4. Projection layer → heart rate [batch, seq_len, 1]
    
    Note: This is a Transformer-based architecture inspired by Lag-Llama.
          For actual pretrained Lag-Llama, you would need GluonTS integration.
    """
    
    def __init__(
        self,
        context_length=500,      # Input sequence length
        prediction_length=500,   # Output sequence length (same for our task)
        num_users=None,          # Number of users for embeddings
        embedding_dim=16,        # User embedding dimension
        d_model=128,             # Transformer hidden dimension
        nhead=8,                 # Number of attention heads
        num_layers=4,            # Number of transformer layers
        dim_feedforward=512,     # Feedforward dimension
        dropout=0.1,             # Dropout probability
        device='cuda'
    ):
        """
        Initialize Lag-Llama-style model for HR prediction.
        
        Args:
            context_length: Length of input sequence (default: 500)
            prediction_length: Length of output sequence (default: 500)
            num_users: Number of unique users (optional, for embeddings)
            embedding_dim: User embedding dimension (default: 16)
            d_model: Transformer hidden dimension (default: 128)
            nhead: Number of attention heads (default: 8)
            num_layers: Number of transformer layers (default: 4)
            dim_feedforward: Feedforward network dimension (default: 512)
            dropout: Dropout probability (default: 0.1)
            device: Device to use (cuda or cpu)
        """
        super(LagLlamaHRPredictor, self).__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.device = device
        
        # User embedding (optional)
        if num_users is not None:
            self.user_embedding = nn.Embedding(
                num_embeddings=num_users,
                embedding_dim=embedding_dim
            )
            input_dim = 3 + embedding_dim  # speed + altitude + gender + user_emb
        else:
            self.user_embedding = None
            input_dim = 3  # speed + altitude + gender
        
        # Input projection: [batch, seq_len, input_dim] → [batch, seq_len, d_model]
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len=context_length)
        
        # Transformer encoder layers (Lag-Llama style)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output projection: [batch, seq_len, d_model] → [batch, seq_len, 1]
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, speed, altitude, gender, userId=None, original_lengths=None):
        """
        Forward pass.
        
        Args:
            speed: [batch, seq_len, 1] - Normalized speed sequences
            altitude: [batch, seq_len, 1] - Normalized altitude sequences
            gender: [batch, 1] - Binary gender (1.0=male, 0.0=female)
            userId: [batch, 1] - User IDs (optional, for embeddings)
            original_lengths: [batch, 1] - Original sequence lengths (for masking)
        
        Returns:
            heart_rate_pred: [batch, seq_len, 1] - Predicted heart rate in BPM
        """
        batch_size, seq_len, _ = speed.shape
        
        # Expand gender to match sequence length
        gender_expanded = gender.unsqueeze(1).expand(batch_size, seq_len, 1)
        
        # Concatenate features: [batch, seq_len, 3]
        x = torch.cat([speed, altitude, gender_expanded], dim=2)
        
        # Add user embeddings if available
        if self.user_embedding is not None and userId is not None:
            userId_squeezed = userId.squeeze(1)  # [batch]
            user_emb = self.user_embedding(userId_squeezed)  # [batch, embedding_dim]
            user_emb_expanded = user_emb.unsqueeze(1).expand(batch_size, seq_len, self.embedding_dim)
            x = torch.cat([x, user_emb_expanded], dim=2)  # [batch, seq_len, 3+embedding_dim]
        
        # Project to transformer dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Create attention mask for padded sequences (optional)
        src_mask = None
        if original_lengths is not None:
            src_mask = self._create_mask(original_lengths, seq_len).to(x.device)
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)  # [batch, seq_len, d_model]
        
        # Project to heart rate
        heart_rate_pred = self.output_projection(x)  # [batch, seq_len, 1]
        
        return heart_rate_pred
    
    def _create_mask(self, lengths, max_len):
        """
        Create attention mask for padded sequences.
        
        Args:
            lengths: [batch, 1] - Original sequence lengths
            max_len: Maximum sequence length
        
        Returns:
            mask: [batch, max_len] - Boolean mask (True for padded positions)
        """
        batch_size = lengths.shape[0]
        mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths
        return mask
    
    def count_parameters(self):
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Adds sinusoidal position information to embeddings.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Add positional encoding to input.
        
        Args:
            x: [batch, seq_len, d_model]
        
        Returns:
            x + positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WorkoutDataset(Dataset):
    """
    PyTorch Dataset for workout data (compatible with Lag-Llama model).
    
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
    print("LAG-LLAMA INSPIRED MODEL FOR HEART RATE PREDICTION")
    print("="*80)
    print("\nNote: This is a Transformer-based architecture inspired by Lag-Llama.")
    print("For full pretrained Lag-Llama integration, use GluonTS library.")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    num_users = 100  # Example: 100 unique users
    model = LagLlamaHRPredictor(
        context_length=500,
        prediction_length=500,
        num_users=num_users,
        embedding_dim=16,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        device=device
    )
    
    print(f"\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {model.count_parameters():,}")
    print(f"Number of users: {num_users}")
    print(f"Embedding dimension: 16")
    print(f"Transformer dimension: 128")
    print(f"Device: {device}")
    
    # Test forward pass with dummy data
    batch_size = 4
    seq_len = 500
    
    dummy_speed = torch.randn(batch_size, seq_len, 1).to(device)
    dummy_altitude = torch.randn(batch_size, seq_len, 1).to(device)
    dummy_gender = torch.randint(0, 2, (batch_size, 1)).float().to(device)
    dummy_userId = torch.randint(0, num_users, (batch_size, 1)).long().to(device)
    dummy_lengths = torch.randint(50, seq_len, (batch_size, 1)).to(device)
    
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
    model = model.to(device)
    with torch.no_grad():
        output = model(dummy_speed, dummy_altitude, dummy_gender, dummy_userId, dummy_lengths)
    
    print(f"\nOutput shape: {output.shape}")
    print(f"Output range: [{output.min().item():.2f}, {output.max().item():.2f}]")
    
    print("\n✓ Model initialized successfully!")
    print("\nTo train this model, use: python3 Model/train.py --model lag_llama")
