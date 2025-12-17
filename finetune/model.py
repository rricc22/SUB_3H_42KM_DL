"""
LSTM Model Architecture for Heart Rate Prediction
Reconstructed from pre-trained checkpoint
"""

import torch
import torch.nn as nn


class HeartRateLSTM(nn.Module):
    """
    Bidirectional LSTM for heart rate prediction from workout sequences.
    
    Architecture:
    - 4-layer bidirectional LSTM
    - Input: (batch, seq_len, 3) - [speed, altitude, heart_rate]
    - Hidden size: 128 per direction (256 total)
    - Output: (batch, seq_len, 1) - predicted heart rate
    """
    
    def __init__(
        self,
        input_size: int = 3,
        hidden_size: int = 128,
        num_layers: int = 4,
        dropout: float = 0.35,
        bidirectional: bool = True
    ):
        super(HeartRateLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size * self.num_directions, 1)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: (batch, seq_len, input_size)
        
        Returns:
            output: (batch, seq_len, 1)
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size * num_directions)
        
        # Fully connected layer
        output = self.fc(lstm_out)
        # output: (batch, seq_len, 1)
        
        return output
    
    def freeze_layers(self, layer_indices):
        """
        Freeze specific LSTM layers
        
        Args:
            layer_indices: List of layer indices to freeze (0-indexed)
        """
        for idx in layer_indices:
            if idx >= self.num_layers:
                print(f"Warning: Layer {idx} doesn't exist (max: {self.num_layers-1})")
                continue
            
            # Freeze forward direction
            for param_name in [f'weight_ih_l{idx}', f'weight_hh_l{idx}', 
                              f'bias_ih_l{idx}', f'bias_hh_l{idx}']:
                param = getattr(self.lstm, param_name)
                param.requires_grad = False
            
            # Freeze reverse direction if bidirectional
            if self.bidirectional:
                for param_name in [f'weight_ih_l{idx}_reverse', f'weight_hh_l{idx}_reverse',
                                  f'bias_ih_l{idx}_reverse', f'bias_hh_l{idx}_reverse']:
                    param = getattr(self.lstm, param_name)
                    param.requires_grad = False
        
        print(f"Frozen LSTM layers: {layer_indices}")
    
    def unfreeze_all_layers(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        print("All layers unfrozen")
    
    def count_parameters(self):
        """Count trainable and total parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total
    
    def print_parameter_status(self):
        """Print which parameters are trainable"""
        print("\n=== Parameter Status ===")
        for name, param in self.named_parameters():
            status = "✓ Trainable" if param.requires_grad else "✗ Frozen"
            print(f"{name:40s} {str(param.shape):30s} {status}")
        
        trainable, total = self.count_parameters()
        print(f"\nTrainable: {trainable:,} / Total: {total:,} ({100*trainable/total:.1f}%)")


def load_pretrained_model(checkpoint_path: str, device: str = 'cuda'):
    """
    Load pre-trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model to
    
    Returns:
        model: Loaded HeartRateLSTM model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters from config (or args for backward compatibility)
    config = checkpoint.get('config', checkpoint.get('args', {}))
    
    # Create model
    model = HeartRateLSTM(
        input_size=config.get('input_size', 3),  # speed, altitude, heart_rate
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 4),
        dropout=config.get('dropout', 0.35),
        bidirectional=config.get('bidirectional', True)
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Loaded pre-trained model from: {checkpoint_path}")
    print(f"Original training epochs: {len(checkpoint.get('history', {}).get('train_loss', []))}")
    print(f"Original val loss: {checkpoint.get('history', {}).get('val_loss', [])[-1]:.4f}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Test model creation
    model = HeartRateLSTM()
    print(model)
    
    trainable, total = model.count_parameters()
    print(f"\nTotal parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    # Test forward pass
    batch_size = 8
    seq_len = 500
    x = torch.randn(batch_size, seq_len, 3)
    output = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test freezing
    print("\n=== Testing Layer Freezing ===")
    model.freeze_layers([0, 1, 2])
    trainable, total = model.count_parameters()
    print(f"After freezing layers 0,1,2: {trainable:,} trainable / {total:,} total")
