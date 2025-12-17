"""
Configuration for Multi-Stage Fine-Tuning
"""

# Stage 1: Freeze layer 0 - Train only layer 1 + FC
STAGE1_CONFIG = {
    # Model
    'freeze_layers': [0],  # Freeze first LSTM layer (only 2 layers in pretrained model)
    
    # Training
    'learning_rate': 5e-4,       # Conservative LR
    'batch_size': 32,             # Small batch for small dataset
    'epochs': 50,                # Maximum epochs
    'patience': 10,               # Early stopping patience
    
    # Regularization
    'weight_decay': 1e-4,        # L2 regularization
    'gradient_clip': 1.0,        # Gradient clipping
    
    # Scheduler
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,     # LR reduction factor
    'scheduler_patience': 3,     # Scheduler patience
    'scheduler_min_lr': 1e-6,    # Minimum LR
    
    # Data
    'num_workers': 2,            # DataLoader workers
    'use_mask': True,            # Use masking for padded sequences
    
    # Device
    'device': 'cuda',            # 'cuda' or 'cpu'
    
    # Logging
    'log_interval': 5,           # Print every N batches
    'save_interval': 1,          # Save checkpoint every N epochs
}

# Stage 2: Unfreeze all - Train all layers
STAGE2_CONFIG = {
    # Model
    'freeze_layers': [],     # Unfreeze all layers
    
    # Training
    'learning_rate': 1e-4,       # Lower LR for more layers
    'batch_size': 32,            # Same batch size
    'epochs': 50,                # Maximum epochs
    'patience': 10,              # Early stopping patience
    
    # Regularization
    'weight_decay': 1e-4,        # L2 regularization
    'gradient_clip': 1.0,        # Gradient clipping
    
    # Scheduler
    'scheduler': 'ReduceLROnPlateau',
    'scheduler_factor': 0.5,     # LR reduction factor
    'scheduler_patience': 3,     # Scheduler patience
    'scheduler_min_lr': 1e-7,    # Minimum LR (lower than Stage 1)
    
    # Data
    'num_workers': 2,            # DataLoader workers
    'use_mask': True,            # Use masking for padded sequences
    
    # Device
    'device': 'cuda',            # 'cuda' or 'cpu'
    
    # Logging
    'log_interval': 5,           # Print every N batches
    'save_interval': 1,          # Save checkpoint every N epochs
}

# Paths - Stage 1
STAGE1_PATHS = {
    'pretrained_model': 'experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_best.pt',
    'data_dir': 'DATA/apple_watch_processed',  # Use apple_watch_processed for finetuning
    'checkpoint_dir': 'checkpoints/stage1',
    'results_dir': 'results/stage1',
    'logs_dir': 'logs',
}

# Paths - Stage 2 (loads from Stage 1 best model)
STAGE2_PATHS = {
    'pretrained_model': None,  # Will be set dynamically or by user
    'stage1_checkpoint_dir': 'checkpoints/stage1',  # Look for Stage 1 best model here
    'data_dir': 'DATA/apple_watch_processed',  # Use apple_watch_processed for finetuning
    'checkpoint_dir': 'checkpoints/stage2',
    'results_dir': 'results/stage2',
    'logs_dir': 'logs',
}

# Model architecture (must match pretrained)
MODEL_CONFIG = {
    'input_size': 3,
    'hidden_size': 64,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': False,
}


def get_stage1_config():
    """Get complete Stage 1 configuration"""
    config = {
        **STAGE1_CONFIG,
        **STAGE1_PATHS,
        **MODEL_CONFIG,
    }
    return config


def get_stage2_config(stage1_checkpoint=None):
    """Get complete Stage 2 configuration
    
    Args:
        stage1_checkpoint: Path to Stage 1 best model. If None, will look for latest best model.
    """
    from pathlib import Path
    
    config = {
        **STAGE2_CONFIG,
        **STAGE2_PATHS,
        **MODEL_CONFIG,
    }
    
    # Set Stage 1 checkpoint
    if stage1_checkpoint:
        config['pretrained_model'] = stage1_checkpoint
    else:
        # Find latest best model from Stage 1
        stage1_dir = Path(config['stage1_checkpoint_dir'])
        # Try with timestamp pattern first, then fallback to simple best_model.pt
        best_models = sorted(stage1_dir.glob('best_model_*.pt'), reverse=True)
        if best_models:
            config['pretrained_model'] = str(best_models[0])
        elif (stage1_dir / 'best_model.pt').exists():
            config['pretrained_model'] = str(stage1_dir / 'best_model.pt')
        else:
            raise FileNotFoundError(
                f"No Stage 1 best model found in {stage1_dir}. "
                "Please run Stage 1 training first or specify stage1_checkpoint."
            )
    
    return config


def print_config(config, stage=1):
    """Pretty print configuration"""
    print("\n" + "="*60)
    print(f"STAGE {stage} FINE-TUNING CONFIGURATION")
    print("="*60)
    
    print("\n=== Model ===")
    print(f"Input size: {config['input_size']}")
    print(f"Hidden size: {config['hidden_size']}")
    print(f"Num layers: {config['num_layers']}")
    print(f"Dropout: {config['dropout']}")
    print(f"Bidirectional: {config['bidirectional']}")
    print(f"Frozen layers: {config['freeze_layers']}")
    
    print("\n=== Training ===")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"Early stopping patience: {config['patience']}")
    print(f"Weight decay: {config['weight_decay']}")
    print(f"Gradient clip: {config['gradient_clip']}")
    
    print("\n=== Scheduler ===")
    print(f"Type: {config['scheduler']}")
    print(f"Factor: {config['scheduler_factor']}")
    print(f"Patience: {config['scheduler_patience']}")
    print(f"Min LR: {config['scheduler_min_lr']}")
    
    print("\n=== Paths ===")
    print(f"Pretrained model: {config['pretrained_model']}")
    print(f"Data directory: {config['data_dir']}")
    print(f"Checkpoint dir: {config['checkpoint_dir']}")
    print(f"Results dir: {config['results_dir']}")
    print(f"Logs dir: {config['logs_dir']}")
    
    print("\n=== Device ===")
    print(f"Device: {config['device']}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    config = get_stage1_config()
    print_config(config)
