#!/usr/bin/env python3
"""
Generic training script for heart rate prediction models.

Supports training:
1. Basic LSTM
2. LSTM with user embeddings
3. GRU (Robust variant)

Usage:
    python3 Model/train.py --model lstm --epochs 100 --batch_size 32
    python3 Model/train.py --model lstm_embeddings --epochs 100 --batch_size 32
    python3 Model/train.py --model gru --epochs 100 --batch_size 32
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Import model classes
from LSTM import HeartRateLSTM, WorkoutDataset as BasicDataset
from LSTM_with_embeddings import HeartRateLSTMWithEmbeddings, WorkoutDataset as EmbeddingDataset
# Importation du nouveau modèle GRU
from GRU import HeartRateGRU


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train RNN models (LSTM/GRU) for heart rate prediction')
    
    # Model selection - AJOUT DE 'gru'
    parser.add_argument('--model', type=str, default='gru', choices=['lstm', 'lstm_embeddings', 'gru'],
                        help='Model type: lstm, lstm_embeddings, or gru')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=15,
                        help='Number of training epochs (default: 15)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate (default: 0.0005)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='Hidden size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability (default: 0.2)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Use bidirectional RNN')
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='User embedding dimension (for lstm_embeddings, default: 16)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='DATA',
                        help='Directory with preprocessed data (default: DATA/processed)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu (default: auto-detect)')
    
    return parser.parse_args()


def load_data(data_dir, model_type, batch_size):
    """
    Load preprocessed data and create DataLoaders.
    """
    print(f"\nLoading data from {data_dir}...")
    
    # Load tensors
    train_data = torch.load(f'{data_dir}/train.pt', weights_only=False)
    val_data = torch.load(f'{data_dir}/val.pt', weights_only=False)
    test_data = torch.load(f'{data_dir}/test.pt', weights_only=False)
    
    # Load metadata
    with open(f'{data_dir}/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Create datasets
    # Le GRU utilise le même format de données que le LSTM basique
    if model_type in ['lstm', 'gru']:
        train_dataset = BasicDataset(train_data)
        val_dataset = BasicDataset(val_data)
        test_dataset = BasicDataset(test_data)
    else:  # lstm_embeddings
        train_dataset = EmbeddingDataset(train_data)
        val_dataset = EmbeddingDataset(val_data)
        test_dataset = EmbeddingDataset(test_data)
        
        # Add num_users to metadata
        metadata['num_users'] = train_dataset.num_users
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"✓ Data loaded:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    if model_type == 'lstm_embeddings':
        print(f"  Unique users: {metadata['num_users']}")
    
    return train_loader, val_loader, test_loader, metadata


def create_model(model_type, metadata, args):
    """
    Create model based on model_type.
    """
    print(f"\nCreating {model_type} model...")
    
    if model_type == 'lstm':
        model = HeartRateLSTM(
            input_size=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    elif model_type == 'gru':
        # Instanciation du modèle GRU
        model = HeartRateGRU(
            input_size=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    else:  # lstm_embeddings
        model = HeartRateLSTMWithEmbeddings(
            num_users=metadata['num_users'],
            embedding_dim=args.embedding_dim,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional
        )
    
    print(f"✓ Model created:")
    print(f"  Total parameters: {model.count_parameters():,}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device, model_type):
    """
    Train model for one epoch.
    """
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    
    for batch in dataloader:
        # Le GRU et le LSTM partagent la même signature d'entrée
        if model_type in ['lstm', 'gru']:
            speed, altitude, gender, heart_rate, original_lengths = batch
            speed = speed.to(device)
            altitude = altitude.to(device)
            gender = gender.to(device)
            heart_rate = heart_rate.to(device)
            
            # Forward pass
            predictions = model(speed, altitude, gender, original_lengths)
        else:  # lstm_embeddings
            speed, altitude, gender, userId, heart_rate, original_lengths = batch
            speed = speed.to(device)
            altitude = altitude.to(device)
            gender = gender.to(device)
            userId = userId.to(device)
            heart_rate = heart_rate.to(device)
            
            # Forward pass
            predictions = model(speed, altitude, gender, userId, original_lengths)
        
        # Compute loss
        loss = criterion(predictions, heart_rate)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Compute MAE
        mae = torch.abs(predictions - heart_rate).mean()
        
        total_loss += loss.item()
        total_mae += mae.item()
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    
    return avg_loss, avg_mae


def validate(model, dataloader, criterion, device, model_type):
    """
    Validate model.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if model_type in ['lstm', 'gru']:
                speed, altitude, gender, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                heart_rate = heart_rate.to(device)
                
                # Forward pass
                predictions = model(speed, altitude, gender, original_lengths)
            else:  # lstm_embeddings
                speed, altitude, gender, userId, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                userId = userId.to(device)
                heart_rate = heart_rate.to(device)
                
                # Forward pass
                predictions = model(speed, altitude, gender, userId, original_lengths)
            
            # Compute loss
            loss = criterion(predictions, heart_rate)
            
            # Compute MAE
            mae = torch.abs(predictions - heart_rate).mean()
            
            total_loss += loss.item()
            total_mae += mae.item()
            n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_mae = total_mae / n_batches
    
    return avg_loss, avg_mae


def train(model, train_loader, val_loader, optimizer, criterion, scheduler, args):
    """
    Full training loop with early stopping.
    """
    device = args.device
    model = model.to(device)
    
    print(f"\n{'='*80}")
    print(f"TRAINING {args.model.upper()} MODEL")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Early stopping patience: {args.patience}")
    print(f"{'='*80}\n")
    
    history = {
        'train_loss': [],
        'train_mae': [],
        'val_loss': [],
        'val_mae': [],
        'lr': []
    }
    
    best_val_mae = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_mae = train_epoch(
            model, train_loader, criterion, optimizer, device, args.model
        )
        
        # Validate
        val_loss, val_mae = validate(
            model, val_loader, criterion, device, args.model
        )
        
        # Update learning rate
        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_mae'].append(train_mae)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['lr'].append(current_lr)
        
        # Print progress
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] | "
              f"Train Loss: {train_loss:.4f} | Train MAE: {train_mae:.2f} BPM | "
              f"Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.2f} BPM | "
              f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  ✓ New best model! (Val MAE: {val_mae:.2f} BPM)")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                print(f"  Best Val MAE: {best_val_mae:.2f} BPM")
                break
    
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best Val MAE: {best_val_mae:.2f} BPM")
    
    return history, best_model_state


def evaluate(model, test_loader, device, model_type):
    """
    Evaluate model on test set.
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING ON TEST SET")
    print(f"{'='*80}")
    
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            if model_type in ['lstm', 'gru']:
                speed, altitude, gender, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                heart_rate = heart_rate.to(device)
                
                predictions = model(speed, altitude, gender, original_lengths)
            else:  # lstm_embeddings
                speed, altitude, gender, userId, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                userId = userId.to(device)
                heart_rate = heart_rate.to(device)
                
                predictions = model(speed, altitude, gender, userId, original_lengths)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(heart_rate.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    mse = nn.MSELoss()(all_predictions, all_targets).item()
    mae = torch.abs(all_predictions - all_targets).mean().item()
    rmse = np.sqrt(mse)
    
    # Compute R² score
    ss_res = torch.sum((all_targets - all_predictions) ** 2).item()
    ss_tot = torch.sum((all_targets - all_targets.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"Test Metrics:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MAE:  {mae:.2f} BPM")
    print(f"  RMSE: {rmse:.2f} BPM")
    print(f"  R²:   {r2:.4f}")
    
    return metrics


def save_checkpoint(model, optimizer, history, metrics, args):
    """
    Save model checkpoint.
    """
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / f"{args.model}_best.pt"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'metrics': metrics,
        'args': vars(args)
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"\n✓ Checkpoint saved to: {checkpoint_path}")


def plot_training_curves(history, args):
    """
    Plot and save training curves.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot loss
    axes[0].plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontweight='bold')
    axes[0].set_ylabel('Loss (MSE)', fontweight='bold')
    axes[0].set_title(f'{args.model.upper()} - Training and Validation Loss', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(epochs, history['train_mae'], label='Train MAE', linewidth=2)
    axes[1].plot(epochs, history['val_mae'], label='Val MAE', linewidth=2)
    axes[1].axhline(y=5, color='g', linestyle='--', label='Target: 5 BPM', alpha=0.7)
    axes[1].axhline(y=10, color='orange', linestyle='--', label='Acceptable: 10 BPM', alpha=0.7)
    axes[1].set_xlabel('Epoch', fontweight='bold')
    axes[1].set_ylabel('MAE (BPM)', fontweight='bold')
    axes[1].set_title(f'{args.model.upper()} - Training and Validation MAE', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(args.checkpoint_dir) / f"{args.model}_training_curves.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Training curves saved to: {plot_path}")
    plt.close()


def main():
    """Main training pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Handle "auto" device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # GPU optimizations
    if args.device == "cuda":
        torch.cuda.empty_cache()
        # Disable cuDNN for better LSTM/GRU compatibility on some drivers
        torch.backends.cudnn.enabled = False
        
        print(f"\nGPU Configuration:")
        print(f"  Device: {torch.cuda.get_device_name(0)}")
    
    print(f"\n{'='*80}")
    print(f"HEART RATE PREDICTION - MODEL TRAINING")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Device: {args.device}")
    
    # Load data
    train_loader, val_loader, test_loader, metadata = load_data(
        args.data_dir, args.model, args.batch_size
    )
    
    # Create model
    model = create_model(args.model, metadata, args)
    
    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Train
    history, best_model_state = train(
        model, train_loader, val_loader, optimizer, criterion, scheduler, args
    )
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    metrics = evaluate(model, test_loader, args.device, args.model)
    
    # Save checkpoint
    save_checkpoint(model, optimizer, history, metrics, args)
    
    # Plot training curves
    plot_training_curves(history, args)
    
    print(f"\n{'='*80}")
    print(f"✓ TRAINING PIPELINE COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()