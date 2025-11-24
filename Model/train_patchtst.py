#!/usr/bin/env python3
"""
Training script for PatchTST heart rate prediction model.

This script trains a PatchTST (Patch Time Series Transformer) model
for predicting heart rate from speed and altitude sequences.

Usage:
    # Basic training
    python3 Model/train_patchtst.py --epochs 50 --batch_size 32
    
    # Custom architecture
    python3 Model/train_patchtst.py --d_model 256 --num_layers 6 --patch_length 32
    
    # With custom data directory
    python3 Model/train_patchtst.py --data_dir DATA/processed_hf --epochs 100

Prerequisites:
    1. Run preprocessing: python3 Preprocessing/prepare_sequences_hf.py
    2. Ensure HuggingFace datasets are available in DATA/processed_hf/
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import PatchTST model
from PatchTST_HR import PatchTSTHeartRatePredictor, load_data_hf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train PatchTST model for heart rate prediction')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for optimizer (default: 0.01)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # Model architecture - PatchTST specific
    parser.add_argument('--patch_length', type=int, default=16,
                        help='Length of each patch (default: 16)')
    parser.add_argument('--stride', type=int, default=8,
                        help='Stride between patches (default: 8)')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension (default: 128)')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')
    parser.add_argument('--num_hidden_layers', type=int, default=4,
                        help='Number of transformer layers (default: 4)')
    parser.add_argument('--ffn_dim', type=int, default=256,
                        help='Feed-forward network dimension (default: 256)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability (default: 0.1)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='DATA/processed_hf',
                        help='Directory with HuggingFace preprocessed data (default: DATA/processed_hf)')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints (default: checkpoints)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu (default: auto-detect)')
    
    # Logging
    parser.add_argument('--log_interval', type=int, default=10,
                        help='Log every N batches (default: 10)')
    
    return parser.parse_args()


def train_epoch(model, train_loader, optimizer, device, log_interval=10):
    """
    Train model for one epoch.
    
    Args:
        model: PatchTST model
        train_loader: Training data loader
        optimizer: Optimizer
        device: Device to use
        log_interval: Log every N batches
    
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Training', leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move data to device
        past_values = batch['past_values'].to(device)
        future_values = batch['future_values'].to(device)
        past_observed_mask = batch['past_observed_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            past_values=past_values,
            past_observed_mask=past_observed_mask,
            future_values=future_values
        )
        
        loss = outputs['loss']
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Update metrics
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Log batch-level metrics
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            pbar.set_description(f'Training (loss: {avg_loss:.4f})')
    
    return total_loss / num_batches


def evaluate_epoch(model, val_loader, device):
    """
    Evaluate model on validation set.
    
    Args:
        model: PatchTST model
        val_loader: Validation data loader
        device: Device to use
    
    Returns:
        Dictionary with evaluation metrics (loss, MAE)
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validating', leave=False)
        for batch in pbar:
            # Move data to device
            past_values = batch['past_values'].to(device)
            future_values = batch['future_values'].to(device)
            past_observed_mask = batch['past_observed_mask'].to(device)
            
            # Forward pass
            outputs = model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                future_values=future_values
            )
            
            loss = outputs['loss']
            predictions = outputs['prediction']
            
            # Compute MAE
            mae = torch.mean(torch.abs(predictions - future_values)).item()
            
            # Update metrics
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item(), 'MAE': mae})
    
    return {
        'loss': total_loss / num_batches,
        'mae': total_mae / num_batches
    }


def train(model, train_loader, val_loader, optimizer, scheduler, args):
    """
    Full training loop with early stopping.
    
    Args:
        model: PatchTST model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        args: Command-line arguments
    
    Returns:
        history: Dictionary with training history
        best_model_state: State dict of best model
    """
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device, args.log_interval)
        
        # Validate
        val_metrics = evaluate_epoch(model, val_loader, args.device)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_mae'].append(val_metrics['mae'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss:   {val_metrics['loss']:.4f}")
        print(f"Val MAE:    {val_metrics['mae']:.2f} BPM")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step(val_metrics['loss'])
        
        # Early stopping
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✓ New best model! (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best Val Loss: {best_val_loss:.4f}")
    
    return history, best_model_state


def evaluate_test(model, test_loader, device):
    """
    Evaluate model on test set.
    
    Args:
        model: PatchTST model
        test_loader: Test data loader
        device: Device to use
    
    Returns:
        Dictionary with test metrics
    """
    print("\n" + "="*80)
    print("EVALUATING ON TEST SET")
    print("="*80)
    
    model.eval()
    total_loss = 0
    total_mae = 0
    total_rmse = 0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for batch in pbar:
            # Move data to device
            past_values = batch['past_values'].to(device)
            future_values = batch['future_values'].to(device)
            past_observed_mask = batch['past_observed_mask'].to(device)
            
            # Forward pass
            outputs = model(
                past_values=past_values,
                past_observed_mask=past_observed_mask,
                future_values=future_values
            )
            
            loss = outputs['loss']
            predictions = outputs['prediction']
            
            # Compute metrics
            mae = torch.mean(torch.abs(predictions - future_values)).item()
            rmse = torch.sqrt(torch.mean((predictions - future_values) ** 2)).item()
            
            # Update metrics
            total_loss += loss.item()
            total_mae += mae
            total_rmse += rmse
            num_batches += 1
            
            # Store for later analysis
            all_predictions.append(predictions.cpu())
            all_targets.append(future_values.cpu())
            
            # Update progress bar
            pbar.set_postfix({'MAE': mae, 'RMSE': rmse})
    
    # Compute final metrics
    metrics = {
        'loss': total_loss / num_batches,
        'mae': total_mae / num_batches,
        'rmse': total_rmse / num_batches
    }
    
    print(f"\nTest Results:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  MAE:  {metrics['mae']:.2f} BPM")
    print(f"  RMSE: {metrics['rmse']:.2f} BPM")
    
    return metrics, all_predictions, all_targets


def plot_training_curves(history, save_path):
    """
    Plot training curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss curve
    axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss (MSE)')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE curve
    axes[0, 1].plot(history['val_mae'], label='Val MAE', linewidth=2, color='orange')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE (BPM)')
    axes[0, 1].set_title('Validation MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'], linewidth=2, color='green')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Loss comparison (log scale)
    axes[1, 1].semilogy(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1, 1].semilogy(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss (MSE, log scale)')
    axes[1, 1].set_title('Loss Comparison (Log Scale)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Training curves saved to: {save_path}")
    plt.close()


def save_checkpoint(model, optimizer, history, test_metrics, args):
    """
    Save model checkpoint and training information.
    
    Args:
        model: Trained model
        optimizer: Optimizer
        history: Training history
        test_metrics: Test set metrics
        args: Command-line arguments
    """
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate checkpoint name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_name = f"patchtst_bs{args.batch_size}_lr{args.lr}_e{args.epochs}_" \
                     f"d{args.d_model}_l{args.num_hidden_layers}_p{args.patch_length}"
    
    # Save model
    model_path = checkpoint_dir / f"{checkpoint_name}_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model.config.to_dict(),
        'args': vars(args)
    }, model_path)
    
    # Save history
    history_path = checkpoint_dir / f"{checkpoint_name}_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save metrics
    metrics_path = checkpoint_dir / f"{checkpoint_name}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'test_metrics': test_metrics,
            'best_val_loss': min(history['val_loss']),
            'best_val_mae': min(history['val_mae']),
            'final_epoch': len(history['train_loss'])
        }, f, indent=2)
    
    # Plot training curves
    plot_path = checkpoint_dir / f"{checkpoint_name}_training_curves.png"
    plot_training_curves(history, plot_path)
    
    print(f"\n✓ Checkpoint saved:")
    print(f"  Model:    {model_path}")
    print(f"  History:  {history_path}")
    print(f"  Metrics:  {metrics_path}")
    print(f"  Plot:     {plot_path}")


def main():
    """Main training pipeline."""
    args = parse_args()
    
    print("="*80)
    print("PATCHTST HEART RATE PREDICTION - TRAINING")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Model: PatchTST")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {args.device}")
    print(f"\nArchitecture:")
    print(f"  Patch length: {args.patch_length}")
    print(f"  Stride: {args.stride}")
    print(f"  Model dimension: {args.d_model}")
    print(f"  Attention heads: {args.num_attention_heads}")
    print(f"  Hidden layers: {args.num_hidden_layers}")
    print(f"  FFN dimension: {args.ffn_dim}")
    print(f"  Dropout: {args.dropout}")
    
    # Load data
    train_loader, val_loader, test_loader, metadata = load_data_hf(
        args.data_dir, 
        args.batch_size
    )
    
    print(f"\nDataset info:")
    print(f"  Train samples: {metadata['num_train']}")
    print(f"  Val samples: {metadata['num_val']}")
    print(f"  Test samples: {metadata['num_test']}")
    print(f"  Sequence length: {metadata['sequence_length']}")
    
    # Create model
    print("\nInitializing PatchTST model...")
    model = PatchTSTHeartRatePredictor(
        num_input_channels=3,  # speed, altitude, gender
        context_length=metadata['sequence_length'],
        prediction_length=metadata['sequence_length'],
        patch_length=args.patch_length,
        stride=args.stride,
        d_model=args.d_model,
        num_attention_heads=args.num_attention_heads,
        num_hidden_layers=args.num_hidden_layers,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout
    )
    
    model = model.to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Model initialized")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Train
    history, best_model_state = train(
        model, train_loader, val_loader, optimizer, scheduler, args
    )
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set
    test_metrics, predictions, targets = evaluate_test(model, test_loader, args.device)
    
    # Save checkpoint
    save_checkpoint(model, optimizer, history, test_metrics, args)
    
    print("\n" + "="*80)
    print("TRAINING PIPELINE COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Results:")
    print(f"  MAE:  {test_metrics['mae']:.2f} BPM")
    print(f"  RMSE: {test_metrics['rmse']:.2f} BPM")
    
    if test_metrics['mae'] < 5:
        print("\n🎉 EXCELLENT! MAE < 5 BPM (strong performance)")
    elif test_metrics['mae'] < 10:
        print("\n✓ GOOD! MAE < 10 BPM (acceptable performance)")
    else:
        print("\n⚠ MAE > 10 BPM - Consider tuning hyperparameters")


if __name__ == '__main__':
    main()
