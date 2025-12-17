"""
Evaluation and Visualization Script for Fine-Tuned Model
"""

import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from finetune.model import HeartRateLSTM
from finetune.dataset import create_dataloaders
from finetune.train_stage1 import MaskedMSELoss


def load_finetuned_model(checkpoint_path, device='cuda'):
    """Load fine-tuned model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Create model
    model = HeartRateLSTM(
        input_size=3,
        hidden_size=128,
        num_layers=4,
        dropout=0.35,
        bidirectional=True
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded fine-tuned model from: {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")
    
    return model, checkpoint


def evaluate_model(model, dataloader, criterion, device):
    """Evaluate model on dataset"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for features, target, mask in dataloader:
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            
            output = model(features)
            loss = criterion(output, target, mask)
            total_loss += loss.item()
            
            # Store for metrics
            all_predictions.append(output.cpu())
            all_targets.append(target.cpu())
            all_masks.append(mask.cpu())
    
    avg_loss = total_loss / len(dataloader)
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    masks = torch.cat(all_masks, dim=0)
    
    # Calculate metrics (only on valid timesteps)
    valid_preds = predictions[masks == 1]
    valid_targets = targets[masks == 1]
    
    mse = ((valid_preds - valid_targets) ** 2).mean().item()
    mae = (valid_preds - valid_targets).abs().mean().item()
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = ((valid_targets - valid_preds) ** 2).sum().item()
    ss_tot = ((valid_targets - valid_targets.mean()) ** 2).sum().item()
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    return metrics, predictions, targets, masks


def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to: {save_path}")
    plt.close()


def plot_sample_predictions(predictions, targets, masks, num_samples=5, save_path=None):
    """Plot sample predictions vs ground truth"""
    fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
    
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        pred = predictions[i, :, 0].numpy()
        target = targets[i, :, 0].numpy()
        mask = masks[i, :, 0].numpy()
        
        # Get valid length
        valid_len = int(mask.sum())
        
        # Plot
        x = np.arange(valid_len)
        axes[i].plot(x, target[:valid_len], 'b-', label='Ground Truth', linewidth=2, alpha=0.7)
        axes[i].plot(x, pred[:valid_len], 'r--', label='Prediction', linewidth=2, alpha=0.7)
        axes[i].fill_between(x, target[:valid_len], pred[:valid_len], alpha=0.2)
        
        # Calculate error for this sample
        mae = np.abs(pred[:valid_len] - target[:valid_len]).mean()
        
        axes[i].set_xlabel('Timestep')
        axes[i].set_ylabel('Heart Rate (bpm)')
        axes[i].set_title(f'Sample {i+1} - MAE: {mae:.2f} bpm - Length: {valid_len}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved sample predictions to: {save_path}")
    plt.close()


def main():
    """Main evaluation function"""
    print("\n" + "="*60)
    print("STAGE 1 FINE-TUNING EVALUATION")
    print("="*60 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint_path = 'checkpoints/stage1/best_model.pt'
    results_dir = Path('results/stage1')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading fine-tuned model...")
    model, checkpoint = load_finetuned_model(checkpoint_path, device)
    
    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        'DATA/Private_runs_processed',
        batch_size=8,
        num_workers=2
    )
    
    # Setup criterion
    criterion = MaskedMSELoss()
    
    # Evaluate on all splits
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\n1. Training Set:")
    train_metrics, train_preds, train_targets, train_masks = evaluate_model(
        model, train_loader, criterion, device
    )
    print(f"   Loss: {train_metrics['loss']:.4f}")
    print(f"   MSE: {train_metrics['mse']:.4f}")
    print(f"   MAE: {train_metrics['mae']:.2f} bpm")
    print(f"   RMSE: {train_metrics['rmse']:.2f} bpm")
    print(f"   R²: {train_metrics['r2']:.4f}")
    
    print("\n2. Validation Set:")
    val_metrics, val_preds, val_targets, val_masks = evaluate_model(
        model, val_loader, criterion, device
    )
    print(f"   Loss: {val_metrics['loss']:.4f}")
    print(f"   MSE: {val_metrics['mse']:.4f}")
    print(f"   MAE: {val_metrics['mae']:.2f} bpm")
    print(f"   RMSE: {val_metrics['rmse']:.2f} bpm")
    print(f"   R²: {val_metrics['r2']:.4f}")
    
    print("\n3. Test Set:")
    test_metrics, test_preds, test_targets, test_masks = evaluate_model(
        model, test_loader, criterion, device
    )
    print(f"   Loss: {test_metrics['loss']:.4f}")
    print(f"   MSE: {test_metrics['mse']:.4f}")
    print(f"   MAE: {test_metrics['mae']:.2f} bpm")
    print(f"   RMSE: {test_metrics['rmse']:.2f} bpm")
    print(f"   R²: {test_metrics['r2']:.4f}")
    
    # Save metrics
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    metrics_path = results_dir / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved metrics to: {metrics_path}")
    
    # Plot training curves
    print("\nGenerating visualizations...")
    history = checkpoint['history']
    plot_training_curves(history, results_dir / 'training_curves.png')
    
    # Plot sample predictions
    plot_sample_predictions(
        test_preds, test_targets, test_masks,
        num_samples=5,
        save_path=results_dir / 'sample_predictions.png'
    )
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved in: {results_dir}")
    print(f"\nKey Metrics (Test Set):")
    print(f"  MAE: {test_metrics['mae']:.2f} bpm")
    print(f"  RMSE: {test_metrics['rmse']:.2f} bpm")
    print(f"  R²: {test_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
