#!/usr/bin/env python3
"""
Evaluate finetuned models on test set
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path
import json
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from finetune.model import HeartRateLSTM
from finetune.dataset import create_dataloaders


class MaskedMSELoss(nn.Module):
    """MSE Loss with masking for padded sequences"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target, mask):
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()


def evaluate_finetuned_model(checkpoint_path: str, stage: str):
    """Evaluate a finetuned model"""
    print(f"\n{'='*80}")
    print(f"EVALUATING {stage.upper()} MODEL")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint['config']
    
    # Create model
    model = HeartRateLSTM(
        input_size=config['input_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional']
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"\nModel architecture:")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Num layers: {config['num_layers']}")
    print(f"  Bidirectional: {config['bidirectional']}")
    print(f"  Dropout: {config['dropout']}")
    
    print(f"\nTraining info:")
    print(f"  Trained epochs: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    
    # Create dataloaders
    print(f"\nLoading data from: {config['data_dir']}")
    _, _, test_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_mask=config['use_mask']
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    criterion = MaskedMSELoss()
    
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for features, target, mask in test_loader:
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            
            output = model(features)
            loss = criterion(output, target, mask)
            test_loss += loss.item()
            
            # Store for metrics calculation
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
            all_masks.append(mask.cpu().numpy())
    
    test_loss /= len(test_loader)
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    # Apply mask for metrics calculation
    valid_mask = masks.squeeze(-1).astype(bool)
    valid_predictions = predictions.squeeze(-1)[valid_mask]
    valid_targets = targets.squeeze(-1)[valid_mask]
    
    # Calculate metrics
    mae = np.mean(np.abs(valid_predictions - valid_targets))
    rmse = np.sqrt(np.mean((valid_predictions - valid_targets) ** 2))
    
    # R² score
    ss_res = np.sum((valid_targets - valid_predictions) ** 2)
    ss_tot = np.sum((valid_targets - np.mean(valid_targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n{'='*80}")
    print(f"TEST SET RESULTS")
    print(f"{'='*80}")
    print(f"Test Loss (MSE): {test_loss:.4f}")
    print(f"MAE: {mae:.4f} BPM")
    print(f"RMSE: {rmse:.4f} BPM")
    print(f"R²: {r2:.4f}")
    print(f"Valid samples: {len(valid_predictions):,}")
    
    # Save metrics
    metrics = {
        'stage': stage,
        'checkpoint': str(checkpoint_path),
        'test_loss': float(test_loss),
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'epoch': int(checkpoint['epoch']),
        'best_val_loss': float(checkpoint['best_val_loss']),
        'config': {
            'hidden_size': config['hidden_size'],
            'num_layers': config['num_layers'],
            'batch_size': config['batch_size'],
            'learning_rate': config['learning_rate'],
        }
    }
    
    # Save to results directory
    results_dir = Path(config['results_dir'])
    metrics_file = results_dir / 'test_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n✓ Saved metrics to: {metrics_file}")
    
    return metrics


def main():
    """Evaluate all finetuned models"""
    print("="*80)
    print("EVALUATING FINETUNED MODELS")
    print("="*80)
    
    all_metrics = []
    
    # Check Stage 1
    stage1_checkpoint = Path('checkpoints/stage1/best_model.pt')
    if stage1_checkpoint.exists():
        print(f"\nFound Stage 1 checkpoint: {stage1_checkpoint}")
        metrics = evaluate_finetuned_model(str(stage1_checkpoint), 'stage1')
        all_metrics.append(metrics)
    else:
        print(f"\n⚠ Stage 1 checkpoint not found: {stage1_checkpoint}")
    
    # Check Stage 2
    stage2_checkpoint = Path('checkpoints/stage2/best_model.pt')
    if not stage2_checkpoint.exists():
        # Try timestamped filenames
        stage2_dir = Path('checkpoints/stage2')
        best_models = sorted(stage2_dir.glob('best_model_*.pt'), reverse=True)
        if best_models:
            stage2_checkpoint = best_models[0]
    
    if stage2_checkpoint.exists():
        print(f"\nFound Stage 2 checkpoint: {stage2_checkpoint}")
        metrics = evaluate_finetuned_model(str(stage2_checkpoint), 'stage2')
        all_metrics.append(metrics)
    else:
        print(f"\n⚠ Stage 2 checkpoint not found: {stage2_checkpoint}")
    
    # Summary
    if all_metrics:
        print(f"\n{'='*80}")
        print("SUMMARY - FINETUNING RESULTS")
        print(f"{'='*80}")
        
        for metrics in all_metrics:
            print(f"\n{metrics['stage'].upper()}:")
            print(f"  MAE: {metrics['mae']:.4f} BPM")
            print(f"  RMSE: {metrics['rmse']:.4f} BPM")
            print(f"  R²: {metrics['r2']:.4f}")
            print(f"  Epochs trained: {metrics['epoch']}")
        
        # Compare stages
        if len(all_metrics) == 2:
            mae_improvement = all_metrics[0]['mae'] - all_metrics[1]['mae']
            print(f"\nImprovement from Stage 1 to Stage 2:")
            print(f"  MAE: {mae_improvement:+.4f} BPM")
            if mae_improvement > 0:
                print(f"  Stage 2 is better by {mae_improvement:.4f} BPM!")
            else:
                print(f"  Stage 1 is better by {-mae_improvement:.4f} BPM")
    
    print(f"\n{'='*80}")
    print("EVALUATION COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
