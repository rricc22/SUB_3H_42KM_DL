#!/usr/bin/env python3
"""
Evaluate model on test set with detailed metrics and visualizations.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from LSTM import HeartRateLSTM, WorkoutDataset as BasicDataset
from LSTM_with_embeddings import HeartRateLSTMWithEmbeddings, WorkoutDataset as EmbeddingDataset
from GRU import HeartRateGRU


def load_model(checkpoint_path, device='cpu'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = checkpoint['args']
    
    # Handle both dict and argparse.Namespace
    if isinstance(args, dict):
        model_type = args['model']
        hidden_size = args['hidden_size']
        num_layers = args['num_layers']
        dropout = args['dropout']
        bidirectional = args.get('bidirectional', False)
        embedding_dim = args.get('embedding_dim', 32)
    else:
        model_type = args.model
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        dropout = args.dropout
        bidirectional = args.bidirectional if hasattr(args, 'bidirectional') else False
        embedding_dim = args.embedding_dim if hasattr(args, 'embedding_dim') else 32
    
    if model_type == 'lstm':
        model = HeartRateLSTM(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    elif model_type == 'gru':
        model = HeartRateGRU(
            input_size=3,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    elif model_type == 'lstm_embeddings':
        model = HeartRateLSTMWithEmbeddings(
            num_users=checkpoint.get('num_users', 100),
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
    else:
        # Default fallback for unknown types
        raise ValueError(f"Unknown model type: {model_type}. Supported: 'lstm', 'gru', 'lstm_embeddings'")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Return args as namespace for easier access
    if isinstance(args, dict):
        from argparse import Namespace
        args = Namespace(**args)
    
    return model, model_type, args


def evaluate(model, data_loader, model_type, device):
    """Run evaluation and collect predictions."""
    all_predictions = []
    all_targets = []
    all_lengths = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if model_type in ['lstm', 'gru']:
                speed, altitude, gender, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                heart_rate = heart_rate.to(device)
                original_lengths = original_lengths.to(device)
                
                predictions = model(speed, altitude, gender, original_lengths)
            else:  # lstm_embeddings
                speed, altitude, gender, userId, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                userId = userId.to(device)
                heart_rate = heart_rate.to(device)
                original_lengths = original_lengths.to(device)
                
                predictions = model(speed, altitude, gender, userId, original_lengths)
            
            predictions = predictions.cpu().numpy()
            heart_rate = heart_rate.cpu().numpy()
            lengths = original_lengths.cpu().numpy().flatten()
            
            for i in range(len(predictions)):
                length = int(lengths[i])
                pred = np.clip(predictions[i, :length, 0], 50, 220)
                target = heart_rate[i, :length, 0]
                
                all_predictions.append(pred)
                all_targets.append(target)
                all_lengths.append(length)
    
    return all_predictions, all_targets, all_lengths


def compute_metrics(predictions, targets):
    """Compute all metrics."""
    pred_flat = np.concatenate(predictions)
    target_flat = np.concatenate(targets)
    
    errors = pred_flat - target_flat
    abs_errors = np.abs(errors)
    
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean(errors ** 2))
    
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((target_flat - np.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Per-workout metrics
    per_workout_mae = [np.mean(np.abs(p - t)) for p, t in zip(predictions, targets)]
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'max_error': np.max(abs_errors),
        'median_error': np.median(abs_errors),
        'std_error': np.std(errors),
        'mean_pred': np.mean(pred_flat),
        'mean_target': np.mean(target_flat),
        'per_workout_mae': per_workout_mae,
        'errors': errors,
        'abs_errors': abs_errors,
        'predictions': pred_flat,
        'targets': target_flat
    }


def plot_results(metrics, predictions, targets, lengths, output_path):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Scatter: Predicted vs True
    ax1 = plt.subplot(2, 4, 1)
    pred_flat = metrics['predictions']
    target_flat = metrics['targets']
    ax1.scatter(target_flat, pred_flat, alpha=0.3, s=1)
    ax1.plot([50, 220], [50, 220], 'r--', label='Perfect prediction')
    ax1.set_xlabel('True HR (BPM)')
    ax1.set_ylabel('Predicted HR (BPM)')
    ax1.set_title(f'Predictions vs True\nMAE: {metrics["mae"]:.2f} BPM')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Error distribution
    ax2 = plt.subplot(2, 4, 2)
    ax2.hist(metrics['errors'], bins=100, alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='r', linestyle='--', label='Zero error')
    ax2.set_xlabel('Error (Predicted - True) BPM')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Error Distribution\nMean: {np.mean(metrics["errors"]):.2f} BPM')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Absolute error distribution
    ax3 = plt.subplot(2, 4, 3)
    ax3.hist(metrics['abs_errors'], bins=100, alpha=0.7, color='orange', edgecolor='black')
    ax3.axvline(5, color='g', linestyle='--', label='Target: 5 BPM')
    ax3.axvline(10, color='orange', linestyle='--', label='Acceptable: 10 BPM')
    ax3.axvline(metrics['mae'], color='r', linestyle='-', linewidth=2, label=f'MAE: {metrics["mae"]:.2f}')
    ax3.set_xlabel('Absolute Error (BPM)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Absolute Error Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Per-workout MAE
    ax4 = plt.subplot(2, 4, 4)
    per_workout_mae = metrics['per_workout_mae']
    ax4.hist(per_workout_mae, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax4.axvline(np.mean(per_workout_mae), color='r', linestyle='-', linewidth=2, label=f'Mean: {np.mean(per_workout_mae):.2f}')
    ax4.set_xlabel('MAE per Workout (BPM)')
    ax4.set_ylabel('Number of Workouts')
    ax4.set_title(f'Per-Workout Performance\n{len(per_workout_mae)} workouts')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Example workout 1
    ax5 = plt.subplot(2, 4, 5)
    idx = 0
    time = np.arange(len(predictions[idx]))
    ax5.plot(time, targets[idx], 'b-', label='True HR', linewidth=1.5)
    ax5.plot(time, predictions[idx], 'r-', label='Predicted HR', alpha=0.7, linewidth=1.5)
    ax5.fill_between(time, targets[idx]-5, targets[idx]+5, alpha=0.2, color='green', label='±5 BPM')
    ax5.set_xlabel('Time (timesteps)')
    ax5.set_ylabel('Heart Rate (BPM)')
    ax5.set_title(f'Workout Example 1\nMAE: {per_workout_mae[idx]:.2f} BPM')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Example workout 2
    ax6 = plt.subplot(2, 4, 6)
    idx = len(predictions) // 2
    time = np.arange(len(predictions[idx]))
    ax6.plot(time, targets[idx], 'b-', label='True HR', linewidth=1.5)
    ax6.plot(time, predictions[idx], 'r-', label='Predicted HR', alpha=0.7, linewidth=1.5)
    ax6.fill_between(time, targets[idx]-5, targets[idx]+5, alpha=0.2, color='green', label='±5 BPM')
    ax6.set_xlabel('Time (timesteps)')
    ax6.set_ylabel('Heart Rate (BPM)')
    ax6.set_title(f'Workout Example 2\nMAE: {per_workout_mae[idx]:.2f} BPM')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Error by HR range
    ax7 = plt.subplot(2, 4, 7)
    bins = [50, 100, 120, 140, 160, 180, 220]
    bin_centers = [(bins[i] + bins[i+1])/2 for i in range(len(bins)-1)]
    bin_labels = [f'{bins[i]}-{bins[i+1]}' for i in range(len(bins)-1)]
    
    mae_by_range = []
    for i in range(len(bins)-1):
        mask = (target_flat >= bins[i]) & (target_flat < bins[i+1])
        if np.sum(mask) > 0:
            mae_by_range.append(np.mean(metrics['abs_errors'][mask]))
        else:
            mae_by_range.append(0)
    
    ax7.bar(range(len(bin_centers)), mae_by_range, tick_label=bin_labels, color='teal', alpha=0.7)
    ax7.set_xlabel('HR Range (BPM)')
    ax7.set_ylabel('MAE (BPM)')
    ax7.set_title('Error by HR Range')
    ax7.grid(True, alpha=0.3, axis='y')
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45)
    
    # 8. Metrics summary
    ax8 = plt.subplot(2, 4, 8)
    ax8.axis('off')
    
    # Evaluation criteria
    if metrics['mae'] < 5:
        evaluation = '🏆 EXCELLENT'
        color = 'green'
    elif metrics['mae'] < 10:
        evaluation = '✅ ACCEPTABLE'
        color = 'orange'
    else:
        evaluation = '⚠️ NEEDS IMPROVEMENT'
        color = 'red'
    
    summary_text = f"""
    OVERALL METRICS
    {'='*40}
    
    MAE:  {metrics['mae']:.2f} BPM
    RMSE: {metrics['rmse']:.2f} BPM
    R²:   {metrics['r2']:.4f}
    
    Max Error:    {metrics['max_error']:.2f} BPM
    Median Error: {metrics['median_error']:.2f} BPM
    Std Error:    {metrics['std_error']:.2f} BPM
    
    {'='*40}
    EVALUATION: {evaluation}
    {'='*40}
    
    Workouts tested: {len(predictions)}
    Total timesteps: {len(pred_flat):,}
    
    % within ±5 BPM:  {100 * np.mean(metrics['abs_errors'] < 5):.1f}%
    % within ±10 BPM: {100 * np.mean(metrics['abs_errors'] < 10):.1f}%
    % within ±15 BPM: {100 * np.mean(metrics['abs_errors'] < 15):.1f}%
    """
    
    ax8.text(0.1, 0.5, summary_text, transform=ax8.transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor=color, alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/lstm_best.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='DATA/processed/test.pt',
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='results/test_evaluation.png',
                        help='Output path for visualization')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*80}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Test data: {args.data}")
    print(f"Device: {device}")
    
    # Load model
    print(f"\nLoading model...")
    model, model_type, train_args = load_model(args.checkpoint, device)
    print(f"✓ Model: {model_type}")
    print(f"  Hidden size: {train_args.hidden_size}")
    print(f"  Num layers: {train_args.num_layers}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data
    print(f"\nLoading test data...")
    data = torch.load(args.data, weights_only=False)
    
    if model_type in ['lstm', 'gru']:
        dataset = BasicDataset(data)
    else:  # lstm_embeddings
        dataset = EmbeddingDataset(data)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✓ Test samples: {len(dataset)}")
    
    # Evaluate
    print(f"\n{'='*80}")
    print(f"RUNNING EVALUATION")
    print(f"{'='*80}")
    
    predictions, targets, lengths = evaluate(model, data_loader, model_type, device)
    print(f"✓ Evaluation complete")
    
    # Compute metrics
    print(f"\nComputing metrics...")
    metrics = compute_metrics(predictions, targets)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"\nOverall Metrics:")
    print(f"  MAE:  {metrics['mae']:.2f} BPM")
    print(f"  RMSE: {metrics['rmse']:.2f} BPM")
    print(f"  R²:   {metrics['r2']:.4f}")
    print(f"  Max Error:    {metrics['max_error']:.2f} BPM")
    print(f"  Median Error: {metrics['median_error']:.2f} BPM")
    
    print(f"\nAccuracy Breakdown:")
    print(f"  Within ±5 BPM:  {100 * np.mean(metrics['abs_errors'] < 5):.1f}%")
    print(f"  Within ±10 BPM: {100 * np.mean(metrics['abs_errors'] < 10):.1f}%")
    print(f"  Within ±15 BPM: {100 * np.mean(metrics['abs_errors'] < 15):.1f}%")
    
    print(f"\nPer-Workout Stats:")
    print(f"  Best workout MAE:  {np.min(metrics['per_workout_mae']):.2f} BPM")
    print(f"  Worst workout MAE: {np.max(metrics['per_workout_mae']):.2f} BPM")
    print(f"  Median workout MAE: {np.median(metrics['per_workout_mae']):.2f} BPM")
    
    # Evaluation
    if metrics['mae'] < 5:
        print(f"\n🏆 EVALUATION: EXCELLENT (MAE < 5 BPM)")
    elif metrics['mae'] < 10:
        print(f"\n✅ EVALUATION: ACCEPTABLE (MAE < 10 BPM)")
    else:
        print(f"\n⚠️ EVALUATION: NEEDS IMPROVEMENT (MAE > 10 BPM)")
    
    # Create visualization
    print(f"\n{'='*80}")
    print(f"CREATING VISUALIZATION")
    print(f"{'='*80}")
    
    plot_results(metrics, predictions, targets, lengths, args.output)
    
    print(f"\n{'='*80}")
    print(f"✅ EVALUATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
