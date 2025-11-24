#!/usr/bin/env python3
"""
Visualization script for batch size comparison results.

Creates comprehensive plots comparing different batch sizes.

Usage:
    # Visualize results from batch size search
    python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search
    
    # Specify model type
    python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search --model lstm
    
    # Custom output path
    python3 Model/visualize_batch_comparison.py --input experiments/batch_size_search --output results/batch_comparison.png
"""

import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize batch size comparison results')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with batch size search results')
    parser.add_argument('--model', type=str, default='lstm',
                        help='Model type (default: lstm)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot path (default: auto-generated)')
    
    return parser.parse_args()


def load_results(input_dir, model):
    """
    Load results from JSON file.
    
    Args:
        input_dir: Directory with results
        model: Model type
    
    Returns:
        results: List of result dictionaries
    """
    input_path = Path(input_dir)
    json_file = input_path / f'batch_size_results_{model}.json'
    
    if not json_file.exists():
        raise FileNotFoundError(f"Results file not found: {json_file}")
    
    with open(json_file, 'r') as f:
        results = json.load(f)
    
    # Sort by batch size
    results = sorted(results, key=lambda x: x['batch_size'])
    
    return results


def create_comparison_plots(results, model, input_dir, output_path=None):
    """
    Create comprehensive comparison plots.
    
    Args:
        results: List of result dictionaries
        model: Model type
        input_dir: Input directory path
        output_path: Output file path (optional)
    """
    # Extract data
    batch_sizes = [r['batch_size'] for r in results]
    test_mae = [r['test_mae'] for r in results]
    test_rmse = [r['test_rmse'] for r in results]
    test_r2 = [r['test_r2'] for r in results]
    val_mae = [r['best_val_mae'] for r in results]
    train_mae = [r['final_train_mae'] for r in results]
    training_time = [r['training_time_min'] for r in results]
    
    # Create figure with 6 subplots
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    fig.suptitle(f'Batch Size Comparison - {model.upper()}', fontsize=16, fontweight='bold', y=0.995)
    
    # Plot 1: Test MAE vs Batch Size
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(batch_sizes, test_mae, marker='o', linewidth=2.5, markersize=10, color='#2E86AB', label='Test MAE')
    ax1.axhline(y=5, color='g', linestyle='--', linewidth=1.5, label='Target: 5 BPM', alpha=0.7)
    ax1.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, label='Acceptable: 10 BPM', alpha=0.7)
    ax1.set_xlabel('Batch Size', fontweight='bold', fontsize=11)
    ax1.set_ylabel('MAE (BPM)', fontweight='bold', fontsize=11)
    ax1.set_title('Test MAE vs Batch Size', fontweight='bold', fontsize=12)
    ax1.set_xscale('log', base=2)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(fontsize=9)
    
    # Annotate points
    for bs, mae in zip(batch_sizes, test_mae):
        ax1.annotate(f'{mae:.1f}', (bs, mae), textcoords="offset points", 
                     xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
    
    # Plot 2: Validation MAE vs Batch Size
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(batch_sizes, val_mae, marker='s', linewidth=2.5, markersize=10, color='#A23B72', label='Val MAE')
    ax2.axhline(y=5, color='g', linestyle='--', linewidth=1.5, label='Target: 5 BPM', alpha=0.7)
    ax2.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, label='Acceptable: 10 BPM', alpha=0.7)
    ax2.set_xlabel('Batch Size', fontweight='bold', fontsize=11)
    ax2.set_ylabel('MAE (BPM)', fontweight='bold', fontsize=11)
    ax2.set_title('Best Validation MAE vs Batch Size', fontweight='bold', fontsize=12)
    ax2.set_xscale('log', base=2)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(fontsize=9)
    
    # Annotate points
    for bs, mae in zip(batch_sizes, val_mae):
        ax2.annotate(f'{mae:.1f}', (bs, mae), textcoords="offset points", 
                     xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
    
    # Plot 3: Train/Val/Test MAE Comparison
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(batch_sizes))
    width = 0.25
    
    ax3.bar(x - width, train_mae, width, label='Train MAE', color='#F18F01', alpha=0.8)
    ax3.bar(x, val_mae, width, label='Val MAE', color='#A23B72', alpha=0.8)
    ax3.bar(x + width, test_mae, width, label='Test MAE', color='#2E86AB', alpha=0.8)
    
    ax3.axhline(y=5, color='g', linestyle='--', linewidth=1.5, label='Target: 5 BPM', alpha=0.7)
    ax3.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, label='Acceptable: 10 BPM', alpha=0.7)
    
    ax3.set_xlabel('Batch Size', fontweight='bold', fontsize=11)
    ax3.set_ylabel('MAE (BPM)', fontweight='bold', fontsize=11)
    ax3.set_title('Train/Val/Test MAE Comparison', fontweight='bold', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels([str(bs) for bs in batch_sizes])
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Plot 4: Test R² Score vs Batch Size
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(batch_sizes, test_r2, marker='^', linewidth=2.5, markersize=10, color='#06A77D', label='Test R²')
    ax4.set_xlabel('Batch Size', fontweight='bold', fontsize=11)
    ax4.set_ylabel('R² Score', fontweight='bold', fontsize=11)
    ax4.set_title('Test R² vs Batch Size', fontweight='bold', fontsize=12)
    ax4.set_xscale('log', base=2)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.legend(fontsize=9)
    
    # Annotate points
    for bs, r2 in zip(batch_sizes, test_r2):
        ax4.annotate(f'{r2:.3f}', (bs, r2), textcoords="offset points", 
                     xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
    
    # Plot 5: Training Time vs Batch Size
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(batch_sizes, training_time, marker='D', linewidth=2.5, markersize=10, color='#C73E1D', label='Training Time')
    ax5.set_xlabel('Batch Size', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Time (minutes)', fontweight='bold', fontsize=11)
    ax5.set_title('Training Time vs Batch Size', fontweight='bold', fontsize=12)
    ax5.set_xscale('log', base=2)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.legend(fontsize=9)
    
    # Annotate points
    for bs, t in zip(batch_sizes, training_time):
        ax5.annotate(f'{t:.1f}m', (bs, t), textcoords="offset points", 
                     xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
    
    # Plot 6: Test RMSE vs Batch Size
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(batch_sizes, test_rmse, marker='p', linewidth=2.5, markersize=10, color='#7209B7', label='Test RMSE')
    ax6.set_xlabel('Batch Size', fontweight='bold', fontsize=11)
    ax6.set_ylabel('RMSE (BPM)', fontweight='bold', fontsize=11)
    ax6.set_title('Test RMSE vs Batch Size', fontweight='bold', fontsize=12)
    ax6.set_xscale('log', base=2)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.legend(fontsize=9)
    
    # Annotate points
    for bs, rmse in zip(batch_sizes, test_rmse):
        ax6.annotate(f'{rmse:.1f}', (bs, rmse), textcoords="offset points", 
                     xytext=(0,8), ha='center', fontsize=9, fontweight='bold')
    
    # Save plot
    if output_path is None:
        output_path = Path(input_dir) / f'batch_size_comparison_{model}.png'
    else:
        output_path = Path(output_path)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_path}")
    
    # Also save as PDF for publications
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ PDF version saved to: {pdf_path}")
    
    plt.close()


def create_summary_table(results, model, input_dir):
    """
    Create a formatted summary table.
    
    Args:
        results: List of result dictionaries
        model: Model type
        input_dir: Input directory
    """
    print(f"\n{'='*100}")
    print(f"BATCH SIZE COMPARISON SUMMARY - {model.upper()}")
    print(f"{'='*100}\n")
    
    # Header
    header = f"{'Batch':<8} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10} {'Val MAE':<12} {'Train MAE':<12} {'Epochs':<8} {'Time (min)':<12}"
    print(header)
    print('='*100)
    
    # Rows
    for r in results:
        row = (f"{r['batch_size']:<8} "
               f"{r['test_mae']:<12.2f} "
               f"{r['test_rmse']:<12.2f} "
               f"{r['test_r2']:<10.4f} "
               f"{r['best_val_mae']:<12.2f} "
               f"{r['final_train_mae']:<12.2f} "
               f"{r['epochs_trained']:<8} "
               f"{r['training_time_min']:<12.2f}")
        print(row)
    
    print('='*100)
    
    # Find best by different metrics
    best_test_mae = min(results, key=lambda x: x['test_mae'])
    best_val_mae = min(results, key=lambda x: x['best_val_mae'])
    best_r2 = max(results, key=lambda x: x['test_r2'])
    fastest = min(results, key=lambda x: x['training_time_min'])
    
    print(f"\nBest by Test MAE:  Batch Size {best_test_mae['batch_size']} ({best_test_mae['test_mae']:.2f} BPM)")
    print(f"Best by Val MAE:   Batch Size {best_val_mae['batch_size']} ({best_val_mae['best_val_mae']:.2f} BPM)")
    print(f"Best by R²:        Batch Size {best_r2['batch_size']} ({best_r2['test_r2']:.4f})")
    print(f"Fastest training:  Batch Size {fastest['batch_size']} ({fastest['training_time_min']:.2f} min)")
    
    print(f"\n{'='*100}")
    print(f"RECOMMENDATION")
    print(f"{'='*100}")
    
    # Calculate generalization gap
    for r in results:
        r['gen_gap'] = r['test_mae'] - r['best_val_mae']
    
    best_overall = min(results, key=lambda x: (x['test_mae'], x['gen_gap']))
    
    print(f"\nRecommended Batch Size: {best_overall['batch_size']}")
    print(f"  Test MAE:         {best_overall['test_mae']:.2f} BPM")
    print(f"  Val MAE:          {best_overall['best_val_mae']:.2f} BPM")
    print(f"  Generalization:   {best_overall['gen_gap']:.2f} BPM gap")
    print(f"  R² Score:         {best_overall['test_r2']:.4f}")
    print(f"  Training Time:    {best_overall['training_time_min']:.2f} min")
    print(f"  Checkpoint:       {best_overall['checkpoint_file']}")
    
    print(f"\n{'='*100}\n")


def main():
    """Main execution."""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"BATCH SIZE COMPARISON VISUALIZATION")
    print(f"{'='*80}")
    print(f"Input directory: {args.input}")
    print(f"Model: {args.model}")
    print(f"{'='*80}\n")
    
    # Load results
    try:
        results = load_results(args.input, args.model)
        print(f"✓ Loaded results for {len(results)} batch sizes: {[r['batch_size'] for r in results]}\n")
    except FileNotFoundError as e:
        print(f"✗ Error: {e}")
        print(f"\nAvailable files in {args.input}:")
        input_path = Path(args.input)
        if input_path.exists():
            for f in input_path.glob('*.json'):
                print(f"  {f.name}")
        return
    
    # Create summary table
    create_summary_table(results, args.model, args.input)
    
    # Create visualizations
    create_comparison_plots(results, args.model, args.input, args.output)
    
    print(f"{'='*80}")
    print(f"✓ VISUALIZATION COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
