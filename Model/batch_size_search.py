#!/usr/bin/env python3
"""
Automated batch size search script for heart rate prediction models.

Launches parallel training runs with different batch sizes and compares results.

Usage:
    # Test batch sizes 16, 32, 64 in parallel
    python3 Model/batch_size_search.py --model lstm --batch_sizes 16 32 64 --epochs 50
    
    # Test with LSTM embeddings
    python3 Model/batch_size_search.py --model lstm_embeddings --batch_sizes 16 32 64 --epochs 50
    
    # Quick test with fewer epochs
    python3 Model/batch_size_search.py --model lstm --batch_sizes 16 32 --epochs 20
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
import torch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch size search for heart rate prediction')
    
    # Model selection
    parser.add_argument('--model', type=str, default='lstm', 
                        choices=['lstm', 'lstm_embeddings', 'lag_llama', 'patchtst'],
                        help='Model type')
    
    # Search parameters
    parser.add_argument('--batch_sizes', type=int, nargs='+', 
                        default=[16, 32, 64],
                        help='Batch sizes to test (default: 16 32 64)')
    
    # Training hyperparameters (fixed across runs)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs per run (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    
    # Model architecture
    parser.add_argument('--hidden_size', type=int, default=64,
                        help='LSTM hidden size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers (default: 2)')
    parser.add_argument('--embedding_dim', type=int, default=16,
                        help='User embedding dimension (default: 16)')
    
    # Paths
    parser.add_argument('--data_dir', type=str, default='DATA/processed',
                        help='Directory with preprocessed data')
    parser.add_argument('--output_dir', type=str, default='experiments/batch_size_search',
                        help='Directory to save results')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use: cuda or cpu (default: cuda)')
    
    return parser.parse_args()


def launch_training(batch_size, args):
    """
    Launch training process for specific batch size.
    
    Args:
        batch_size: Batch size to use
        args: Arguments namespace
    
    Returns:
        process: Subprocess object
        log_file: Path to log file
        start_time: Start timestamp
    """
    print(f"Launching training with batch_size={batch_size}...")
    
    # Create command
    cmd = [
        'python3', 'Model/train.py',
        '--model', args.model,
        '--batch_size', str(batch_size),
        '--epochs', str(args.epochs),
        '--lr', str(args.lr),
        '--patience', str(args.patience),
        '--hidden_size', str(args.hidden_size),
        '--num_layers', str(args.num_layers),
        '--embedding_dim', str(args.embedding_dim),
        '--data_dir', args.data_dir,
        '--checkpoint_dir', str(Path(args.output_dir) / f'bs{batch_size}'),
        '--device', args.device
    ]
    
    # Log file
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f'training_bs{batch_size}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Launch process
    log_f = open(log_file, 'w')
    process = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    start_time = time.time()
    
    print(f"  PID: {process.pid}")
    print(f"  Log: {log_file}")
    
    return process, log_file, start_time, log_f


def monitor_processes(processes_info):
    """
    Monitor running processes and collect results.
    
    Args:
        processes_info: List of tuples (batch_size, process, log_file, start_time, log_f)
    
    Returns:
        results: List of result dictionaries
    """
    results = []
    
    print(f"\n{'='*80}")
    print(f"MONITORING {len(processes_info)} PARALLEL TRAINING PROCESSES")
    print(f"{'='*80}\n")
    
    # Wait for all processes to complete
    while processes_info:
        for i, (batch_size, process, log_file, start_time, log_f) in enumerate(processes_info):
            # Check if process finished
            retcode = process.poll()
            
            if retcode is not None:
                # Process finished
                elapsed_time = time.time() - start_time
                log_f.close()
                
                if retcode == 0:
                    print(f"\n✓ Batch size {batch_size} completed successfully!")
                    print(f"  Time: {elapsed_time/60:.2f} minutes")
                    print(f"  Log: {log_file}")
                    
                    # Extract results
                    result = extract_results(batch_size, log_file, elapsed_time)
                    if result:
                        results.append(result)
                else:
                    print(f"\n✗ Batch size {batch_size} failed with return code {retcode}")
                    print(f"  Log: {log_file}")
                
                # Remove from list
                processes_info.pop(i)
                break
        
        # Sleep briefly to avoid busy waiting
        time.sleep(5)
    
    return results


def extract_results(batch_size, log_file, elapsed_time):
    """
    Extract results from completed training run.
    
    Args:
        batch_size: Batch size used
        log_file: Path to log file
        elapsed_time: Training time in seconds
    
    Returns:
        result: Dictionary with results (or None if extraction failed)
    """
    try:
        # Find checkpoint directory
        checkpoint_dir = log_file.parent.parent / f'bs{batch_size}'
        checkpoint_files = list(checkpoint_dir.glob('*.pt'))
        
        if not checkpoint_files:
            print(f"  Warning: No checkpoint found for batch_size={batch_size}")
            return None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_files[0], weights_only=False)
        metrics = checkpoint['metrics']
        history = checkpoint['history']
        
        result = {
            'batch_size': batch_size,
            'test_mae': metrics['mae'],
            'test_rmse': metrics['rmse'],
            'test_r2': metrics['r2'],
            'test_mse': metrics['mse'],
            'best_val_mae': min(history['val_mae']),
            'final_train_mae': history['train_mae'][-1],
            'final_val_mae': history['val_mae'][-1],
            'epochs_trained': len(history['train_loss']),
            'training_time_min': elapsed_time / 60,
            'checkpoint_file': str(checkpoint_files[0]),
            'log_file': str(log_file)
        }
        
        return result
        
    except Exception as e:
        print(f"  Error extracting results for batch_size={batch_size}: {e}")
        return None


def save_results(results, args):
    """
    Save results to JSON and CSV.
    
    Args:
        results: List of result dictionaries
        args: Arguments namespace
    """
    if not results:
        print("\n✗ No results to save!")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sort by batch size
    results = sorted(results, key=lambda x: x['batch_size'])
    
    # Save JSON
    json_file = output_dir / f'batch_size_results_{args.model}.json'
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {json_file}")
    
    # Save CSV
    try:
        import pandas as pd
        df = pd.DataFrame(results)
        csv_file = output_dir / f'batch_size_results_{args.model}.csv'
        df.to_csv(csv_file, index=False)
        print(f"✓ Results saved to: {csv_file}")
    except ImportError:
        print("  (pandas not available - skipping CSV export)")
    
    # Print summary table
    print(f"\n{'='*80}")
    print(f"BATCH SIZE SEARCH RESULTS - {args.model.upper()}")
    print(f"{'='*80}\n")
    
    print(f"{'Batch':<8} {'Test MAE':<12} {'Test RMSE':<12} {'Test R²':<10} {'Val MAE':<12} {'Time (min)':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['batch_size']:<8} {r['test_mae']:<12.2f} {r['test_rmse']:<12.2f} {r['test_r2']:<10.4f} {r['best_val_mae']:<12.2f} {r['training_time_min']:<12.2f}")
    
    # Find best
    best = min(results, key=lambda x: x['test_mae'])
    print(f"\n{'='*80}")
    print(f"BEST BATCH SIZE: {best['batch_size']}")
    print(f"  Test MAE:  {best['test_mae']:.2f} BPM")
    print(f"  Test RMSE: {best['test_rmse']:.2f} BPM")
    print(f"  Test R²:   {best['test_r2']:.4f}")
    print(f"  Val MAE:   {best['best_val_mae']:.2f} BPM")
    print(f"  Time:      {best['training_time_min']:.2f} min")
    print(f"{'='*80}\n")


def main():
    """Main execution."""
    args = parse_args()
    
    print(f"\n{'='*80}")
    print(f"BATCH SIZE SEARCH - PARALLEL EXECUTION")
    print(f"{'='*80}")
    print(f"Model: {args.model}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}\n")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Launch all training processes in parallel
    print("Launching training processes...\n")
    processes_info = []
    
    for batch_size in args.batch_sizes:
        process, log_file, start_time, log_f = launch_training(batch_size, args)
        processes_info.append((batch_size, process, log_file, start_time, log_f))
        time.sleep(2)  # Small delay between launches
    
    print(f"\n✓ Launched {len(processes_info)} training processes")
    
    # Monitor and collect results
    overall_start = time.time()
    results = monitor_processes(processes_info)
    overall_time = time.time() - overall_start
    
    # Save results
    if results:
        save_results(results, args)
        
        print(f"\n{'='*80}")
        print(f"✓ BATCH SIZE SEARCH COMPLETE!")
        print(f"{'='*80}")
        print(f"Total time: {overall_time/60:.2f} minutes")
        print(f"Successful runs: {len(results)}/{len(args.batch_sizes)}")
        print(f"\nNext step: Run visualization script")
        print(f"  python3 Model/visualize_batch_comparison.py --input {args.output_dir}")
        print(f"{'='*80}\n")
    else:
        print(f"\n✗ No successful training runs!")
        print(f"Check log files in: {args.output_dir}/logs/\n")


if __name__ == '__main__':
    main()
