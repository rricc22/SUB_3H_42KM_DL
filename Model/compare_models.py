#!/usr/bin/env python3
"""
Compare all trained models and extract their metrics
"""

import torch
import json
from pathlib import Path
import pandas as pd
from typing import Dict, List

def extract_metrics_from_checkpoint(checkpoint_path: str) -> Dict:
    """Extract metrics from a model checkpoint"""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get model name from path
        model_name = Path(checkpoint_path).stem
        
        # Extract metrics
        metrics = {
            'model_name': model_name,
            'path': str(checkpoint_path),
        }
        
        # Get metrics from checkpoint
        if 'metrics' in checkpoint:
            m = checkpoint['metrics']
            metrics['mae'] = m.get('mae', None)
            metrics['rmse'] = m.get('rmse', None)
            metrics['r2'] = m.get('r2', None)
        
        # Get hyperparameters
        if 'args' in checkpoint:
            args = checkpoint['args']
            metrics['batch_size'] = args.get('batch_size', None)
            metrics['lr'] = args.get('lr', None)
            metrics['epochs'] = args.get('epochs', None)
            metrics['hidden_size'] = args.get('hidden_size', None)
            metrics['num_layers'] = args.get('num_layers', None)
            metrics['model_type'] = args.get('model', None)
        
        # Get validation loss from history
        if 'history' in checkpoint:
            history = checkpoint['history']
            if 'val_loss' in history and len(history['val_loss']) > 0:
                metrics['val_loss'] = history['val_loss'][-1]
        
        return metrics
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def extract_metrics_from_json(json_path: str) -> Dict:
    """Extract metrics from PatchTST JSON file"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        model_name = Path(json_path).stem.replace('_metrics', '')
        
        metrics = {
            'model_name': model_name,
            'path': str(json_path),
            'mae': data.get('test_mae', None),
            'rmse': data.get('test_rmse', None),
            'r2': data.get('test_r2', None),
            'model_type': 'patchtst',
        }
        
        return metrics
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None

def main():
    """Extract and compare all model metrics"""
    print("="*80)
    print("MODEL COMPARISON - EXTRACTING METRICS FROM ALL TRAINED MODELS")
    print("="*80)
    
    all_metrics = []
    
    # 1. Check batch size search experiments
    print("\n1. Checking batch size search experiments...")
    batch_search_dir = Path('experiments/batch_size_search')
    for bs_dir in batch_search_dir.glob('bs*'):
        for model_file in bs_dir.glob('*_best.pt'):
            print(f"   Loading: {model_file}")
            metrics = extract_metrics_from_checkpoint(str(model_file))
            if metrics:
                all_metrics.append(metrics)
    
    # 2. Check archived checkpoints
    print("\n2. Checking archived checkpoints...")
    archive_dir = Path('Archives/checkpoints_archives')
    
    # Main archived models
    for model_file in archive_dir.glob('*_best.pt'):
        if not model_file.is_dir():
            print(f"   Loading: {model_file}")
            metrics = extract_metrics_from_checkpoint(str(model_file))
            if metrics:
                all_metrics.append(metrics)
    
    # Apple watch v2 models
    apple_watch_dir = archive_dir / 'apple_watch_v2_lstm_emb'
    if apple_watch_dir.exists():
        for model_file in apple_watch_dir.glob('*_best.pt'):
            print(f"   Loading: {model_file}")
            metrics = extract_metrics_from_checkpoint(str(model_file))
            if metrics:
                all_metrics.append(metrics)
    
    # Lag-Llama transfer learning
    lag_llama_dir = archive_dir / 'lag_llama_transfert_learning'
    if lag_llama_dir.exists():
        for model_file in lag_llama_dir.glob('*_best.pt'):
            print(f"   Loading: {model_file}")
            metrics = extract_metrics_from_checkpoint(str(model_file))
            if metrics:
                all_metrics.append(metrics)
    
    # PatchTST models (JSON format)
    for json_file in archive_dir.glob('patchtst_*_metrics.json'):
        print(f"   Loading: {json_file}")
        metrics = extract_metrics_from_json(str(json_file))
        if metrics:
            all_metrics.append(metrics)
    
    # 3. Check finetuned models (load from test_metrics.json)
    print("\n3. Checking finetuned models...")
    finetune_stages = ['stage1', 'stage2']
    for stage in finetune_stages:
        metrics_file = Path(f'results/{stage}/test_metrics.json')
        if metrics_file.exists():
            print(f"   Loading: {metrics_file}")
            try:
                with open(metrics_file, 'r') as f:
                    data = json.load(f)
                
                metrics = {
                    'model_name': f'best_model',
                    'model_type': f'finetuned_{stage}',
                    'mae': data.get('mae'),
                    'rmse': data.get('rmse'),
                    'r2': data.get('r2'),
                    'val_loss': data.get('best_val_loss'),
                    'batch_size': data.get('config', {}).get('batch_size'),
                    'lr': data.get('config', {}).get('learning_rate'),
                    'epochs': data.get('epoch'),
                    'hidden_size': data.get('config', {}).get('hidden_size'),
                    'num_layers': data.get('config', {}).get('num_layers'),
                }
                all_metrics.append(metrics)
            except Exception as e:
                print(f"   Error loading {metrics_file}: {e}")
        else:
            # Try to load from checkpoint and show warning
            checkpoint_path = Path(f'checkpoints/{stage}/best_model.pt')
            if checkpoint_path.exists():
                print(f"   ⚠ Found checkpoint but no test metrics: {checkpoint_path}")
                print(f"   Run: python3 Model/evaluate_finetuned.py to generate metrics")
            else:
                print(f"   No {stage} model found")
    
    # 4. Create DataFrame and sort
    print("\n4. Creating comparison table...")
    df = pd.DataFrame(all_metrics)
    
    # Select relevant columns
    columns = ['model_name', 'model_type', 'mae', 'rmse', 'r2', 'val_loss', 
               'batch_size', 'lr', 'epochs', 'hidden_size', 'num_layers']
    
    # Only keep columns that exist
    columns = [col for col in columns if col in df.columns]
    df = df[columns]
    
    # Sort by MAE (ascending - lower is better)
    if 'mae' in df.columns:
        df = df.sort_values('mae', ascending=True)
    
    # Print table
    print("\n" + "="*80)
    print("RESULTS - ALL MODELS COMPARISON")
    print("="*80)
    print(df.to_string(index=False))
    
    # Save to CSV
    output_path = Path('Model/model_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    if 'mae' in df.columns:
        print(f"\nBest MAE: {df['mae'].min():.4f} ({df.loc[df['mae'].idxmin(), 'model_name']})")
        print(f"Worst MAE: {df['mae'].max():.4f} ({df.loc[df['mae'].idxmax(), 'model_name']})")
        print(f"Mean MAE: {df['mae'].mean():.4f}")
        print(f"Median MAE: {df['mae'].median():.4f}")
    
    if 'model_type' in df.columns:
        print("\nModels by type:")
        print(df.groupby('model_type')['mae'].agg(['count', 'mean', 'min', 'max']))
    
    # Save summary
    summary_path = Path('Model/model_comparison_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MODEL COMPARISON SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n" + "="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")
        if 'mae' in df.columns:
            f.write(f"\nBest MAE: {df['mae'].min():.4f} ({df.loc[df['mae'].idxmin(), 'model_name']})\n")
            f.write(f"Worst MAE: {df['mae'].max():.4f} ({df.loc[df['mae'].idxmax(), 'model_name']})\n")
            f.write(f"Mean MAE: {df['mae'].mean():.4f}\n")
            f.write(f"Median MAE: {df['mae'].median():.4f}\n")
    
    print(f"✓ Saved summary to: {summary_path}")
    
    print("\n" + "="*80)
    print(f"Total models compared: {len(df)}")
    print("="*80)

if __name__ == "__main__":
    main()
