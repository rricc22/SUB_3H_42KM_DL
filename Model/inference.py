#!/usr/bin/env python3
"""
Inference script for heart rate prediction models.
Supports both LSTM and LSTM with embeddings.
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from LSTM import HeartRateLSTM, WorkoutDataset as BasicDataset
from LSTM_with_embeddings import HeartRateLSTMWithEmbeddings, WorkoutDataset as EmbeddingDataset


def load_model_and_args(checkpoint_path, device='cpu'):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint
        device: 'cuda' or 'cpu'
    
    Returns:
        model: Loaded model in eval mode
        args: Training arguments
        model_type: 'lstm' or 'lstm_embeddings'
    """
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args = checkpoint['args']
    model_type = args.model
    
    # Create model based on type
    if model_type == 'lstm':
        model = HeartRateLSTM(
            input_size=3,  # speed, altitude, gender
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            bidirectional=args.bidirectional if hasattr(args, 'bidirectional') else False
        )
    elif model_type == 'lstm_embeddings':
        # Need to get num_users from checkpoint
        model = HeartRateLSTMWithEmbeddings(
            input_size=3,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            embedding_dim=args.embedding_dim,
            num_users=checkpoint.get('num_users', 100),  # Fallback if not saved
            bidirectional=args.bidirectional if hasattr(args, 'bidirectional') else False
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded: {model_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, args, model_type


def load_scaler_params(scaler_path):
    """Load normalization parameters."""
    with open(scaler_path, 'r') as f:
        params = json.load(f)
    print(f"✓ Scaler params loaded")
    print(f"  Speed: mean={params['speed_mean']:.2f}, std={params['speed_std']:.2f}")
    print(f"  Altitude: mean={params['altitude_mean']:.2f}, std={params['altitude_std']:.2f}")
    return params


def preprocess_workout(speed, altitude, gender, original_length, scaler_params, seq_length=500):
    """
    Preprocess a single workout for inference.
    
    Args:
        speed: array of speed values
        altitude: array of altitude values
        gender: 'male' or 'female'
        original_length: original workout length (for depadding later)
        scaler_params: dict with normalization params
        seq_length: target sequence length (default: 500)
    
    Returns:
        speed_norm: normalized and padded speed [seq_length]
        altitude_norm: normalized and padded altitude [seq_length]
        gender_encoded: 1.0 for male, 0.0 for female
        original_length: original length before padding
    """
    # Pad or truncate
    def pad_or_truncate(seq, length):
        if len(seq) >= length:
            return seq[:length]
        else:
            padding = np.full(length - len(seq), seq[-1])
            return np.concatenate([seq, padding])
    
    speed = pad_or_truncate(speed, seq_length)
    altitude = pad_or_truncate(altitude, seq_length)
    
    # Normalize
    speed_norm = (speed - scaler_params['speed_mean']) / scaler_params['speed_std']
    altitude_norm = (altitude - scaler_params['altitude_mean']) / scaler_params['altitude_std']
    
    # Encode gender
    gender_encoded = 1.0 if str(gender).lower() == 'male' else 0.0
    
    return speed_norm, altitude_norm, gender_encoded, original_length


def predict_single(model, speed, altitude, gender, original_length, model_type, userId=None, device='cpu'):
    """
    Run inference on a single workout.
    
    Args:
        model: Loaded model
        speed: normalized speed [seq_length]
        altitude: normalized altitude [seq_length]
        gender: encoded gender (1.0 or 0.0)
        original_length: original workout length
        model_type: 'lstm' or 'lstm_embeddings'
        userId: user ID (required for lstm_embeddings)
        device: 'cuda' or 'cpu'
    
    Returns:
        predicted_hr: array of predicted heart rates [original_length]
    """
    # Convert to tensors
    speed_tensor = torch.FloatTensor(speed).unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    altitude_tensor = torch.FloatTensor(altitude).unsqueeze(0).unsqueeze(-1)
    gender_tensor = torch.FloatTensor([gender]).unsqueeze(-1)  # [1, 1]
    lengths_tensor = torch.LongTensor([original_length]).unsqueeze(-1)  # [1, 1]
    
    # Move to device
    speed_tensor = speed_tensor.to(device)
    altitude_tensor = altitude_tensor.to(device)
    gender_tensor = gender_tensor.to(device)
    lengths_tensor = lengths_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        if model_type == 'lstm':
            predictions = model(speed_tensor, altitude_tensor, gender_tensor, lengths_tensor)
        elif model_type == 'lstm_embeddings':
            if userId is None:
                raise ValueError("userId required for lstm_embeddings model")
            userId_tensor = torch.LongTensor([userId]).unsqueeze(-1).to(device)
            predictions = model(speed_tensor, altitude_tensor, gender_tensor, userId_tensor, lengths_tensor)
    
    # Convert to numpy and depad
    predicted_hr = predictions.squeeze().cpu().numpy()  # [seq_len]
    predicted_hr = predicted_hr[:original_length]  # Remove padding
    
    # Clip to valid HR range
    predicted_hr = np.clip(predicted_hr, 50, 220)
    
    return predicted_hr


def predict_batch(model, data_loader, model_type, device='cpu'):
    """
    Run inference on a batch of workouts (efficient for GPU).
    
    Args:
        model: Loaded model
        data_loader: PyTorch DataLoader
        model_type: 'lstm' or 'lstm_embeddings'
        device: 'cuda' or 'cpu'
    
    Returns:
        all_predictions: list of arrays (predicted HR for each workout)
        all_targets: list of arrays (true HR for each workout)
        all_lengths: list of original lengths
    """
    all_predictions = []
    all_targets = []
    all_lengths = []
    
    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            if model_type == 'lstm':
                speed, altitude, gender, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                heart_rate = heart_rate.to(device)
                original_lengths = original_lengths.to(device)
                
                predictions = model(speed, altitude, gender, original_lengths)
            
            elif model_type == 'lstm_embeddings':
                speed, altitude, gender, userId, heart_rate, original_lengths = batch
                speed = speed.to(device)
                altitude = altitude.to(device)
                gender = gender.to(device)
                userId = userId.to(device)
                heart_rate = heart_rate.to(device)
                original_lengths = original_lengths.to(device)
                
                predictions = model(speed, altitude, gender, userId, original_lengths)
            
            # Move to CPU and process each sample in batch
            predictions = predictions.cpu().numpy()  # [batch, seq_len, 1]
            heart_rate = heart_rate.cpu().numpy()
            lengths = original_lengths.cpu().numpy().flatten()
            
            for i in range(len(predictions)):
                length = int(lengths[i])
                pred = predictions[i, :length, 0]  # Depad
                target = heart_rate[i, :length, 0]
                
                # Clip predictions
                pred = np.clip(pred, 50, 220)
                
                all_predictions.append(pred)
                all_targets.append(target)
                all_lengths.append(length)
    
    return all_predictions, all_targets, all_lengths


def compute_metrics(predictions, targets):
    """
    Compute evaluation metrics.
    
    Args:
        predictions: list of arrays or single array
        targets: list of arrays or single array
    
    Returns:
        metrics: dict with MAE, RMSE, R2, etc.
    """
    # Flatten if list
    if isinstance(predictions, list):
        predictions = np.concatenate(predictions)
        targets = np.concatenate(targets)
    
    # Compute metrics
    errors = predictions - targets
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    
    # R² score
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Additional stats
    max_error = np.max(np.abs(errors))
    median_error = np.median(np.abs(errors))
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'max_error': max_error,
        'median_error': median_error,
        'mean_pred': np.mean(predictions),
        'mean_target': np.mean(targets),
        'std_pred': np.std(predictions),
        'std_target': np.std(targets)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Inference for heart rate prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data file (.pt)')
    parser.add_argument('--scaler', type=str, default='DATA/processed/scaler_params.json',
                        help='Path to scaler params')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"\n{'='*80}")
    print(f"HEART RATE PREDICTION - INFERENCE")
    print(f"{'='*80}")
    print(f"Device: {device}")
    
    # Load model
    model, train_args, model_type = load_model_and_args(args.checkpoint, device)
    
    # Load scaler
    scaler_params = load_scaler_params(args.scaler)
    
    # Load data
    print(f"\nLoading data from {args.data}...")
    data = torch.load(args.data, weights_only=False)
    
    if model_type == 'lstm':
        dataset = BasicDataset(data)
    else:
        dataset = EmbeddingDataset(data)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"✓ Data loaded: {len(dataset)} samples")
    
    # Run inference
    print(f"\n{'='*80}")
    print(f"RUNNING INFERENCE")
    print(f"{'='*80}")
    
    predictions, targets, lengths = predict_batch(model, data_loader, model_type, device)
    
    print(f"✓ Inference complete: {len(predictions)} workouts")
    
    # Compute metrics
    print(f"\n{'='*80}")
    print(f"METRICS")
    print(f"{'='*80}")
    
    metrics = compute_metrics(predictions, targets)
    
    print(f"MAE: {metrics['mae']:.2f} BPM")
    print(f"RMSE: {metrics['rmse']:.2f} BPM")
    print(f"R²: {metrics['r2']:.4f}")
    print(f"Max Error: {metrics['max_error']:.2f} BPM")
    print(f"Median Error: {metrics['median_error']:.2f} BPM")
    print(f"\nPrediction Stats:")
    print(f"  Mean: {metrics['mean_pred']:.2f} BPM")
    print(f"  Std: {metrics['std_pred']:.2f} BPM")
    print(f"\nTarget Stats:")
    print(f"  Mean: {metrics['mean_target']:.2f} BPM")
    print(f"  Std: {metrics['std_target']:.2f} BPM")
    
    print(f"\n{'='*80}")
    print(f"✓ INFERENCE COMPLETE!")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
