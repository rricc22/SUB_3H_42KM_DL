#!/usr/bin/env python3
"""
Animated visualization of heart rate prediction.

Creates a GIF animation showing the predicted and actual heart rate curves
being drawn point-by-point, along with the input features (speed, altitude).

Usage:
    python3 Model/animate_prediction.py --sample_idx 5
    python3 Model/animate_prediction.py --sample_idx 10 --output results/sample10.gif
"""

import torch
import numpy as np
import json
import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from LSTM import HeartRateLSTM, WorkoutDataset


# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_CHECKPOINT = "experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_best.pt"
DEFAULT_DATA_DIR = "DATA/apple_watch_processed"
DEFAULT_SCALER = "DATA/apple_watch_processed/scaler_params.json"
DEFAULT_OUTPUT = "results/prediction_animation.gif"

# Animation settings
ANIMATION_DURATION_SEC = 30  # Target duration in seconds
FPS = 20  # Frames per second
TOTAL_FRAMES = ANIMATION_DURATION_SEC * FPS  # ~600 frames for smooth animation

# Visual style
STYLE = {
    'figure_facecolor': '#1a1a2e',
    'axes_facecolor': '#16213e',
    'text_color': '#e8e8e8',
    'grid_color': '#2a2a4a',
    'pred_color': '#00d9ff',      # Cyan for prediction
    'actual_color': '#ff6b6b',    # Coral red for actual
    'speed_color': '#4ade80',     # Green for speed
    'altitude_color': '#fbbf24',  # Amber for altitude
    'pred_linewidth': 2.5,
    'actual_linewidth': 2.0,
    'input_linewidth': 1.5,
    'input_alpha': 0.8,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_model(checkpoint_path, device='cpu'):
    """Load trained LSTM model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    args = checkpoint['args']
    
    # Handle both dict and namespace args
    if isinstance(args, dict):
        hidden_size = args.get('hidden_size', 64)
        num_layers = args.get('num_layers', 2)
        dropout = args.get('dropout', 0.2)
        bidirectional = args.get('bidirectional', False)
    else:
        hidden_size = args.hidden_size
        num_layers = args.num_layers
        dropout = args.dropout
        bidirectional = getattr(args, 'bidirectional', False)
    
    model = HeartRateLSTM(
        input_size=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, args


def load_scaler_params(scaler_path):
    """Load normalization parameters."""
    with open(scaler_path, 'r') as f:
        return json.load(f)


def denormalize(data, mean, std):
    """Denormalize data back to original scale."""
    return data * std + mean


def get_sample_data(test_data, sample_idx, scaler_params):
    """
    Extract and prepare a single sample for visualization.
    
    Returns:
        dict with speed, altitude, heart_rate (all denormalized), 
        gender, seq_length (actual usable length)
    """
    # Get tensors
    speed_norm = test_data['speed'][sample_idx].numpy().flatten()
    altitude_norm = test_data['altitude'][sample_idx].numpy().flatten()
    heart_rate = test_data['heart_rate'][sample_idx].numpy().flatten()
    gender = test_data['gender'][sample_idx].item()
    original_length = int(test_data['original_lengths'][sample_idx].item())
    
    # The data is padded/truncated to 500 timesteps
    # Use the minimum of original_length and actual tensor length
    seq_length = min(original_length, len(speed_norm))
    
    # Denormalize inputs for visualization
    speed = denormalize(speed_norm, scaler_params['speed_mean'], scaler_params['speed_std'])
    altitude = denormalize(altitude_norm, scaler_params['altitude_mean'], scaler_params['altitude_std'])
    
    # Trim to usable length
    speed = speed[:seq_length]
    altitude = altitude[:seq_length]
    heart_rate = heart_rate[:seq_length]
    
    return {
        'speed': speed,
        'altitude': altitude,
        'heart_rate': heart_rate,
        'speed_norm': speed_norm,
        'altitude_norm': altitude_norm,
        'gender': gender,
        'original_length': original_length,  # True original (for info display)
        'seq_length': seq_length,            # Usable length (for animation)
        'sample_idx': sample_idx
    }


def predict_heart_rate(model, test_data, sample_idx, device='cpu'):
    """Run model prediction for a single sample."""
    speed = test_data['speed'][sample_idx:sample_idx+1].to(device)
    altitude = test_data['altitude'][sample_idx:sample_idx+1].to(device)
    gender = test_data['gender'][sample_idx:sample_idx+1].to(device)
    original_length = test_data['original_lengths'][sample_idx:sample_idx+1].to(device)
    
    with torch.no_grad():
        prediction = model(speed, altitude, gender, original_length)
    
    pred = prediction.squeeze().cpu().numpy()
    
    # The prediction is for the full padded sequence (500 timesteps)
    # Trim to usable length (min of original_length and model output)
    seq_length = min(int(original_length.item()), len(pred))
    pred = pred[:seq_length]
    pred = np.clip(pred, 50, 220)
    
    return pred


# ============================================================================
# ANIMATION CLASS
# ============================================================================

class PredictionAnimator:
    """Creates animated visualization of heart rate prediction."""
    
    def __init__(self, sample_data, prediction, scaler_params):
        self.sample_data = sample_data
        self.prediction = prediction
        self.scaler_params = scaler_params
        self.length = sample_data['seq_length']  # Use usable length, not original
        
        # Calculate points per frame for smooth animation
        self.points_per_frame = max(1, self.length / TOTAL_FRAMES)
        self.num_frames = int(np.ceil(self.length / self.points_per_frame)) + 10  # Extra frames at end
        
        # Setup figure
        self._setup_figure()
        
    def _setup_figure(self):
        """Create the figure with subplots."""
        # Use dark style
        plt.style.use('dark_background')
        
        # Create figure with custom layout
        self.fig = plt.figure(figsize=(14, 10), facecolor=STYLE['figure_facecolor'])
        
        # GridSpec: 3 rows - HR prediction (large), Speed, Altitude
        gs = GridSpec(3, 1, height_ratios=[2.5, 1, 1], hspace=0.25)
        
        # Main HR plot
        self.ax_hr = self.fig.add_subplot(gs[0])
        self.ax_speed = self.fig.add_subplot(gs[1])
        self.ax_altitude = self.fig.add_subplot(gs[2])
        
        # Configure all axes
        for ax in [self.ax_hr, self.ax_speed, self.ax_altitude]:
            ax.set_facecolor(STYLE['axes_facecolor'])
            ax.tick_params(colors=STYLE['text_color'])
            ax.spines['bottom'].set_color(STYLE['grid_color'])
            ax.spines['top'].set_color(STYLE['grid_color'])
            ax.spines['left'].set_color(STYLE['grid_color'])
            ax.spines['right'].set_color(STYLE['grid_color'])
            ax.grid(True, alpha=0.3, color=STYLE['grid_color'], linestyle='--')
        
        # Set axis limits
        x_max = self.length
        
        # HR axis
        hr_min = min(self.sample_data['heart_rate'].min(), self.prediction.min()) - 10
        hr_max = max(self.sample_data['heart_rate'].max(), self.prediction.max()) + 10
        self.ax_hr.set_xlim(0, x_max)
        self.ax_hr.set_ylim(hr_min, hr_max)
        self.ax_hr.set_ylabel('Heart Rate (BPM)', color=STYLE['text_color'], fontsize=12, fontweight='bold')
        
        # Speed axis
        speed_min = self.sample_data['speed'].min() - 1
        speed_max = self.sample_data['speed'].max() + 1
        self.ax_speed.set_xlim(0, x_max)
        self.ax_speed.set_ylim(speed_min, speed_max)
        self.ax_speed.set_ylabel('Speed (km/h)', color=STYLE['speed_color'], fontsize=11, fontweight='bold')
        
        # Altitude axis
        alt_min = self.sample_data['altitude'].min() - 5
        alt_max = self.sample_data['altitude'].max() + 5
        self.ax_altitude.set_xlim(0, x_max)
        self.ax_altitude.set_ylim(alt_min, alt_max)
        self.ax_altitude.set_ylabel('Altitude (m)', color=STYLE['altitude_color'], fontsize=11, fontweight='bold')
        self.ax_altitude.set_xlabel('Timestep', color=STYLE['text_color'], fontsize=12)
        
        # Initialize lines (empty)
        self.line_pred, = self.ax_hr.plot([], [], color=STYLE['pred_color'], 
                                           linewidth=STYLE['pred_linewidth'],
                                           label='Predicted HR', zorder=10)
        self.line_actual, = self.ax_hr.plot([], [], color=STYLE['actual_color'],
                                             linewidth=STYLE['actual_linewidth'],
                                             label='Actual HR', linestyle='-', alpha=0.9, zorder=5)
        self.line_speed, = self.ax_speed.plot([], [], color=STYLE['speed_color'],
                                               linewidth=STYLE['input_linewidth'],
                                               alpha=STYLE['input_alpha'])
        self.line_altitude, = self.ax_altitude.plot([], [], color=STYLE['altitude_color'],
                                                     linewidth=STYLE['input_linewidth'],
                                                     alpha=STYLE['input_alpha'])
        
        # Add markers for current position
        self.marker_pred, = self.ax_hr.plot([], [], 'o', color=STYLE['pred_color'], 
                                             markersize=8, zorder=15)
        self.marker_actual, = self.ax_hr.plot([], [], 'o', color=STYLE['actual_color'],
                                               markersize=6, zorder=12)
        
        # Legend for HR plot
        self.ax_hr.legend(loc='upper right', facecolor=STYLE['axes_facecolor'],
                          edgecolor=STYLE['grid_color'], labelcolor=STYLE['text_color'],
                          fontsize=10)
        
        # Title with sample info
        gender_str = "Male" if self.sample_data['gender'] == 1.0 else "Female"
        title = f"Heart Rate Prediction | Sample #{self.sample_data['sample_idx']} | {gender_str} | {self.length} timesteps"
        self.fig.suptitle(title, color=STYLE['text_color'], fontsize=14, fontweight='bold', y=0.98)
        
        # MAE text (will be updated)
        self.mae_text = self.ax_hr.text(
            0.02, 0.95, '', transform=self.ax_hr.transAxes,
            color=STYLE['text_color'], fontsize=11, fontweight='bold',
            verticalalignment='top', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor=STYLE['axes_facecolor'], 
                     edgecolor=STYLE['pred_color'], alpha=0.8)
        )
        
        # Progress text
        self.progress_text = self.ax_hr.text(
            0.98, 0.95, '', transform=self.ax_hr.transAxes,
            color=STYLE['text_color'], fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            alpha=0.7
        )
        
        self.fig.subplots_adjust(left=0.08, right=0.95, top=0.93, bottom=0.08, hspace=0.25)
    
    def init_animation(self):
        """Initialize animation (empty frame)."""
        self.line_pred.set_data([], [])
        self.line_actual.set_data([], [])
        self.line_speed.set_data([], [])
        self.line_altitude.set_data([], [])
        self.marker_pred.set_data([], [])
        self.marker_actual.set_data([], [])
        self.mae_text.set_text('')
        self.progress_text.set_text('')
        return (self.line_pred, self.line_actual, self.line_speed, self.line_altitude,
                self.marker_pred, self.marker_actual, self.mae_text, self.progress_text)
    
    def update_frame(self, frame):
        """Update animation for each frame."""
        # Calculate how many points to show
        n_points = min(int(frame * self.points_per_frame), self.length)
        
        if n_points == 0:
            return self.init_animation()
        
        # Time indices
        x = np.arange(n_points)
        
        # Update lines
        self.line_pred.set_data(x, self.prediction[:n_points])
        self.line_actual.set_data(x, self.sample_data['heart_rate'][:n_points])
        self.line_speed.set_data(x, self.sample_data['speed'][:n_points])
        self.line_altitude.set_data(x, self.sample_data['altitude'][:n_points])
        
        # Update markers (current position)
        if n_points > 0:
            self.marker_pred.set_data([n_points-1], [self.prediction[n_points-1]])
            self.marker_actual.set_data([n_points-1], [self.sample_data['heart_rate'][n_points-1]])
        
        # Calculate running MAE
        mae = np.mean(np.abs(self.prediction[:n_points] - self.sample_data['heart_rate'][:n_points]))
        self.mae_text.set_text(f'Running MAE: {mae:.2f} BPM')
        
        # Progress
        progress = (n_points / self.length) * 100
        self.progress_text.set_text(f'{n_points}/{self.length} ({progress:.0f}%)')
        
        return (self.line_pred, self.line_actual, self.line_speed, self.line_altitude,
                self.marker_pred, self.marker_actual, self.mae_text, self.progress_text)
    
    def create_animation(self):
        """Create the animation object."""
        anim = animation.FuncAnimation(
            self.fig, 
            self.update_frame,
            init_func=self.init_animation,
            frames=self.num_frames,
            interval=1000 / FPS,  # milliseconds per frame
            blit=True
        )
        return anim
    
    def save_gif(self, output_path, dpi=100):
        """Save animation as GIF."""
        print(f"\nGenerating animation with {self.num_frames} frames...")
        print(f"Target duration: ~{self.num_frames / FPS:.1f} seconds at {FPS} FPS")
        
        anim = self.create_animation()
        
        # Use PillowWriter (no ffmpeg required)
        writer = animation.PillowWriter(fps=FPS)
        
        # Save with progress callback
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving to {output_path}...")
        anim.save(str(output_path), writer=writer, dpi=dpi)
        
        # Get file size
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"Saved: {output_path} ({size_mb:.2f} MB)")
        
        plt.close(self.fig)
        return output_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create animated visualization of heart rate prediction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 Model/animate_prediction.py --sample_idx 5
    python3 Model/animate_prediction.py --sample_idx 10 --output results/sample10.gif
    python3 Model/animate_prediction.py --split train --sample_idx 29  # Use train set
    python3 Model/animate_prediction.py --sample_idx 0 --dpi 150  # Higher quality
        """
    )
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index of sample to animate (default: 0)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Data split to use: train, val, or test (default: test)')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'Path to model checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help=f'Path to data directory (default: {DEFAULT_DATA_DIR})')
    parser.add_argument('--scaler', type=str, default=DEFAULT_SCALER,
                        help=f'Path to scaler params (default: {DEFAULT_SCALER})')
    parser.add_argument('--output', type=str, default=DEFAULT_OUTPUT,
                        help=f'Output GIF path (default: {DEFAULT_OUTPUT})')
    parser.add_argument('--dpi', type=int, default=100,
                        help='Output DPI (default: 100, use 150 for higher quality)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto (default: auto)')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("=" * 70)
    print("HEART RATE PREDICTION ANIMATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Sample index: {args.sample_idx}")
    
    # Load model
    model, model_args = load_model(args.checkpoint, device)
    
    # Load scaler params
    scaler_params = load_scaler_params(args.scaler)
    print(f"  Scaler params loaded")
    
    # Load data from specified split
    data_path = f"{args.data_dir}/{args.split}.pt"
    print(f"Loading data from {data_path}...")
    test_data = torch.load(data_path, weights_only=False)
    n_samples = len(test_data['speed'])
    print(f"  Loaded {n_samples} test samples")
    
    # Validate sample index
    if args.sample_idx < 0 or args.sample_idx >= n_samples:
        print(f"ERROR: sample_idx must be between 0 and {n_samples - 1}")
        sys.exit(1)
    
    # Get sample data
    sample_data = get_sample_data(test_data, args.sample_idx, scaler_params)
    print(f"\nSample #{args.sample_idx}:")
    print(f"  Original length: {sample_data['original_length']} timesteps")
    print(f"  Usable length: {sample_data['seq_length']} timesteps (truncated to model input)")
    print(f"  Gender: {'Male' if sample_data['gender'] == 1.0 else 'Female'}")
    print(f"  Speed range: {sample_data['speed'].min():.1f} - {sample_data['speed'].max():.1f} km/h")
    print(f"  Altitude range: {sample_data['altitude'].min():.1f} - {sample_data['altitude'].max():.1f} m")
    print(f"  HR range: {sample_data['heart_rate'].min():.0f} - {sample_data['heart_rate'].max():.0f} BPM")
    
    # Run prediction
    print("\nRunning prediction...")
    prediction = predict_heart_rate(model, test_data, args.sample_idx, device)
    
    # Calculate final MAE
    mae = np.mean(np.abs(prediction - sample_data['heart_rate']))
    print(f"  Prediction MAE: {mae:.2f} BPM")
    
    # Create animator and save GIF
    print("\n" + "=" * 70)
    print("CREATING ANIMATION")
    print("=" * 70)
    
    animator = PredictionAnimator(sample_data, prediction, scaler_params)
    output_path = animator.save_gif(args.output, dpi=args.dpi)
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)
    print(f"Animation saved to: {output_path}")


if __name__ == '__main__':
    main()
