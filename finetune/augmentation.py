#!/usr/bin/env python3
"""
Data Augmentation for Heart Rate Time-Series

Techniques:
1. Time Warping - Stretch/compress time
2. Magnitude Warping - Scale intensity
3. Window Slicing - Extract random windows
4. Jittering - Add sensor noise
5. Rotation - Rotate features
6. Mixup - Interpolate between samples
"""

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


class TimeSeriesAugmenter:
    """Augmentation for heart rate prediction from speed/altitude"""
    
    def __init__(self, config):
        self.config = config
        self.methods = {
            'time_warp': self.time_warp,
            'magnitude_warp': self.magnitude_warp,
            'window_slice': self.window_slice,
            'jitter': self.jitter,
            'rotation': self.rotation,
            'mixup': self.mixup,
        }
    
    def augment(self, features, target, mask, methods=None, prob=0.5):
        """
        Apply random augmentation
        
        Args:
            features: [seq_len, 3] - speed, altitude, hr_prev
            target: [seq_len, 1] - heart rate
            mask: [seq_len, 1] - valid timesteps
            methods: list of method names (default: all)
            prob: probability of applying each method
        
        Returns:
            Augmented features, target, mask
        """
        if methods is None:
            methods = ['time_warp', 'magnitude_warp', 'jitter']
        
        # Convert to numpy
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        target_np = target.cpu().numpy() if torch.is_tensor(target) else target
        mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else mask
        
        # Extract components
        speed = features_np[:, 0]
        altitude = features_np[:, 1]
        hr = target_np.squeeze()
        mask_1d = mask_np.squeeze()
        
        # Apply augmentations with probability
        for method_name in methods:
            if np.random.rand() < prob:
                method = self.methods[method_name]
                speed, altitude, hr = method(speed, altitude, hr, mask_1d)
        
        # Reconstruct
        features_aug = np.stack([speed, altitude, hr], axis=-1)
        target_aug = hr.reshape(-1, 1)
        
        # Convert back to tensor
        if torch.is_tensor(features):
            features_aug = torch.from_numpy(features_aug).float()
            target_aug = torch.from_numpy(target_aug).float()
        
        return features_aug, target_aug, mask
    
    def time_warp(self, speed, altitude, hr, mask, warp_factor=0.15):
        """
        Time warping: stretch or compress time
        
        Args:
            warp_factor: 0.15 means ±15% time change
        """
        seq_len = len(speed)
        valid_len = int(mask.sum())
        
        if valid_len < 10:
            return speed, altitude, hr
        
        # Random warp factor
        factor = np.random.uniform(1-warp_factor, 1+warp_factor)
        
        # New time axis
        old_time = np.arange(valid_len)
        new_len = int(valid_len * factor)
        new_len = max(10, min(new_len, seq_len))  # Clamp to valid range
        new_time = np.linspace(0, valid_len-1, new_len)
        
        try:
            # Interpolate
            speed_interp = interp1d(old_time, speed[:valid_len], kind='linear', fill_value='extrapolate')
            altitude_interp = interp1d(old_time, altitude[:valid_len], kind='linear', fill_value='extrapolate')
            hr_interp = interp1d(old_time, hr[:valid_len], kind='linear', fill_value='extrapolate')
            
            speed_warped = speed_interp(new_time)
            altitude_warped = altitude_interp(new_time)
            hr_warped = hr_interp(new_time)
            
            # Pad or truncate to original length
            if new_len > seq_len:
                speed_warped = speed_warped[:seq_len]
                altitude_warped = altitude_warped[:seq_len]
                hr_warped = hr_warped[:seq_len]
            else:
                pad_len = seq_len - new_len
                speed_warped = np.pad(speed_warped, (0, pad_len), mode='constant')
                altitude_warped = np.pad(altitude_warped, (0, pad_len), mode='constant')
                hr_warped = np.pad(hr_warped, (0, pad_len), mode='constant')
            
            return speed_warped, altitude_warped, hr_warped
        except:
            # If interpolation fails, return original
            return speed, altitude, hr
    
    def magnitude_warp(self, speed, altitude, hr, mask, sigma=0.2):
        """
        Magnitude warping: multiply by smooth random curve
        
        Args:
            sigma: standard deviation of warping magnitude
        """
        valid_len = int(mask.sum())
        
        if valid_len < 10:
            return speed, altitude, hr
        
        # Generate smooth random curve
        knots = max(4, valid_len // 100)  # Adaptive number of knots
        random_curve = np.random.normal(1.0, sigma, knots)
        
        # Interpolate to full length
        curve_interp = interp1d(
            np.linspace(0, valid_len-1, knots),
            random_curve,
            kind='cubic',
            fill_value='extrapolate'
        )
        full_curve = curve_interp(np.arange(valid_len))
        
        # Smooth the curve
        full_curve = gaussian_filter1d(full_curve, sigma=max(1, valid_len//20))
        
        # Clip to reasonable range (±30%)
        full_curve = np.clip(full_curve, 0.7, 1.3)
        
        # Apply to speed and HR (altitude is terrain, shouldn't change much)
        speed_warped = speed.copy()
        hr_warped = hr.copy()
        
        speed_warped[:valid_len] *= full_curve
        
        # HR scales sublinearly with speed (physiological model)
        # HR ~ speed^0.5 approximately
        hr_warped[:valid_len] *= full_curve ** 0.5
        
        return speed_warped, altitude, hr_warped
    
    def window_slice(self, speed, altitude, hr, mask):
        """
        Window slicing: extract random window
        Note: Returns only one window (caller should loop for multiple)
        """
        seq_len = len(speed)
        valid_len = int(mask.sum())
        window_size = self.config.get('window_size', 400)
        
        if valid_len < window_size:
            return speed, altitude, hr
        
        # Random start position
        start = np.random.randint(0, valid_len - window_size + 1)
        end = start + window_size
        
        # Extract window
        speed_win = np.zeros_like(speed)
        altitude_win = np.zeros_like(altitude)
        hr_win = np.zeros_like(hr)
        
        speed_win[:window_size] = speed[start:end]
        altitude_win[:window_size] = altitude[start:end]
        hr_win[:window_size] = hr[start:end]
        
        return speed_win, altitude_win, hr_win
    
    def jitter(self, speed, altitude, hr, mask, noise_level=0.03):
        """
        Jittering: add small Gaussian noise
        
        Args:
            noise_level: fraction of signal std (3% default)
        """
        valid_len = int(mask.sum())
        
        if valid_len < 10:
            return speed, altitude, hr
        
        # Calculate noise magnitude from valid region
        speed_noise = np.std(speed[:valid_len]) * noise_level
        altitude_noise = np.std(altitude[:valid_len]) * noise_level
        
        # Add noise only to valid region
        speed_jittered = speed.copy()
        altitude_jittered = altitude.copy()
        
        speed_jittered[:valid_len] += np.random.normal(0, speed_noise, valid_len)
        altitude_jittered[:valid_len] += np.random.normal(0, altitude_noise, valid_len)
        
        # Ensure non-negative (speed can't be negative)
        speed_jittered = np.maximum(speed_jittered, 0)
        
        return speed_jittered, altitude_jittered, hr
    
    def rotation(self, speed, altitude, hr, mask, angle_range=10):
        """
        Rotation: rotate speed-altitude in 2D space
        
        Args:
            angle_range: max rotation angle in degrees
        """
        valid_len = int(mask.sum())
        
        if valid_len < 10:
            return speed, altitude, hr
        
        # Random angle
        angle = np.random.uniform(-angle_range, angle_range) * np.pi / 180
        
        # Normalize features (z-score)
        speed_valid = speed[:valid_len]
        altitude_valid = altitude[:valid_len]
        
        speed_mean, speed_std = np.mean(speed_valid), np.std(speed_valid) + 1e-8
        altitude_mean, altitude_std = np.mean(altitude_valid), np.std(altitude_valid) + 1e-8
        
        speed_norm = (speed_valid - speed_mean) / speed_std
        altitude_norm = (altitude_valid - altitude_mean) / altitude_std
        
        # Rotation matrix
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        speed_rot = cos_a * speed_norm - sin_a * altitude_norm
        altitude_rot = sin_a * speed_norm + cos_a * altitude_norm
        
        # Denormalize
        speed_rot = speed_rot * speed_std + speed_mean
        altitude_rot = altitude_rot * altitude_std + altitude_mean
        
        # Put back in array
        speed_rotated = speed.copy()
        altitude_rotated = altitude.copy()
        speed_rotated[:valid_len] = speed_rot
        altitude_rotated[:valid_len] = altitude_rot
        
        return speed_rotated, altitude_rotated, hr
    
    def mixup(self, speed, altitude, hr, mask, speed2=None, altitude2=None, hr2=None, alpha=0.2):
        """
        Mixup: interpolate between two samples
        
        Args:
            alpha: Beta distribution parameter (0.2 = subtle mixing)
        """
        if speed2 is None:
            # Can't mixup without second sample
            return speed, altitude, hr
        
        # Sample mixing coefficient from Beta distribution
        lam = np.random.beta(alpha, alpha)
        
        # Mix features
        speed_mixed = lam * speed + (1 - lam) * speed2
        altitude_mixed = lam * altitude + (1 - lam) * altitude2
        hr_mixed = lam * hr + (1 - lam) * hr2
        
        return speed_mixed, altitude_mixed, hr_mixed


# Augmented dataset wrapper
class AugmentedDataset(torch.utils.data.Dataset):
    """Dataset wrapper with on-the-fly augmentation"""
    
    def __init__(self, base_dataset, augmenter, methods=None, prob=0.5, multiplier=1):
        """
        Args:
            base_dataset: Original dataset
            augmenter: TimeSeriesAugmenter instance
            methods: List of augmentation methods to use
            prob: Probability of applying each method
            multiplier: How many augmented versions per original sample
        """
        self.base_dataset = base_dataset
        self.augmenter = augmenter
        self.methods = methods or ['time_warp', 'magnitude_warp', 'jitter']
        self.prob = prob
        self.multiplier = multiplier
    
    def __len__(self):
        return len(self.base_dataset) * self.multiplier
    
    def __getitem__(self, idx):
        # Map to base dataset index
        base_idx = idx % len(self.base_dataset)
        features, target, mask = self.base_dataset[base_idx]
        
        # Apply augmentation (except for first occurrence of each sample)
        if idx >= len(self.base_dataset):
            features, target, mask = self.augmenter.augment(
                features, target, mask,
                methods=self.methods,
                prob=self.prob
            )
        
        return features, target, mask


if __name__ == "__main__":
    # Test augmentation
    print("Testing Time-Series Augmentation")
    print("="*80)
    
    # Create dummy data
    seq_len = 500
    speed = np.random.rand(seq_len) * 3 + 2  # 2-5 m/s
    altitude = np.random.randn(seq_len).cumsum() * 2  # Cumulative altitude
    hr = 120 + speed * 10 + np.random.randn(seq_len) * 5  # HR correlated with speed
    mask = np.ones(seq_len)
    mask[400:] = 0  # Padding after 400 steps
    
    # Create augmenter
    config = {'window_size': 400}
    augmenter = TimeSeriesAugmenter(config)
    
    # Test each method
    methods = ['time_warp', 'magnitude_warp', 'jitter', 'rotation']
    
    for method in methods:
        print(f"\nTesting {method}:")
        speed_aug, altitude_aug, hr_aug = augmenter.methods[method](
            speed.copy(), altitude.copy(), hr.copy(), mask
        )
        
        # Check statistics
        valid_len = int(mask.sum())
        print(f"  Original speed: mean={speed[:valid_len].mean():.3f}, std={speed[:valid_len].std():.3f}")
        print(f"  Augmented speed: mean={speed_aug[:valid_len].mean():.3f}, std={speed_aug[:valid_len].std():.3f}")
        print(f"  Original HR: mean={hr[:valid_len].mean():.3f}, std={hr[:valid_len].std():.3f}")
        print(f"  Augmented HR: mean={hr_aug[:valid_len].mean():.3f}, std={hr_aug[:valid_len].std():.3f}")
    
    print("\n" + "="*80)
    print("✓ Augmentation tests complete!")
