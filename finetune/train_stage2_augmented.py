#!/usr/bin/env python3
"""
Stage 2 Fine-tuning with Data Augmentation

Uses augmentation to reduce overfitting on small Apple Watch dataset
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from finetune.config import get_stage2_config, print_config
from finetune.model import load_pretrained_model
from finetune.dataset import create_dataloaders
from finetune.augmentation import TimeSeriesAugmenter, AugmentedDataset


class MaskedMSELoss(nn.Module):
    """MSE Loss with masking for padded sequences"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target, mask):
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()


def train_epoch(model, train_loader, criterion, optimizer, device, gradient_clip):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    for features, target, mask in train_loader:
        features = features.to(device)
        target = target.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, target, mask)
        loss.backward()
        
        if gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for features, target, mask in val_loader:
            features = features.to(device)
            target = target.to(device)
            mask = mask.to(device)
            
            output = model(features)
            loss = criterion(output, target, mask)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main():
    """Main training loop"""
    print("="*80)
    print("STAGE 2 FINE-TUNING WITH DATA AUGMENTATION")
    print("="*80)
    
    # Get configuration
    config = get_stage2_config()
    
    # Add augmentation settings
    config['aug_methods'] = ['time_warp', 'magnitude_warp', 'jitter']
    config['aug_prob'] = 0.5
    config['aug_multiplier'] = 3
    config['checkpoint_dir'] = 'checkpoints/stage2_aug'
    config['results_dir'] = 'results/stage2_aug'
    
    print_config(config, stage=2)
    
    print("\n=== Augmentation ===")
    print(f"Methods: {', '.join(config['aug_methods'])}")
    print(f"Probability: {config['aug_prob']}")
    print(f"Multiplier: {config['aug_multiplier']}x")
    print(f"Original samples: 196 → Augmented: {196 * config['aug_multiplier']}")
    
    # Setup
    device = torch.device(config['device'])
    Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['results_dir']).mkdir(parents=True, exist_ok=True)
    
    # Load pretrained model
    print("\n1. Loading pretrained model...")
    model, pretrained_checkpoint = load_pretrained_model(
        config['pretrained_model'],
        device=str(device)
    )
    
    # Unfreeze all layers (Stage 2)
    print("\n2. Unfreezing all layers...")
    for param in model.parameters():
        param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create dataloaders WITH AUGMENTATION
    print("\n3. Creating dataloaders with augmentation...")
    base_train_loader, val_loader, test_loader = create_dataloaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        use_mask=config['use_mask']
    )
    
    # Wrap training dataset with augmentation
    base_train_dataset = base_train_loader.dataset
    aug_config = {'window_size': 400}
    augmenter = TimeSeriesAugmenter(aug_config)
    
    augmented_train_dataset = AugmentedDataset(
        base_train_dataset,
        augmenter,
        methods=config['aug_methods'],
        prob=config['aug_prob'],
        multiplier=config['aug_multiplier']
    )
    
    train_loader = DataLoader(
        augmented_train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    print(f"Original train samples: {len(base_train_dataset)}")
    print(f"Augmented train samples: {len(augmented_train_dataset)} ({config['aug_multiplier']}x)")
    print(f"Train batches: {len(train_loader)}")
    
    # Setup training
    criterion = MaskedMSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config['scheduler_factor'],
        patience=config['scheduler_patience'],
        min_lr=config['scheduler_min_lr']
    )
    
    # Training loop
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': [],
        'epoch_time': []
    }
    
    for epoch in range(config['epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, 
            device, config['gradient_clip']
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Scheduler step
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        epoch_time = time.time() - start_time
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rate'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            best_checkpoint = Path(config['checkpoint_dir']) / f'best_model_{timestamp}.pt'
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'history': history,
                'config': config,
                'best_val_loss': best_val_loss
            }, best_checkpoint)
            
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"\nEarly stopping - no improvement for {config['patience']} epochs")
                break
    
    # Save training history
    history_file = Path(config['results_dir']) / 'training_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (MSE)')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(history['learning_rate'], linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    curves_file = Path(config['results_dir']) / 'training_curves.png'
    plt.savefig(curves_file, dpi=300, bbox_inches='tight')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {config['checkpoint_dir']}")
    print(f"Results: {config['results_dir']}")
    print(f"\nTo evaluate: python3 Model/evaluate_finetuned.py")


if __name__ == "__main__":
    main()
