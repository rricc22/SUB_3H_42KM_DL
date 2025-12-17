"""
Stage 1 Fine-Tuning Training Script
Freeze layers 0, 1, 2 and train only layer 3 + FC
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
import json
import time
from datetime import datetime
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from finetune.model import HeartRateLSTM, load_pretrained_model
from finetune.dataset import create_dataloaders
from finetune.config import get_stage1_config, print_config


class MaskedMSELoss(nn.Module):
    """MSE Loss with masking for padded sequences"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target, mask):
        """
        Args:
            pred: (batch, seq_len, 1)
            target: (batch, seq_len, 1)
            mask: (batch, seq_len, 1) - 1 for valid, 0 for padding
        """
        loss = self.mse(pred, target)
        masked_loss = loss * mask
        return masked_loss.sum() / mask.sum()


class Trainer:
    """Training manager for Stage 1 fine-tuning"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        # Create output directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.results_dir = Path(config['results_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_time = None
        
        print(f"\n{'='*60}")
        print(f"STAGE 1 FINE-TUNING TRAINER INITIALIZED")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Checkpoint dir: {self.checkpoint_dir}")
        print(f"Results dir: {self.results_dir}")
    
    def setup(self):
        """Setup model, data, optimizer, scheduler"""
        print(f"\n{'='*60}")
        print("SETUP")
        print(f"{'='*60}")
        
        # Load pretrained model
        print("\n1. Loading pretrained model...")
        self.model, self.pretrained_checkpoint = load_pretrained_model(
            self.config['pretrained_model'],
            device=self.device
        )
        
        # Freeze layers
        print("\n2. Freezing layers...")
        self.model.freeze_layers(self.config['freeze_layers'])
        self.model.print_parameter_status()
        
        # Create dataloaders
        print("\n3. Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            self.config['data_dir'],
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            use_mask=self.config['use_mask']
        )
        
        # Loss function
        print("\n4. Setting up loss function...")
        self.criterion = MaskedMSELoss()
        print("Using MaskedMSELoss (handles padded sequences)")
        
        # Optimizer (only trainable parameters)
        print("\n5. Setting up optimizer...")
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.Adam(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        print(f"Optimizer: Adam")
        print(f"Learning rate: {self.config['learning_rate']}")
        print(f"Weight decay: {self.config['weight_decay']}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        # Scheduler
        print("\n6. Setting up scheduler...")
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config['scheduler_factor'],
            patience=self.config['scheduler_patience'],
            min_lr=self.config['scheduler_min_lr'],
            verbose=True
        )
        print(f"Scheduler: ReduceLROnPlateau")
        print(f"Factor: {self.config['scheduler_factor']}")
        print(f"Patience: {self.config['scheduler_patience']}")
        
        print(f"\n{'='*60}")
        print("SETUP COMPLETE")
        print(f"{'='*60}\n")
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, (features, target, mask) in enumerate(self.train_loader):
            # Move to device
            features = features.to(self.device)
            target = target.to(self.device)
            mask = mask.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(features)
            loss = self.criterion(output, target, mask)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['gradient_clip']
            )
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
            
            # Logging
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"  Batch [{batch_idx+1}/{num_batches}] - Loss: {avg_loss:.4f}")
        
        return epoch_loss / num_batches
    
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        val_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for features, target, mask in self.val_loader:
                features = features.to(self.device)
                target = target.to(self.device)
                mask = mask.to(self.device)
                
                output = self.model(features)
                loss = self.criterion(output, target, mask)
                val_loss += loss.item()
        
        return val_loss / num_batches
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint (overwrites previous checkpoints)"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        # Always save as latest_model.pt (overwrites previous)
        latest_path = self.checkpoint_dir / 'latest_model.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best model (overwrites previous best)
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def save_history(self):
        """Save training history"""
        history_path = self.results_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"  Saved training history to: {history_path}")
    
    def plot_training_curves(self):
        """Generate and save training curve plots"""
        print("\n" + "="*60)
        print("GENERATING TRAINING CURVES")
        print("="*60)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Loss curves
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Mark best epoch
        best_epoch = self.history['val_loss'].index(min(self.history['val_loss'])) + 1
        best_val_loss = min(self.history['val_loss'])
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (Epoch {best_epoch})')
        axes[0].legend()
        
        # Learning rate
        axes[1].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_title('Learning Rate Schedule')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        save_path = self.results_dir / f'training_curves_{timestamp}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved training curves to: {save_path}")
        plt.close()
        
        return save_path
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print("STARTING TRAINING")
        print(f"{'='*60}\n")
        
        self.start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch}/{self.config['epochs']}")
            print("-" * 40)
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_loss = self.validate()
            
            # Scheduler step
            self.scheduler.step(val_loss)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Time
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_time'].append(epoch_time)
            
            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Check for best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config['patience']:
                print(f"\n{'='*60}")
                print(f"EARLY STOPPING - No improvement for {self.config['patience']} epochs")
                print(f"{'='*60}")
                break
        
        # Training complete
        total_time = time.time() - self.start_time
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save final history
        self.save_history()
        
        # Generate training curves
        self.plot_training_curves()
        
        return self.history


def main():
    """Main training function"""
    # Get configuration
    config = get_stage1_config()
    print_config(config)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Setup
    trainer.setup()
    
    # Train
    history = trainer.train()
    
    print(f"\n{'='*60}")
    print("STAGE 1 FINE-TUNING FINISHED")
    print(f"{'='*60}")
    print(f"\nCheckpoints saved in: {config['checkpoint_dir']}")
    print(f"Results saved in: {config['results_dir']}")
    print(f"\nBest model: {config['checkpoint_dir']}/best_model.pt")
    print(f"Val loss: {trainer.best_val_loss:.4f}")
    print(f"\nTo evaluate: python finetune/evaluate.py")


if __name__ == "__main__":
    main()
