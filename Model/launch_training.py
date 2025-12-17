#!/usr/bin/env python3
"""
Launch script for Stage 1 training with logging
Redirects all output to a timestamped log file while also showing it on screen
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path


def main():
    """Launch training with logging"""
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Generate timestamp for log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'training_stage1_{timestamp}.log'
    
    print(f"{'='*60}")
    print(f"LAUNCHING STAGE 1 TRAINING")
    print(f"{'='*60}")
    print(f"Log file: {log_file}")
    print(f"{'='*60}\n")
    
    # Run training script using tee to show output and save to file
    # Use unbuffered output (-u) for real-time logging
    cmd = f'python -u finetune/train_stage1.py 2>&1 | tee "{log_file}"'
    
    try:
        # Run with shell to enable pipe and tee
        process = subprocess.run(
            cmd,
            shell=True,
            executable='/bin/bash'
        )
        
        print(f"\n{'='*60}")
        print(f"TRAINING FINISHED")
        print(f"{'='*60}")
        print(f"Log saved to: {log_file}")
        print(f"Exit code: {process.returncode}")
        
        return process.returncode
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print(f"Partial log saved to: {log_file}")
        return 1
    except Exception as e:
        print(f"\nError launching training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
