"""
Core utilities for loading and managing model checkpoints.

This module provides the CheckpointLoader class for loading saved model checkpoints
and extracting model information.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.modelmodule import INRTraining, OCINRTraining


class CheckpointLoader:
    """Utility class for loading and analyzing model checkpoints."""
    
    def __init__(self, log_dir: str):
        """
        Initialize checkpoint loader.
        
        Args:
            log_dir: Path to the log directory containing checkpoints
        """
        self.log_dir = Path(log_dir)
        self.checkpoints = self._find_checkpoints()
        
    def _find_checkpoints(self) -> Dict[int, Path]:
        """Find all epoch checkpoints in the log directory."""
        checkpoints = {}
        
        # Find checkpoint_epoch_*.ckpt files
        for ckpt_file in self.log_dir.glob("checkpoint_epoch_*.ckpt"):
            # Extract epoch number from filename
            epoch_str = ckpt_file.stem.split('_')[-1]
            try:
                epoch = int(epoch_str)
                checkpoints[epoch] = ckpt_file
            except ValueError:
                print(f"Warning: Could not parse epoch from {ckpt_file}")
                
        return checkpoints
    
    def list_checkpoints(self) -> List[int]:
        """List all available checkpoint epochs."""
        return sorted(self.checkpoints.keys())
    
    def load_checkpoint(self, epoch: int, model_class=INRTraining) -> Tuple[torch.nn.Module, Dict]:
        """
        Load a checkpoint from a specific epoch.
        
        Args:
            epoch: Epoch number to load
            model_class: Model class (INRTraining or OCINRTraining)
            
        Returns:
            Tuple of (model, checkpoint_dict)
        """
        if epoch not in self.checkpoints:
            raise ValueError(
                f"Checkpoint for epoch {epoch} not found. "
                f"Available epochs: {self.list_checkpoints()}"
            )
        
        ckpt_path = self.checkpoints[epoch]
        
        # Load the checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        
        # Load the Lightning module (this will automatically restore the model)
        lightning_module = model_class.load_from_checkpoint(str(ckpt_path))
        
        # Extract the actual model
        model = lightning_module.model
        
        return model, checkpoint
    
    def get_checkpoint_info(self, epoch: int) -> Dict:
        """Get information about a checkpoint without loading the full model."""
        if epoch not in self.checkpoints:
            raise ValueError(f"Checkpoint for epoch {epoch} not found.")
        
        checkpoint = torch.load(self.checkpoints[epoch], map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 'unknown'),
            'global_step': checkpoint.get('global_step', 'unknown'),
            'pytorch-lightning_version': checkpoint.get('pytorch-lightning_version', 'unknown'),
            'file_path': str(self.checkpoints[epoch]),
            'file_size_mb': self.checkpoints[epoch].stat().st_size / (1024 * 1024),
        }
        
        return info
    
    def get_model_from_checkpoint(self, epoch: int, model_class=INRTraining) -> torch.nn.Module:
        """
        Load and return only the model (not the full Lightning module).
        
        Args:
            epoch: Epoch number
            model_class: Model class
            
        Returns:
            The model
        """
        model, _ = self.load_checkpoint(epoch, model_class=model_class)
        return model


if __name__ == "__main__":
    # Example usage
    log_dir = "/home/exx/Projects/OCINR/logs/2025-10-09_12-45-01"
    
    print("="*60)
    print("CHECKPOINT LOADER - EXAMPLE USAGE")
    print("="*60)
    
    # Create loader
    loader = CheckpointLoader(log_dir)
    
    # List available checkpoints
    epochs = loader.list_checkpoints()
    print(f"\nAvailable checkpoints: {epochs}")
    
    # Get info about each checkpoint
    print("\nCheckpoint Information:")
    print("-"*60)
    for epoch in epochs:
        info = loader.get_checkpoint_info(epoch)
        print(f"\nEpoch {epoch}:")
        print(f"  Global Step: {info['global_step']}")
        print(f"  File Size: {info['file_size_mb']:.2f} MB")
    
    # Load a specific checkpoint
    print("\n" + "="*60)
    print("Loading Epoch 100...")
    print("="*60)
    
    model, checkpoint = loader.load_checkpoint(100, model_class=INRTraining)
    
    print(f"\nModel type: {type(model).__name__}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Show model components
    print(f"\nModel components:")
    if hasattr(model, 'input_proj'):
        print(f"  ✓ input_proj")
    if hasattr(model, 'hidden_blocks'):
        print(f"  ✓ hidden_blocks (count: {len(model.hidden_blocks)})")
    if hasattr(model, 'output_proj'):
        print(f"  ✓ output_proj")
    
    print("\n✓ Example complete!")
