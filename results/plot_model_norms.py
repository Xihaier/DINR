"""
Compare parameter norms between different models (FFNet vs OCFFNet).

This script calculates and compares norms for:
- FFNet: input_proj, hidden_blocks, output_proj
- OCFFNet: input_proj, ode_func, output_proj
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import scienceplots

# Add src to path
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
        
        # Load the checkpoint dictionary
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        
        # Extract the model state dict (which has "model." prefix)
        state_dict = checkpoint['state_dict']
        
        # Remove the "model." prefix from keys to get the actual model state dict
        model_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('model.'):
                # Remove "model." prefix
                new_key = key[6:]  # len("model.") = 6
                model_state_dict[new_key] = value
        
        # Reconstruct the network from hyperparameters in checkpoint
        try:
            # Get hyperparameters from checkpoint
            hparams = checkpoint.get('hyper_parameters', {})
            
            # Check if 'net' config exists in hyperparameters
            if 'net' not in hparams or hparams['net'] is None:
                raise ValueError("Network configuration not found in checkpoint hyperparameters")
            
            # Instantiate the network from hyperparameters
            from hydra.utils import instantiate
            net = instantiate(hparams['net'])
            
            # Load the model state dict into the network
            net.load_state_dict(model_state_dict, strict=True)
            
            model = net
            
        except Exception as e:
            print(f"Warning: Could not reconstruct from hyperparameters: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: Try to instantiate directly from the state dict structure
            # Import the model class based on what we find in the state dict
            try:
                from src.models.components.FFNet import FourierFeatureNetwork
                from src.models.components.OC_FFNet import OCFourierFeatureNetwork
                
                # Detect which model type based on keys in state dict
                is_ocffnet = any('ode_func' in k for k in model_state_dict.keys())
                
                print(f"   Detected model type: {'OCFFNet' if is_ocffnet else 'FFNet'}")
                print(f"   Sample keys: {list(model_state_dict.keys())[:5]}")
                
                if any('fourier_features' in k for k in model_state_dict.keys()):
                    # It's an FFNet or OCFFNet
                    # Try to infer parameters from the state dict
                    
                    # Get dimensions from the actual weights
                    has_input_proj = ('input_proj.0.weight' in model_state_dict or 
                                     'input_proj.weight' in model_state_dict or
                                     'input_proj.bias' in model_state_dict)
                    
                    if has_input_proj or 'fourier_features.B' in model_state_dict:
                        mapping_size = model_state_dict['fourier_features.B'].shape[1] * 2
                        input_dim = model_state_dict['fourier_features.B'].shape[0]
                        
                        # Get hidden_dim and output_dim
                        if 'input_proj.weight' in model_state_dict:
                            # input_proj is nn.Linear
                            hidden_dim = model_state_dict['input_proj.weight'].shape[0]
                        elif 'input_proj.0.weight' in model_state_dict:
                            # input_proj is nn.Sequential with Linear
                            hidden_dim = model_state_dict['input_proj.0.weight'].shape[0]
                        else:
                            hidden_dim = mapping_size
                        
                        # Get output dim
                        output_keys = [k for k in model_state_dict.keys() if 'output_proj' in k and 'weight' in k]
                        if output_keys:
                            # Find the final linear layer
                            for key in sorted(output_keys):
                                if '.weight' in key and not '.norm' in key:
                                    output_dim = model_state_dict[key].shape[0]
                        else:
                            output_dim = 1
                        
                        if is_ocffnet:
                            # It's OCFFNet - count ODE function layers
                            # Look for patterns like: ode_func.layers.0.linear1.weight, ode_func.layers.1.linear1.weight, etc.
                            ode_layer_indices = set()
                            for k in model_state_dict.keys():
                                if k.startswith('ode_func.layers.'):
                                    # Extract layer index
                                    parts = k.split('.')
                                    if len(parts) >= 3 and parts[2].isdigit():
                                        ode_layer_indices.add(int(parts[2]))
                            
                            num_layers = len(ode_layer_indices) if ode_layer_indices else 3
                            print(f"   Detected {num_layers} ODE function layers")
                            
                            # Create OCFFNet
                            model = OCFourierFeatureNetwork(
                                input_dim=input_dim,
                                mapping_size=mapping_size,
                                hidden_dim=hidden_dim,
                                output_dim=output_dim,
                                num_layers=num_layers,
                                dropout_rate=0.0,  # Set to 0 for inference
                                activation="GELU",
                                block_type="residual",
                                num_steps=10,
                                total_time=1.0,
                                ot_lambda=0.001,
                                sigma=10.0,
                                final_activation=None
                            )
                        else:
                            # It's FFNet - count hidden blocks
                            hidden_block_indices = set()
                            for k in model_state_dict.keys():
                                if k.startswith('hidden_blocks.'):
                                    parts = k.split('.')
                                    if len(parts) >= 2 and parts[1].isdigit():
                                        hidden_block_indices.add(int(parts[1]))
                            
                            num_layers = len(hidden_block_indices) if hidden_block_indices else 5
                            print(f"   Detected {num_layers} hidden blocks")
                            
                            # Create FFNet
                            model = FourierFeatureNetwork(
                                input_dim=input_dim,
                                mapping_size=mapping_size,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers,
                                output_dim=output_dim,
                                sigma=10.0,  # Default value
                                dropout_rate=0.0,  # Set to 0 for inference
                                activation="GELU",
                                final_activation=None,
                                use_residual=True
                            )
                        
                        print(f"   Created model with:")
                        print(f"     input_dim={input_dim}, mapping_size={mapping_size}")
                        print(f"     hidden_dim={hidden_dim}, num_layers={num_layers}")
                        print(f"     output_dim={output_dim}")
                        
                        # Load the state dict
                        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
                        if missing:
                            print(f"   Warning: Missing keys: {len(missing)}")
                        if unexpected:
                            print(f"   Warning: Unexpected keys: {len(unexpected)}")
                        
                    else:
                        print(f"   Error: No input_proj or fourier_features found")
                        print(f"   Available keys: {list(model_state_dict.keys())[:10]}")
                        raise ValueError("Could not infer model architecture from state dict")
                else:
                    raise ValueError("Unknown model type")
                    
            except Exception as e2:
                raise RuntimeError(
                    f"Could not load checkpoint. Tried multiple methods.\n"
                    f"First error: {e}\n"
                    f"Second error: {e2}\n"
                    f"Please ensure the model architecture matches the checkpoint."
                )
        
        return model, checkpoint
    
    def get_checkpoint_info(self, epoch: int) -> Dict:
        """Get information about a checkpoint without loading the full model."""
        if epoch not in self.checkpoints:
            raise ValueError(f"Checkpoint for epoch {epoch} not found.")
        
        checkpoint = torch.load(self.checkpoints[epoch], map_location='cpu', weights_only=False)
        
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


def detect_model_type(model: torch.nn.Module) -> str:
    """
    Detect the model type based on its attributes.
    
    Args:
        model: The neural network model
        
    Returns:
        Model type: 'FFNet', 'OCFFNet', or 'Unknown'
    """
    if hasattr(model, 'ode_func'):
        return 'OCFFNet'
    elif hasattr(model, 'hidden_blocks'):
        return 'FFNet'
    else:
        return 'Unknown'


def calculate_parameter_norms(model: torch.nn.Module, norm_type: str = 'frobenius') -> Dict[str, float]:
    """
    Calculate norms of parameters for different model components.
    Automatically detects model type and calculates appropriate norms.
    
    Args:
        model: The neural network model (FFNet or OCFFNet)
        norm_type: Type of norm to calculate ('frobenius', 'l1', 'l2', 'nuclear', 'spectral')
    
    Returns:
        Dictionary with norms for each component
    """
    norms = {}
    model_type = detect_model_type(model)
    norms['model_type'] = model_type
    
    # Define norm calculation function
    def compute_norm(param_tensor: torch.Tensor, norm_type: str) -> float:
        """Compute norm based on type."""
        if norm_type == 'frobenius':
            return torch.norm(param_tensor, p='fro').item()
        elif norm_type == 'l1':
            return torch.norm(param_tensor, p=1).item()
        elif norm_type == 'l2':
            return torch.norm(param_tensor, p=2).item()
        elif norm_type == 'nuclear':
            if param_tensor.ndim == 2:
                return torch.norm(param_tensor, p='nuc').item()
            else:
                return torch.norm(param_tensor.flatten(), p=2).item()
        elif norm_type == 'spectral':
            if param_tensor.ndim == 2:
                return torch.linalg.matrix_norm(param_tensor, ord=2).item()
            else:
                return torch.norm(param_tensor.flatten(), p=2).item()
        else:
            raise ValueError(f"Unknown norm type: {norm_type}")
    
    # 1. Input projection norms
    if hasattr(model, 'input_proj'):
        input_proj_norm = 0.0
        param_count = 0
        for name, param in model.input_proj.named_parameters():
            if param.requires_grad:
                input_proj_norm += compute_norm(param, norm_type)
                param_count += 1
        norms['input_proj'] = input_proj_norm
        norms['input_proj_avg'] = input_proj_norm / param_count if param_count > 0 else 0.0
    
    # 2. Main blocks norms (different for FFNet vs OCFFNet)
    if model_type == 'FFNet' and hasattr(model, 'hidden_blocks'):
        # FFNet: hidden_blocks
        hidden_blocks_norm = 0.0
        param_count = 0
        for block in model.hidden_blocks:
            for name, param in block.named_parameters():
                if param.requires_grad:
                    hidden_blocks_norm += compute_norm(param, norm_type)
                    param_count += 1
        norms['main_blocks'] = hidden_blocks_norm
        norms['main_blocks_avg'] = hidden_blocks_norm / param_count if param_count > 0 else 0.0
        norms['main_blocks_name'] = 'hidden_blocks'
        
    elif model_type == 'OCFFNet' and hasattr(model, 'ode_func'):
        # OCFFNet: ode_func
        ode_func_norm = 0.0
        param_count = 0
        for name, param in model.ode_func.named_parameters():
            if param.requires_grad:
                ode_func_norm += compute_norm(param, norm_type)
                param_count += 1
        norms['main_blocks'] = ode_func_norm
        norms['main_blocks_avg'] = ode_func_norm / param_count if param_count > 0 else 0.0
        norms['main_blocks_name'] = 'ode_func'
    
    # 3. Output projection norms
    if hasattr(model, 'output_proj'):
        output_proj_norm = 0.0
        param_count = 0
        for name, param in model.output_proj.named_parameters():
            if param.requires_grad:
                output_proj_norm += compute_norm(param, norm_type)
                param_count += 1
        norms['output_proj'] = output_proj_norm
        norms['output_proj_avg'] = output_proj_norm / param_count if param_count > 0 else 0.0
    
    # 4. Total model norm
    total_norm = 0.0
    for param in model.parameters():
        if param.requires_grad:
            total_norm += compute_norm(param, norm_type)
    norms['total'] = total_norm
    
    return norms


def compare_models_across_epochs(
    log_dirs: Dict[str, str],
    model_classes: Dict[str, type],
    epochs: List[int] = None,
    norm_types: List[str] = ['frobenius', 'l2', 'nuclear'],
    output_dir: str = None
) -> Dict[str, Dict[str, Dict[int, float]]]:
    """
    Compare parameter norms across multiple models and epochs.
    
    Args:
        log_dirs: Dictionary mapping model names to log directories
                  e.g., {'FFNet': 'logs/run1', 'OCFFNet': 'logs/run2'}
        model_classes: Dictionary mapping model names to their Lightning classes
                      e.g., {'FFNet': INRTraining, 'OCFFNet': OCINRTraining}
        epochs: List of epochs to analyze (None = all available)
        norm_types: List of norm types to calculate
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping model_name -> norm_type -> component -> {epoch: value}
    """
    if output_dir is None:
        output_dir = "."
    
    results = {}
    
    print("="*70)
    print(f"Models to compare: {list(log_dirs.keys())}")
    print(f"Norm types: {norm_types}\n")
    
    # Analyze each model
    for model_name, log_dir in log_dirs.items():
        print(f"\n{'='*70}")
        print(f"Analyzing {model_name}")
        print(f"{'='*70}")
        
        # Initialize checkpoint loader
        loader = CheckpointLoader(log_dir)
        available_epochs = loader.list_checkpoints()
        
        if epochs is None:
            model_epochs = available_epochs
        else:
            model_epochs = [e for e in epochs if e in available_epochs]
        
        print(f"Found {len(model_epochs)} checkpoints: {model_epochs}")
        
        # Initialize results for this model
        results[model_name] = {norm_type: {
            'input_proj': {},
            'main_blocks': {},
            'output_proj': {},
            'total': {}
        } for norm_type in norm_types}
        
        model_class = model_classes.get(model_name, INRTraining)
        
        # Analyze each checkpoint
        for epoch in model_epochs:
            print(f"  Processing epoch {epoch}...", end=' ')
            
            try:
                # Load model
                model, _ = loader.load_checkpoint(epoch, model_class=model_class)
                
                # Calculate norms for each norm type
                for norm_type in norm_types:
                    norms = calculate_parameter_norms(model, norm_type=norm_type)
                    
                    for component in ['input_proj', 'main_blocks', 'output_proj', 'total']:
                        if component in norms:
                            results[model_name][norm_type][component][epoch] = norms[component]
                
                print("✓")
                
            except Exception as e:
                print(f"✗ (Error: {e})")
                continue
    
    return results


def plot_model_comparison(
    results: Dict[str, Dict[str, Dict[int, float]]],
    norm_types: List[str],
    output_dir: str
):
    """
    Create comparison plots for different models.
    
    Args:
        results: Results from compare_models_across_epochs
        norm_types: List of norm types to plot
        output_dir: Directory to save plots
    """
    model_names = list(results.keys())
    
    # Define colors and markers for each model
    markers = {
        'FFNet': 'o', 
        'OCFFNet': 's',  # For backward compatibility
        'OCFFNet_without_OT': 's', 
        'OCFFNet_with_OT': '^'
    }
    
    with plt.style.context('science'):
        # Plot 1: Individual component plots (3 separate figures per norm type)
        for norm_type in norm_types:
            components = ['input_proj', 'main_blocks', 'output_proj']
            component_titles = ['Input Projection', 'Main Blocks', 'Output Projection']
            component_names = ['input_proj', 'main_blocks', 'output_proj']
            
            for component, title, comp_name in zip(components, component_titles, component_names):
                fig, ax = plt.subplots(1, 1, figsize=(5, 4))
                
                for model_name in model_names:
                    if component in results[model_name][norm_type]:
                        epochs = sorted(results[model_name][norm_type][component].keys())
                        values = [results[model_name][norm_type][component][e] for e in epochs]
                        
                        marker = markers.get(model_name, 'x')
                        
                        ax.plot(epochs, values, marker=marker, linewidth=2.5, 
                               markersize=8, label=model_name, alpha=0.8)
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(f'{norm_type.capitalize()} Norm')
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'comparison_{norm_type}_{comp_name}.png'), dpi=300)
                print(f"✓ Saved {norm_type} comparison for {comp_name}")
                plt.close()
        
        # Plot 2: Total norms comparison
        fig, axes = plt.subplots(1, len(norm_types), figsize=(5, 4))
        if len(norm_types) == 1:
            axes = [axes]
        
        for idx, norm_type in enumerate(norm_types):
            ax = axes[idx]
            
            for model_name in model_names:
                if 'total' in results[model_name][norm_type]:
                    epochs = sorted(results[model_name][norm_type]['total'].keys())
                    values = [results[model_name][norm_type]['total'][e] for e in epochs]

                    marker = markers.get(model_name, 'x')
                    
                    ax.plot(epochs, values, marker=marker, linewidth=2.5,
                           markersize=8, label=model_name, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Norm')
            ax.set_title('Total Parameter Norms Comparison')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison_total_norms.png'), dpi=300)
        print(f"✓ Saved total norms comparison")
        plt.close()


def save_comparison_results(results: Dict, output_dir: str):
    """Save numerical comparison results to text file."""
    output_file = os.path.join(output_dir, 'model_comparison_results.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MULTI-MODEL PARAMETER NORM COMPARISON RESULTS\n")
        f.write("="*70 + "\n\n")
        
        model_names = list(results.keys())
        
        for norm_type in ['frobenius', 'l2', 'nuclear', 'spectral']:
            f.write(f"\n{norm_type.upper()} NORM COMPARISON:\n")
            f.write("="*70 + "\n")
            
            # Get all epochs (union across models)
            all_epochs = set()
            for model_name in model_names:
                if norm_type in results[model_name]:
                    for component_data in results[model_name][norm_type].values():
                        all_epochs.update(component_data.keys())
            all_epochs = sorted(all_epochs)
            
            # For each component
            for component in ['input_proj', 'main_blocks', 'output_proj', 'total']:
                f.write(f"\n{component.upper()}:\n")
                f.write("-"*70 + "\n")
                
                # Header
                header = f"{'Epoch':<10}"
                for model_name in model_names:
                    header += f"{model_name:<15}"
                f.write(header + "\n")
                f.write("-"*70 + "\n")
                
                # Data rows
                for epoch in all_epochs:
                    row = f"{epoch:<10}"
                    for model_name in model_names:
                        if (norm_type in results[model_name] and 
                            component in results[model_name][norm_type] and
                            epoch in results[model_name][norm_type][component]):
                            value = results[model_name][norm_type][component][epoch]
                            row += f"{value:<15.6e}"
                        else:
                            row += f"{'N/A':<15}"
                    f.write(row + "\n")
    
    print(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    log_dirs = {
        'FFNet': "/home/exx/Projects/OCINR/logs/2025-10-09_12-45-01",
        'OCFFNet_without_OT': "/home/exx/Projects/OCINR/logs/2025-10-10_08-02-03",
        'OCFFNet_with_OT': "/home/exx/Projects/OCINR/logs/2025-10-10_09-50-33",
    }
    
    model_classes = {
        'FFNet': INRTraining,
        'OCFFNet_without_OT': OCINRTraining,
        'OCFFNet_with_OT': OCINRTraining,
    }
    
    output_dir = "/home/exx/Projects/OCINR/results"
    
    # Norm types to calculate
    norm_types = ['l2']
    
    print("="*70)
    print(f"Output directory: {output_dir}\n")
    
    # Check if log directories exist
    valid_models = {}
    valid_classes = {}
    for model_name, log_dir in log_dirs.items():
        if os.path.exists(log_dir):
            valid_models[model_name] = log_dir
            valid_classes[model_name] = model_classes[model_name]
            print(f"✓ Found {model_name} logs: {log_dir}")
        else:
            print(f"✗ {model_name} log directory not found: {log_dir}")
    
    if not valid_models:
        print("\n❌ No valid log directories found!")
        print("Please update the log_dirs dictionary in the script.")
        sys.exit(1)
    
    print(f"\nWill compare {len(valid_models)} models: {list(valid_models.keys())}")
    
    # Run comparison
    results = compare_models_across_epochs(
        log_dirs=valid_models,
        model_classes=valid_classes,
        epochs=None,  # Analyze all available epochs
        norm_types=norm_types,
        output_dir=output_dir
    )
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING COMPARISON VISUALIZATIONS")
    print("="*70)
    
    plot_model_comparison(results, norm_types, output_dir)
    
    # Save results to file
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    save_comparison_results(results, output_dir)
    
    print("\n✓ Comparison complete!")
    print(f"\nGenerated files in {output_dir}:")
    for norm_type in norm_types:
        print(f"  - comparison_{norm_type}_input_proj.png")
        print(f"  - comparison_{norm_type}_main_blocks.png")
        print(f"  - comparison_{norm_type}_output_proj.png")
    print(f"  - comparison_total_norms.png")
    print(f"  - model_comparison_results.txt")

