"""
Analyze and track parameter norms across checkpoint epochs.

This script calculates various norms of model parameters for different components:
- input_proj: Input projection layer
- hidden_blocks: Hidden layers (residual or MLP blocks)
- output_proj: Output projection layer

Tracks how these norms evolve during training.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import scienceplots

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from load_checkpoints import CheckpointLoader
from src.models.modelmodule import INRTraining, OCINRTraining


def calculate_parameter_norms(model: torch.nn.Module, norm_type: str = 'frobenius') -> Dict[str, float]:
    """
    Calculate norms of parameters for different model components.
    
    Args:
        model: The neural network model (should be FFNet or similar)
        norm_type: Type of norm to calculate ('frobenius', 'l1', 'l2', 'nuclear')
    
    Returns:
        Dictionary with norms for each component
    """
    norms = {}
    
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
            # Nuclear norm (sum of singular values) - only for 2D matrices
            if param_tensor.ndim == 2:
                return torch.norm(param_tensor, p='nuc').item()
            else:
                # For non-2D tensors, flatten and use Frobenius
                return torch.norm(param_tensor.flatten(), p=2).item()
        elif norm_type == 'spectral':
            # Spectral norm (largest singular value) - only for 2D matrices
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
    
    # 2. Hidden blocks norms
    if hasattr(model, 'hidden_blocks'):
        hidden_blocks_norm = 0.0
        param_count = 0
        for block in model.hidden_blocks:
            for name, param in block.named_parameters():
                if param.requires_grad:
                    hidden_blocks_norm += compute_norm(param, norm_type)
                    param_count += 1
        norms['hidden_blocks'] = hidden_blocks_norm
        norms['hidden_blocks_avg'] = hidden_blocks_norm / param_count if param_count > 0 else 0.0
    
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


def calculate_weight_statistics(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Calculate statistics (mean, std, min, max) for each component.
    
    Args:
        model: The neural network model
        
    Returns:
        Dictionary with statistics for each component
    """
    stats = {}
    
    def compute_stats(module, name):
        """Compute statistics for a module."""
        all_weights = []
        for param in module.parameters():
            if param.requires_grad:
                all_weights.append(param.data.flatten())
        
        if all_weights:
            all_weights = torch.cat(all_weights)
            stats[name] = {
                'mean': all_weights.mean().item(),
                'std': all_weights.std().item(),
                'min': all_weights.min().item(),
                'max': all_weights.max().item(),
                'abs_mean': all_weights.abs().mean().item(),
            }
    
    if hasattr(model, 'input_proj'):
        compute_stats(model.input_proj, 'input_proj')
    
    if hasattr(model, 'hidden_blocks'):
        compute_stats(model.hidden_blocks, 'hidden_blocks')
    
    if hasattr(model, 'output_proj'):
        compute_stats(model.output_proj, 'output_proj')
    
    return stats


def analyze_norms_across_epochs(
    log_dir: str,
    epochs: List[int] = None,
    norm_types: List[str] = ['frobenius', 'l2', 'nuclear'],
    model_class = INRTraining,
    output_dir: str = None
) -> Dict[str, Dict[int, float]]:
    """
    Analyze parameter norms across multiple checkpoint epochs.
    
    Args:
        log_dir: Path to log directory containing checkpoints
        epochs: List of epochs to analyze (None = all available)
        norm_types: List of norm types to calculate
        model_class: Model class to use for loading
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping norm types to epoch->norm mappings
    """
    if output_dir is None:
        output_dir = log_dir
    
    # Initialize checkpoint loader
    loader = CheckpointLoader(log_dir)
    available_epochs = loader.list_checkpoints()
    
    if epochs is None:
        epochs = available_epochs
    else:
        epochs = [e for e in epochs if e in available_epochs]
    
    print(f"Analyzing {len(epochs)} checkpoints at epochs: {epochs}")
    print(f"Norm types: {norm_types}\n")
    
    # Storage for results
    results = {norm_type: {
        'input_proj': {},
        'hidden_blocks': {},
        'output_proj': {},
        'total': {}
    } for norm_type in norm_types}
    
    statistics = {}
    
    # Analyze each checkpoint
    for epoch in epochs:
        print(f"Processing epoch {epoch}...", end=' ')
        
        # Load model
        model, _ = loader.load_checkpoint(epoch, model_class=model_class)
        
        # Calculate norms for each norm type
        for norm_type in norm_types:
            norms = calculate_parameter_norms(model, norm_type=norm_type)
            
            for component in ['input_proj', 'hidden_blocks', 'output_proj', 'total']:
                if component in norms:
                    results[norm_type][component][epoch] = norms[component]
        
        # Calculate statistics
        stats = calculate_weight_statistics(model)
        statistics[epoch] = stats
        
        print("✓")
    
    # Print summary
    print("\n" + "="*70)
    print("PARAMETER NORM ANALYSIS SUMMARY")
    print("="*70)
    
    for norm_type in norm_types:
        print(f"\n{norm_type.upper()} NORM:")
        print("-" * 70)
        
        # Print table header
        print(f"{'Epoch':<10} {'Input Proj':<15} {'Hidden Blocks':<15} {'Output Proj':<15} {'Total':<15}")
        print("-" * 70)
        
        for epoch in epochs:
            row = f"{epoch:<10}"
            for component in ['input_proj', 'hidden_blocks', 'output_proj', 'total']:
                value = results[norm_type][component].get(epoch, 0.0)
                row += f"{value:<15.4e}"
            print(row)
    
    # Print weight statistics
    print("\n" + "="*70)
    print("WEIGHT STATISTICS (Last Epoch)")
    print("="*70)
    
    last_epoch = max(epochs)
    if last_epoch in statistics:
        for component, stat in statistics[last_epoch].items():
            print(f"\n{component}:")
            for key, value in stat.items():
                print(f"  {key}: {value:.6e}")
    
    return results, statistics


def plot_norm_evolution(
    results: Dict[str, Dict[int, float]],
    epochs: List[int],
    output_dir: str,
    norm_types: List[str] = ['frobenius', 'l2', 'nuclear']
):
    """
    Plot the evolution of parameter norms across epochs.
    
    Args:
        results: Results from analyze_norms_across_epochs
        epochs: List of epochs
        output_dir: Directory to save plots
        norm_types: List of norm types to plot
    """
    epochs_sorted = sorted(epochs)
    
    with plt.style.context('science'):
        # Plot 1: All components for each norm type
        for norm_type in norm_types:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            components = ['input_proj', 'hidden_blocks', 'output_proj']
            colors = ['blue', 'orange', 'green']
            markers = ['o', 's', '^']
            
            for component, color, marker in zip(components, colors, markers):
                if component in results[norm_type]:
                    values = [results[norm_type][component].get(e, 0.0) for e in epochs_sorted]
                    ax.plot(epochs_sorted, values, marker=marker, linewidth=2.5, 
                           markersize=8, label=component.replace('_', ' ').title(),
                           color=color)
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(f'{norm_type.capitalize()} Norm', fontsize=12)
            ax.set_title(f'Parameter Norms Evolution ({norm_type.capitalize()})', fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'norms_{norm_type}.png'), dpi=300)
            print(f"✓ Saved {norm_type} norm plot")
            plt.close()
        
        # Plot 2: Compare all norm types for each component
        components = ['input_proj', 'hidden_blocks', 'output_proj']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, component in enumerate(components):
            ax = axes[idx]
            
            for norm_type in norm_types:
                if component in results[norm_type]:
                    values = [results[norm_type][component].get(e, 0.0) for e in epochs_sorted]
                    ax.plot(epochs_sorted, values, 'o-', linewidth=2.5, markersize=8,
                           label=norm_type.capitalize())
            
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel('Norm Value', fontsize=12)
            ax.set_title(component.replace('_', ' ').title(), fontsize=12)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
        
        plt.suptitle('Comparison of Different Norm Types', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'norms_comparison.png'), dpi=300)
        print(f"✓ Saved norm comparison plot")
        plt.close()
        
        # Plot 3: Relative changes from initial epoch
        if len(epochs_sorted) > 1:
            initial_epoch = epochs_sorted[0]
            
            fig, axes = plt.subplots(1, len(norm_types), figsize=(6*len(norm_types), 5))
            if len(norm_types) == 1:
                axes = [axes]
            
            for idx, norm_type in enumerate(norm_types):
                ax = axes[idx]
                
                for component in components:
                    if component in results[norm_type]:
                        initial_value = results[norm_type][component].get(initial_epoch, 1.0)
                        relative_changes = [
                            (results[norm_type][component].get(e, 0.0) - initial_value) / initial_value * 100
                            for e in epochs_sorted
                        ]
                        ax.plot(epochs_sorted, relative_changes, 'o-', linewidth=2.5, 
                               markersize=8, label=component.replace('_', ' ').title())
                
                ax.set_xlabel('Epoch', fontsize=12)
                ax.set_ylabel('Relative Change (%)', fontsize=12)
                ax.set_title(f'{norm_type.capitalize()} Norm', fontsize=12)
                ax.legend(fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
            
            plt.suptitle('Relative Change in Norms from Initial Epoch', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'norms_relative_change.png'), dpi=300)
            print(f"✓ Saved relative change plot")
            plt.close()


def save_results_to_file(results: Dict, statistics: Dict, output_dir: str):
    """Save numerical results to text file."""
    output_file = os.path.join(output_dir, 'parameter_norms_results.txt')
    
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("PARAMETER NORM ANALYSIS RESULTS\n")
        f.write("="*70 + "\n\n")
        
        for norm_type, components in results.items():
            f.write(f"\n{norm_type.upper()} NORM:\n")
            f.write("-" * 70 + "\n")
            
            epochs = sorted(list(next(iter(components.values())).keys()))
            
            # Write header
            f.write(f"{'Epoch':<10} {'Input Proj':<15} {'Hidden Blocks':<15} {'Output Proj':<15} {'Total':<15}\n")
            f.write("-" * 70 + "\n")
            
            for epoch in epochs:
                row = f"{epoch:<10}"
                for component in ['input_proj', 'hidden_blocks', 'output_proj', 'total']:
                    value = components[component].get(epoch, 0.0)
                    row += f"{value:<15.6e}"
                f.write(row + "\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("WEIGHT STATISTICS\n")
        f.write("="*70 + "\n")
        
        for epoch in sorted(statistics.keys()):
            f.write(f"\nEpoch {epoch}:\n")
            for component, stats in statistics[epoch].items():
                f.write(f"  {component}:\n")
                for key, value in stats.items():
                    f.write(f"    {key}: {value:.6e}\n")
    
    print(f"✓ Results saved to {output_file}")


if __name__ == "__main__":
    # Configuration
    log_dir = "/home/exx/Projects/OCINR/logs/2025-10-09_12-45-01"
    output_dir = "/home/exx/Projects/OCINR/results"
    
    # Norm types to calculate
    norm_types = ['frobenius', 'l2', 'nuclear', 'spectral']
    
    print("="*70)
    print("PARAMETER NORM ANALYSIS")
    print("="*70)
    print(f"Log directory: {log_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Run analysis
    results, statistics = analyze_norms_across_epochs(
        log_dir=log_dir,
        epochs=None,  # Analyze all available epochs
        norm_types=norm_types,
        model_class=INRTraining,
        output_dir=output_dir
    )
    
    # Get epochs
    epochs = sorted(list(next(iter(next(iter(results.values())).values())).keys()))
    
    # Create visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    plot_norm_evolution(results, epochs, output_dir, norm_types)
    
    # Save results to file
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    
    save_results_to_file(results, statistics, output_dir)
    
    print("\n✓ Analysis complete!")
    print(f"\nGenerated files in {output_dir}:")
    print("  - norms_frobenius.png")
    print("  - norms_l2.png")
    print("  - norms_nuclear.png")
    print("  - norms_spectral.png")
    print("  - norms_comparison.png")
    print("  - norms_relative_change.png")
    print("  - parameter_norms_results.txt")

