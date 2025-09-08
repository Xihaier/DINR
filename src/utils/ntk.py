"""
Neural Tangent Kernel (NTK) Analysis Pipeline

References:
    - Neural Tangent Kernel: Convergence and Generalization in Neural Networks (Jacot et al., 2018)
    - On the Inductive Bias of Neural Tangent Kernels (Bietti & Mairal, 2019)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union
import warnings
import math

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class NTKResult:
    """Container for NTK analysis results.
    
    Attributes:
        eigenvalues: Eigenvalues sorted in descending order
        eigenvectors: Corresponding eigenvectors (optional)
        condition_number: λ_max / λ_min (for positive eigenvalues)
        trace: Sum of all eigenvalues
        effective_rank: Participation ratio (trace²/||λ||²)
        explained_variance: Cumulative explained variance ratio
        spectrum_decay: Rate of eigenvalue decay
        gram_matrix: Full NTK matrix (optional, for small datasets)
        metadata: Additional analysis metadata
    """
    eigenvalues: np.ndarray
    condition_number: float
    trace: float
    effective_rank: float
    explained_variance: np.ndarray
    spectrum_decay: float
    eigenvectors: Optional[np.ndarray] = None
    gram_matrix: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None


class NTKAnalyzer:
    """Neural Tangent Kernel analyzer for OCINR models."""
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize NTK analyzer.
        
        Args:
            model: Neural network model to analyze
            device: Device to run computations on (auto-detected if None)
            dtype: Computation dtype for numerical stability
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.dtype = dtype
        
        # Move model to specified device and dtype
        self.model = self.model.to(self.device, dtype=self.dtype)
        
        # Cache trainable parameters
        self._params = [p for p in self.model.parameters() if p.requires_grad]
        if len(self._params) == 0:
            raise ValueError("Model has no trainable parameters")
        
        self._param_shapes = [p.shape for p in self._params]
        self._param_sizes = [p.numel() for p in self._params]
        self._total_params = sum(self._param_sizes)
        
    def _extract_prediction(self, output: Union[Tensor, Tuple, List]) -> Tensor:
        """Extract prediction tensor from model output.
        
        Handles models that return:
        - Single tensor: y
        - Tuple/List: (y, aux_terms...)
        
        Args:
            output: Model output
            
        Returns:
            Prediction tensor of shape (N, C)
        """
        if isinstance(output, (tuple, list)):
            y = output[0]
        else:
            y = output
            
        # Ensure 2D shape (N, C)
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        elif y.ndim > 2:
            raise ValueError(f"Expected 1D or 2D output, got shape {y.shape}")
            
        return y
    
    def _flatten_gradients(
        self, 
        gradients: List[Optional[Tensor]]
    ) -> Tensor:
        """Flatten gradients into a single vector.
        
        Args:
            gradients: List of per-parameter gradients (may contain None)
            
        Returns:
            Flattened gradient vector
        """
        flat_grads = []
        for grad, size in zip(gradients, self._param_sizes):
            if grad is None:
                # Handle unused parameters with zero gradients
                flat_grads.append(torch.zeros(size, device=self.device, dtype=self.dtype))
            else:
                flat_grads.append(grad.reshape(-1))
                
        return torch.cat(flat_grads, dim=0)
    
    def compute_ntk_matrix(
        self,
        inputs: Tensor,
        *,
        normalize: Literal["none", "trace", "params", "params_outputs"] = "none",
        center_kernel: bool = False,
        show_progress: bool = False,
        chunk_size: Optional[int] = None
    ) -> Tensor:
        """Compute empirical NTK matrix.
        
        The empirical NTK is defined as:
        K_ij = Σ_c ⟨∂f_c(x_i)/∂θ, ∂f_c(x_j)/∂θ⟩
        
        Args:
            inputs: Input coordinates of shape (N, input_dim)
            normalize: Normalization scheme for the kernel
            center_kernel: Whether to apply kernel centering
            show_progress: Whether to show computation progress
            chunk_size: Process inputs in chunks to save memory
            
        Returns:
            NTK matrix of shape (N, N) on CPU
        """
        inputs = inputs.to(self.device, dtype=self.dtype)
        N = inputs.shape[0]
        
        # Store original training state
        was_training = self.model.training
        self.model.eval()
        
        try:
            if chunk_size is not None and N > chunk_size:
                # Chunked computation for large datasets
                return self._compute_ntk_chunked(
                    inputs, normalize, center_kernel, show_progress, chunk_size
                )
            else:
                # Standard computation
                return self._compute_ntk_standard(
                    inputs, normalize, center_kernel, show_progress
                )
                
        finally:
            # Restore training state
            self.model.train(was_training)
    
    def _compute_ntk_standard(
        self,
        inputs: Tensor,
        normalize: str,
        center_kernel: bool,
        show_progress: bool
    ) -> Tensor:
        """Standard NTK computation for moderate-sized datasets."""
        N = inputs.shape[0]
        
        # Forward pass with gradient tracking
        with torch.enable_grad():
            inputs.requires_grad_(True)
            output = self.model(inputs)
            y = self._extract_prediction(output)
            N, C = y.shape
            
            # Compute Jacobian matrix J ∈ R^{(N×C) × P}
            jacobian_rows = []
            total_outputs = N * C
            
            for i in range(N):
                for c in range(C):
                    if show_progress and (i * C + c) % max(1, total_outputs // 20) == 0:
                        print(f"  Computing gradients: {i * C + c + 1}/{total_outputs}")
                    
                    # Compute gradients for output y[i, c]
                    retain_graph = not (i == N - 1 and c == C - 1)
                    grads = torch.autograd.grad(
                        outputs=y[i, c],
                        inputs=self._params,
                        retain_graph=retain_graph,
                        create_graph=False,
                        allow_unused=True
                    )
                    
                    # Flatten and store gradient
                    grad_flat = self._flatten_gradients(grads)
                    jacobian_rows.append(grad_flat.detach().cpu())
            
            # Stack into Jacobian matrix
            J = torch.stack(jacobian_rows, dim=0)  # Shape: (N×C, P)
        
        # Compute NTK: K = J @ J^T
        K = J @ J.t()  # Shape: (N×C, N×C)
        
        # Sum over output channels to get (N, N) kernel
        if C > 1:
            K = K.view(N, C, N, C).sum(dim=(1, 3))  # Sum over output dimensions
        else:
            K = K.squeeze()
            
        # Apply normalization
        K = self._apply_normalization(K, normalize, C)
        
        # Apply kernel centering if requested
        if center_kernel:
            K = self._center_kernel(K)
            
        # Ensure numerical symmetry
        K = 0.5 * (K + K.t())
        
        return K.detach()
    
    def _compute_ntk_chunked(
        self,
        inputs: Tensor,
        normalize: str,
        center_kernel: bool,
        show_progress: bool,
        chunk_size: int
    ) -> Tensor:
        """Chunked NTK computation for large datasets."""
        N = inputs.shape[0]
        K = torch.zeros(N, N, dtype=torch.float64)
        
        # Process in chunks
        num_chunks = math.ceil(N / chunk_size)
        
        for i in range(num_chunks):
            start_i = i * chunk_size
            end_i = min((i + 1) * chunk_size, N)
            chunk_i = inputs[start_i:end_i]
            
            for j in range(i, num_chunks):  # Only compute upper triangle
                start_j = j * chunk_size
                end_j = min((j + 1) * chunk_size, N)
                chunk_j = inputs[start_j:end_j]
                
                if show_progress:
                    print(f"  Processing chunk ({i+1}/{num_chunks}, {j+1}/{num_chunks})")
                
                # Compute kernel block
                K_block = self._compute_kernel_block(chunk_i, chunk_j)
                K[start_i:end_i, start_j:end_j] = K_block
                
                # Fill symmetric part
                if i != j:
                    K[start_j:end_j, start_i:end_i] = K_block.t()
        
        # Apply post-processing
        K = self._apply_normalization(K, normalize, 1)  # Assume C=1 for chunked
        if center_kernel:
            K = self._center_kernel(K)
            
        return K
    
    def _compute_kernel_block(self, inputs_i: Tensor, inputs_j: Tensor) -> Tensor:
        """Compute NTK block between two input chunks."""
        # This is a simplified version - in practice, you'd compute the full
        # cross-Jacobian between the two chunks
        raise NotImplementedError("Chunked computation not fully implemented")
    
    def _apply_normalization(self, K: Tensor, normalize: str, num_outputs: int) -> Tensor:
        """Apply normalization to the kernel matrix."""
        if normalize == "none":
            return K
        elif normalize == "trace":
            trace = torch.trace(K)
            return K / (trace + 1e-12)
        elif normalize == "params":
            return K / self._total_params
        elif normalize == "params_outputs":
            return K / (self._total_params * num_outputs)
        else:
            raise ValueError(f"Unknown normalization: {normalize}")
    
    def _center_kernel(self, K: Tensor) -> Tensor:
        """Apply kernel centering: K_centered = H @ K @ H."""
        N = K.shape[0]
        H = torch.eye(N, dtype=K.dtype) - torch.ones(N, N, dtype=K.dtype) / N
        return H @ K @ H
    
    def analyze_spectrum(
        self,
        inputs: Tensor,
        *,
        top_k: Optional[int] = None,
        normalize: Literal["none", "trace", "params", "params_outputs"] = "none",
        center_kernel: bool = False,
        return_eigenvectors: bool = False,
        return_matrix: bool = False,
        show_progress: bool = False
    ) -> NTKResult:
        """Perform comprehensive NTK spectrum analysis.
        
        Args:
            inputs: Input coordinates
            top_k: Number of top eigenvalues to keep (None = all)
            normalize: Kernel normalization scheme
            center_kernel: Whether to center the kernel
            return_eigenvectors: Whether to compute and return eigenvectors
            return_matrix: Whether to return the full kernel matrix
            show_progress: Whether to show computation progress
            
        Returns:
            NTKResult with comprehensive spectrum analysis
        """
        if show_progress:
            print("Computing NTK matrix...")
            
        # Compute kernel matrix
        K = self.compute_ntk_matrix(
            inputs,
            normalize=normalize,
            center_kernel=center_kernel,
            show_progress=show_progress
        )
        
        if show_progress:
            print("Computing eigendecomposition...")
        
        # Eigendecomposition (use double precision for stability)
        K_double = K.to(torch.float64)
        if return_eigenvectors:
            eigenvals, eigenvecs = torch.linalg.eigh(K_double)
            # Sort in descending order
            idx = torch.argsort(eigenvals, descending=True)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            eigenvecs = eigenvecs.numpy() if return_eigenvectors else None
        else:
            eigenvals = torch.linalg.eigvalsh(K_double)
            eigenvals = torch.sort(eigenvals, descending=True)[0]
            eigenvecs = None
        
        eigenvals = eigenvals.numpy()
        
        # Truncate to top_k if specified
        if top_k is not None:
            eigenvals = eigenvals[:top_k]
            if eigenvecs is not None:
                eigenvecs = eigenvecs[:, :top_k]
        
        # Compute spectrum statistics
        stats = self._compute_spectrum_statistics(eigenvals)
        
        # Create result
        result = NTKResult(
            eigenvalues=eigenvals,
            eigenvectors=eigenvecs,
            condition_number=stats["condition_number"],
            trace=stats["trace"],
            effective_rank=stats["effective_rank"],
            explained_variance=stats["explained_variance"],
            spectrum_decay=stats["spectrum_decay"],
            gram_matrix=K.numpy() if return_matrix else None,
            metadata={
                "num_samples": inputs.shape[0],
                "num_params": self._total_params,
                "normalization": normalize,
                "centered": center_kernel,
                "input_dim": inputs.shape[1],
                "model_type": type(self.model).__name__
            }
        )
        
        return result
    
    def _compute_spectrum_statistics(self, eigenvals: np.ndarray) -> Dict:
        """Compute comprehensive spectrum statistics."""
        eps = 1e-12
        positive_eigs = eigenvals[eigenvals > eps]
        
        # Basic statistics
        trace = float(eigenvals.sum())
        max_eig = float(eigenvals[0]) if len(eigenvals) > 0 else 0.0
        min_pos_eig = float(positive_eigs[-1]) if len(positive_eigs) > 0 else eps
        
        # Condition number
        condition_number = max_eig / min_pos_eig if len(positive_eigs) > 0 else float('inf')
        
        # Effective rank (participation ratio)
        sum_eigs = eigenvals.sum()
        sum_eigs_sq = (eigenvals ** 2).sum()
        effective_rank = (sum_eigs ** 2) / (sum_eigs_sq + eps)
        
        # Explained variance ratio
        explained_variance = np.cumsum(eigenvals) / (sum_eigs + eps)
        
        # Spectrum decay rate (fit exponential decay)
        if len(eigenvals) > 2:
            # Fit log(λ_i) ≈ a - b*i
            indices = np.arange(1, len(eigenvals) + 1)
            log_eigs = np.log(np.maximum(eigenvals, eps))
            spectrum_decay = -np.polyfit(indices, log_eigs, 1)[0]
        else:
            spectrum_decay = 0.0
        
        return {
            "condition_number": condition_number,
            "trace": trace,
            "effective_rank": effective_rank,
            "explained_variance": explained_variance,
            "spectrum_decay": spectrum_decay
        }
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        inputs: Tensor,
        **kwargs
    ) -> Dict[str, NTKResult]:
        """Compare NTK spectra across multiple models.
        
        Args:
            models: Dictionary of model_name -> model
            inputs: Common input coordinates
            **kwargs: Arguments passed to analyze_spectrum
            
        Returns:
            Dictionary of model_name -> NTKResult
        """
        results = {}
        
        for name, model in models.items():
            print(f"\nAnalyzing model: {name}")
            analyzer = NTKAnalyzer(model, device=self.device, dtype=self.dtype)
            results[name] = analyzer.analyze_spectrum(inputs, **kwargs)
            
        return results
    
    def print_spectrum_summary(self, result: NTKResult, top_k: int = 10):
        """Print a summary of spectrum analysis results."""
        print(f"\n=== NTK Spectrum Analysis ===")
        print(f"Model: {result.metadata['model_type']}")
        print(f"Samples: {result.metadata['num_samples']}")
        print(f"Parameters: {result.metadata['num_params']:,}")
        print(f"Input dimension: {result.metadata['input_dim']}")
        print(f"\n--- Spectrum Statistics ---")
        print(f"Trace: {result.trace:.4f}")
        print(f"Condition number: {result.condition_number:.2e}")
        print(f"Effective rank: {result.effective_rank:.2f}")
        print(f"Spectrum decay rate: {result.spectrum_decay:.4f}")
        
        # Top eigenvalues
        k = min(top_k, len(result.eigenvalues))
        print(f"\n--- Top {k} Eigenvalues ---")
        for i in range(k):
            evr = result.explained_variance[i] if i < len(result.explained_variance) else 0
            print(f"λ_{i+1:2d}: {result.eigenvalues[i]:8.4f} (EVR: {evr:.3f})")


def analyze_model_ntk(
    model: nn.Module,
    inputs: Tensor,
    *,
    device: Optional[torch.device] = None,
    **kwargs
) -> NTKResult:
    """Convenience function for single-model NTK analysis.
    
    Args:
        model: Neural network model
        inputs: Input coordinates
        device: Computation device
        **kwargs: Additional arguments for analyze_spectrum
        
    Returns:
        NTK analysis result
    """
    analyzer = NTKAnalyzer(model, device=device)
    return analyzer.analyze_spectrum(inputs, **kwargs)


def compare_model_ntks(
    models: Dict[str, nn.Module],
    inputs: Tensor,
    *,
    device: Optional[torch.device] = None,
    **kwargs
) -> Dict[str, NTKResult]:
    """Convenience function for multi-model NTK comparison.
    
    Args:
        models: Dictionary of model_name -> model
        inputs: Input coordinates  
        device: Computation device
        **kwargs: Additional arguments for analyze_spectrum
        
    Returns:
        Dictionary of model_name -> NTKResult
    """
    if not models:
        raise ValueError("No models provided")
    
    # Use first model's device if not specified
    if device is None:
        device = next(iter(models.values())).parameters().__next__().device
    
    analyzer = NTKAnalyzer(list(models.values())[0], device=device)
    return analyzer.compare_models(models, inputs, **kwargs)


# Utility functions for common use cases
def quick_ntk_analysis(
    model: nn.Module,
    input_dim: int,
    num_samples: int = 100,
    *,
    input_range: Tuple[float, float] = (-1.0, 1.0),
    seed: Optional[int] = 42,
    **kwargs
) -> NTKResult:
    """Quick NTK analysis with random inputs.
    
    Args:
        model: Model to analyze
        input_dim: Input dimension
        num_samples: Number of random samples
        input_range: Range for random input generation
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for analyze_spectrum
        
    Returns:
        NTK analysis result
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate random inputs
    inputs = torch.rand(num_samples, input_dim) * (input_range[1] - input_range[0]) + input_range[0]
    
    return analyze_model_ntk(model, inputs, **kwargs)


if __name__ == "__main__":
    # Basic test to ensure imports work
    print("NTK analysis module loaded successfully!")
    print("Available functions:")
    print("- NTKAnalyzer: Main analysis class")
    print("- analyze_model_ntk: Single model analysis")
    print("- compare_model_ntks: Multi-model comparison")
    print("- quick_ntk_analysis: Quick analysis with random inputs")
