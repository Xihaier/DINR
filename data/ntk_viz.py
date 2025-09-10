"""
NTK Eigenvalue Visualization for Neural Network Analysis

This module provides utilities for visualizing Neural Tangent Kernel (NTK) eigenvalue
evolution during neural network training. It creates heatmaps showing how eigenvalues
change across training iterations for different network architectures.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
import scienceplots

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
DEFAULT_TOP_K = 10
DEFAULT_MAX_ITERATIONS = 15
DEFAULT_FIGURE_SIZE = (5, 3.42)
DEFAULT_DPI = 200

class NTKDataLoader:
    """Handles loading and validation of NTK analysis data."""
    
    def __init__(self, data_dir: Union[str, Path] = "."):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing the NTK data files
        """
        self.data_dir = Path(data_dir)
    
    def load_data(self) -> Dict[str, np.ndarray]:
        """
        Load NTK analysis data for all network architectures.
        
        Returns:
            Dictionary mapping network names to their NTK data arrays
            
        Raises:
            FileNotFoundError: If required data files are not found
            ValueError: If data files are corrupted or invalid
        """
        data_files = {
            "FFNet": "ntk_analysis_FFNet.npy",
            "OCFFNet": "ntk_analysis_OCFFNet.npy", 
            "SIREN": "ntk_analysis_SIREN.npy",
            "OCSIREN": "ntk_analysis_OCSIREN.npy"
        }
        
        loaded_data = {}
        for name, filename in data_files.items():
            filepath = self.data_dir / filename
            try:
                data = np.load(filepath, allow_pickle=True)
                if len(data) == 0:
                    raise ValueError(f"Empty data file: {filepath}")
                loaded_data[name] = data
                logger.info(f"Loaded {name} data: {len(data)} iterations")
            except FileNotFoundError:
                logger.error(f"Data file not found: {filepath}")
                raise
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
                raise ValueError(f"Failed to load {name} data") from e
        
        return loaded_data

def extract_eigenvalue_matrix(
    data_array: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    max_iterations: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract eigenvalue matrix from NTK analysis data.
    
    This function processes the raw NTK data to create a structured matrix
    where rows represent training iterations and columns represent eigenvalue ranks.
    
    Args:
        data_array: Array of dictionaries containing eigenvalue data for each iteration
        top_k: Number of top eigenvalues to extract (default: 10)
        max_iterations: Maximum number of iterations to process. If None, uses all
                       available iterations up to DEFAULT_MAX_ITERATIONS
    
    Returns:
        Tuple containing:
        - eigenvalue_matrix: Matrix of shape (num_iterations, top_k) with eigenvalues
        - iteration_indices: Array of iteration indices corresponding to matrix rows
        
    Note:
        Missing eigenvalues are filled with NaN values.
    """
    if max_iterations is None:
        max_iterations = min(DEFAULT_MAX_ITERATIONS, len(data_array))
    else:
        max_iterations = min(max_iterations, len(data_array))
    
    iteration_indices = np.arange(max_iterations)
    eigenvalue_matrix = np.full((max_iterations, top_k), np.nan, dtype=np.float64)
    
    for iter_idx in range(max_iterations):
        iteration_data = data_array[iter_idx]
        
        # Extract eigenvalues (1-indexed in the data)
        for eigen_rank in range(1, top_k + 1):
            eigenvalue_key = f"eigenvalue_{eigen_rank}"
            if eigenvalue_key in iteration_data:
                eigenvalue_matrix[iter_idx, eigen_rank - 1] = iteration_data[eigenvalue_key]
    
    return eigenvalue_matrix, iteration_indices

def create_shared_normalization(
    eigenvalue_matrices: List[np.ndarray],
    top_k: int,
    use_log_scale: bool = False,
    match_scale: bool = True
) -> Optional[Union[Normalize, LogNorm]]:
    """
    Create a shared color normalization across multiple eigenvalue matrices.
    
    This function creates a consistent color scale for visualizing eigenvalue heatmaps
    by computing normalization parameters across all provided matrices. The first
    eigenvalue (typically the largest) is excluded from normalization to focus on
    the more interesting smaller eigenvalues.
    
    Args:
        eigenvalue_matrices: List of eigenvalue matrices to normalize across
        top_k: Number of eigenvalues per matrix
        use_log_scale: Whether to use logarithmic scaling for color normalization
        match_scale: If False, returns None (allowing each plot to autoscale)
    
    Returns:
        Matplotlib normalization object (Normalize or LogNorm) for consistent
        color scaling, or None if match_scale is False or no valid data found
        
    Raises:
        ValueError: If log scale is requested but no positive eigenvalues exist
    """
    if not match_scale:
        return None

    # Collect all eigenvalues excluding the first (largest) eigenvalue
    all_eigenvalues = []
    for matrix in eigenvalue_matrices:
        # Skip first eigenvalue column (index 0), use eigenvalues 2..top_k
        eigenvalue_subset = matrix[:, 1:top_k]
        all_eigenvalues.append(eigenvalue_subset.ravel())
    
    # Combine and filter finite values
    combined_eigenvalues = np.concatenate(all_eigenvalues)
    finite_eigenvalues = combined_eigenvalues[np.isfinite(combined_eigenvalues)]
    
    if finite_eigenvalues.size == 0:
        logger.warning("No finite eigenvalues found for normalization")
        return None

    if use_log_scale:
        positive_eigenvalues = finite_eigenvalues[finite_eigenvalues > 0]
        if positive_eigenvalues.size == 0:
            raise ValueError(
                "Logarithmic scale requested but no positive eigenvalues found. "
                "Consider using linear scale instead."
            )
        
        # Use machine epsilon as minimum to avoid log(0)
        vmin = max(positive_eigenvalues.min(), np.finfo(np.float64).eps)
        vmax = finite_eigenvalues.max()
        return LogNorm(vmin=vmin, vmax=vmax)
    else:
        return Normalize(vmin=finite_eigenvalues.min(), vmax=finite_eigenvalues.max())

class NTKHeatmapPlotter:
    """Handles creation of NTK eigenvalue heatmap visualizations."""
    
    def __init__(
        self,
        figure_size: Tuple[float, float] = DEFAULT_FIGURE_SIZE,
        dpi: int = DEFAULT_DPI,
        colormap: str = "PiYG_r"
    ):
        """
        Initialize the heatmap plotter.
        
        Args:
            figure_size: Figure dimensions as (width, height) in inches
            dpi: Resolution for saved figures
            colormap: Matplotlib colormap name for the heatmap
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.colormap = colormap
    
    def plot_eigenvalue_heatmap(
        self,
        eigenvalue_matrix: np.ndarray,
        title: str,
        top_k: int = DEFAULT_TOP_K,
        normalization: Optional[Union[Normalize, LogNorm]] = None,
        output_path: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Create and save an eigenvalue evolution heatmap.
        
        This function creates a heatmap visualization showing how NTK eigenvalues
        evolve during training. The first (largest) eigenvalue is excluded from
        the visualization to focus on the more interesting smaller eigenvalues.
        
        Args:
            eigenvalue_matrix: Matrix of shape (num_iterations, top_k) containing eigenvalues
            title: Title for the plot
            top_k: Number of eigenvalues in the matrix
            normalization: Optional color normalization for consistent scaling across plots
            output_path: Path to save the figure. If None, figure is not saved
            
        Note:
            - Rows represent eigenvalue ranks (2 through top_k, displayed as 1 through top_k-1)
            - Columns represent training iterations
            - The plot uses scientific plotting style for publication quality
        """
        with plt.style.context('science'):
            # Exclude first eigenvalue and transpose for proper orientation
            # Shape becomes (top_k-1, num_iterations) for heatmap display
            heatmap_data = eigenvalue_matrix[:, 1:top_k].T
            num_eigenvalues, num_iterations = heatmap_data.shape
            
            # Create figure and heatmap
            plt.figure(figsize=self.figure_size)
            heatmap_image = plt.imshow(
                heatmap_data,
                cmap=self.colormap,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
                norm=normalization
            )
            
            # Set labels and title
            plt.title(title, fontsize=12, pad=10)
            plt.xlabel("Training Step", fontsize=10)
            plt.ylabel("NTK Eigenvalue Rank", fontsize=10)
            
            # Configure y-axis ticks (eigenvalue ranks 1 through top_k-1)
            eigenvalue_labels = [str(i) for i in range(1, top_k)]
            plt.yticks(np.arange(num_eigenvalues), eigenvalue_labels)
            
            # Configure x-axis ticks (training iterations, max 10 ticks)
            max_ticks = 10
            tick_step = max(1, num_iterations // max_ticks)
            tick_positions = np.arange(0, num_iterations, tick_step)
            tick_labels = [str(i) for i in tick_positions]
            plt.xticks(tick_positions, tick_labels)
            
            # Add colorbar
            colorbar = plt.colorbar(heatmap_image)
            
            # Apply tight layout and save
            plt.tight_layout()
            
            if output_path is not None:
                output_path = Path(output_path)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
                logger.info(f"Saved heatmap to {output_path}")
            
            plt.close()

class NTKVisualizationPipeline:
    """Main pipeline for generating NTK eigenvalue visualizations."""
    
    def __init__(
        self,
        data_dir: Union[str, Path] = ".",
        output_dir: Union[str, Path] = ".",
        top_k: int = DEFAULT_TOP_K,
        use_log_scale: bool = False,
        match_scale: bool = False
    ):
        """
        Initialize the visualization pipeline.
        
        Args:
            data_dir: Directory containing NTK data files
            output_dir: Directory to save output figures
            top_k: Number of top eigenvalues to analyze
            use_log_scale: Whether to use logarithmic color scaling
            match_scale: Whether to use consistent color scaling across all plots
        """
        self.data_loader = NTKDataLoader(data_dir)
        self.plotter = NTKHeatmapPlotter()
        self.output_dir = Path(output_dir)
        self.top_k = top_k
        self.use_log_scale = use_log_scale
        self.match_scale = match_scale
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _align_matrices_to_common_length(
        self, 
        matrices: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Align all eigenvalue matrices to have the same number of iterations.
        
        Args:
            matrices: Dictionary of network names to eigenvalue matrices
            
        Returns:
            Dictionary of matrices aligned to the minimum common length
        """
        min_iterations = min(matrix.shape[0] for matrix in matrices.values())
        logger.info(f"Aligning matrices to {min_iterations} iterations")
        
        aligned_matrices = {}
        for name, matrix in matrices.items():
            aligned_matrices[name] = matrix[:min_iterations, :self.top_k]
            
        return aligned_matrices
    
    def generate_all_visualizations(self) -> None:
        """
        Generate heatmap visualizations for all network architectures.
        
        This method loads the data, processes eigenvalue matrices, creates
        consistent normalization if requested, and generates individual
        heatmaps for each network architecture.
        """
        logger.info("Starting NTK eigenvalue visualization pipeline")
        
        # Load and process data
        raw_data = self.data_loader.load_data()
        
        # Extract eigenvalue matrices
        eigenvalue_matrices = {}
        for network_name, data_array in raw_data.items():
            matrix, _ = extract_eigenvalue_matrix(data_array, top_k=self.top_k)
            eigenvalue_matrices[network_name] = matrix
            
        # Align matrices to common length
        aligned_matrices = self._align_matrices_to_common_length(eigenvalue_matrices)
        
        # Create shared normalization if requested
        normalization = None
        if self.match_scale:
            normalization = create_shared_normalization(
                list(aligned_matrices.values()),
                top_k=self.top_k,
                use_log_scale=self.use_log_scale,
                match_scale=self.match_scale
            )
        
        # Generate visualizations for each network
        network_configs = [
            ("FFNet", "(a) FFNet", "ffnet_eigs_heatmap.png"),
            ("OCFFNet", "(b) OCFFNet", "ocffnet_eigs_heatmap.png"),
            ("SIREN", "(c) SIREN", "siren_eigs_heatmap.png"),
            ("OCSIREN", "(d) OCSIREN", "ocsiren_eigs_heatmap.png")
        ]
        
        for network_name, plot_title, filename in network_configs:
            if network_name in aligned_matrices:
                output_path = self.output_dir / filename
                self.plotter.plot_eigenvalue_heatmap(
                    eigenvalue_matrix=aligned_matrices[network_name],
                    title=plot_title,
                    top_k=self.top_k,
                    normalization=normalization,
                    output_path=output_path
                )
            else:
                logger.warning(f"No data found for {network_name}")
        
        logger.info("Visualization pipeline completed successfully")


def main():
    """Main execution function for the NTK visualization script."""
    # Configuration parameters
    config = {
        "data_dir": ".",
        "output_dir": ".",
        "top_k": DEFAULT_TOP_K,
        "use_log_scale": False,  # Set True if eigenvalues span orders of magnitude
        "match_scale": False     # Set True for consistent color scaling across plots
    }
    
    try:
        # Initialize and run visualization pipeline
        pipeline = NTKVisualizationPipeline(**config)
        pipeline.generate_all_visualizations()
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()