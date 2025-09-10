"""
NTK Results Visualization
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


class NTKSpectrumPlotter:
    """Handles creation of NTK eigenvalue spectrum comparison plots."""
    
    def __init__(
        self,
        figure_size: Tuple[float, float] = (7, 4.2),
        dpi: int = DEFAULT_DPI
    ):
        """
        Initialize the spectrum plotter.
        
        Args:
            figure_size: Figure dimensions as (width, height) in inches
            dpi: Resolution for saved figures
        """
        self.figure_size = figure_size
        self.dpi = dpi
    
    def plot_eigenvalue_spectrum_comparison(
        self,
        eigenvalue_matrices: Dict[str, np.ndarray],
        iteration_index: int,
        num_eigenvalues: int = DEFAULT_TOP_K,
        skip_first_eigenvalue: bool = False,
        use_log_scale: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        title_suffix: str = ""
    ) -> None:
        """
        Create a comparison plot of eigenvalue spectra across network architectures.
        
        This function generates a line plot comparing the eigenvalue spectra of different
        network architectures at a specific training iteration. Each network is represented
        by a different line with distinct markers.
        
        Args:
            eigenvalue_matrices: Dictionary mapping network names to eigenvalue matrices
            iteration_index: Training iteration to visualize (0-based, negative indices allowed)
            num_eigenvalues: Number of eigenvalues to plot
            skip_first_eigenvalue: If True, exclude the largest eigenvalue from visualization (default: False)
            use_log_scale: Whether to use logarithmic scale for y-axis
            output_path: Path to save the figure. If None, figure is displayed
            title_suffix: Additional text to append to the plot title
            
        Raises:
            IndexError: If iteration_index is out of valid range
            ValueError: If no valid eigenvalue data is found
        """
        if not eigenvalue_matrices:
            raise ValueError("No eigenvalue matrices provided")
        
        # Determine common iteration range across all matrices
        min_iterations = min(matrix.shape[0] for matrix in eigenvalue_matrices.values())
        
        # Handle negative indexing
        if iteration_index < 0:
            iteration_index = min_iterations + iteration_index
        
        if not (0 <= iteration_index < min_iterations):
            raise IndexError(
                f"Iteration index {iteration_index} out of range [0, {min_iterations-1}]"
            )
        
        # Determine eigenvalue slice
        start_col = 1 if skip_first_eigenvalue else 0
        max_available_eigenvalues = min(
            matrix.shape[1] for matrix in eigenvalue_matrices.values()
        ) - start_col
        
        actual_num_eigenvalues = min(num_eigenvalues, max_available_eigenvalues)
        if actual_num_eigenvalues <= 0:
            raise ValueError("No eigenvalues available for plotting")
        
        col_slice = slice(start_col, start_col + actual_num_eigenvalues)
        
        # Extract eigenvalues for the specified iteration
        eigenvalue_data = {}
        for network_name, matrix in eigenvalue_matrices.items():
            eigenvalues = matrix[iteration_index, col_slice]
            # Filter out NaN values
            if not np.all(np.isnan(eigenvalues)):
                eigenvalue_data[network_name] = eigenvalues
        
        if not eigenvalue_data:
            raise ValueError(f"No valid eigenvalue data found for iteration {iteration_index}")
        
        # Create the plot
        with plt.style.context('science'):
            plt.figure(figsize=self.figure_size)
            
            # Define colors and markers for each network type
            plot_styles = {
                "FFNet": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
                "OCFFNet": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
                "SIREN": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
                "OCSIREN": {"color": "#d62728", "marker": "D", "linestyle": ":"}
            }
            
            # X-axis represents eigenvalue ranks
            x_values = np.arange(1, actual_num_eigenvalues + 1)
            
            # Plot each network's eigenvalue spectrum
            for network_name, eigenvalues in eigenvalue_data.items():
                style = plot_styles.get(network_name, {"color": "black", "marker": "o", "linestyle": "-"})
                plt.plot(
                    x_values,
                    eigenvalues,
                    label=network_name,
                    marker=style["marker"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    markersize=6,
                    linewidth=2,
                    alpha=0.8
                )
            
            # Configure plot appearance
            if use_log_scale:
                plt.yscale("log")
                ylabel = "Eigenvalue (log scale)"
            else:
                ylabel = "Eigenvalue"
            
            # Set labels and title
            if skip_first_eigenvalue:
                xlabel = f"Eigenvalue Rank (excluding largest, showing ranks 2-{actual_num_eigenvalues+1})"
                title_core = f"Top {actual_num_eigenvalues} Eigenvalues (excluding largest)"
            else:
                xlabel = f"Eigenvalue Rank (1-{actual_num_eigenvalues})"
                title_core = f"Top {actual_num_eigenvalues} Eigenvalues (including largest)"
            
            plt.xlabel(xlabel, fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            
            title = f"{title_core} at Iteration {iteration_index}"
            if title_suffix:
                title += f" {title_suffix}"
            plt.title(title, fontsize=12, pad=10)
            
            # Add grid and legend
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            
            # Apply tight layout
            plt.tight_layout()
            
            # Save or display
            if output_path is not None:
                output_path = Path(output_path)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
                logger.info(f"Saved spectrum comparison to {output_path}")
                plt.close()
            else:
                plt.show()
    
    def plot_eigenvalue_evolution(
        self,
        eigenvalue_matrices: Dict[str, np.ndarray],
        eigenvalue_rank: int,
        skip_first_eigenvalue: bool = False,
        use_log_scale: bool = False,
        output_path: Optional[Union[str, Path]] = None,
        title_suffix: str = ""
    ) -> None:
        """
        Plot the evolution of a specific eigenvalue rank across training iterations.
        
        Args:
            eigenvalue_matrices: Dictionary mapping network names to eigenvalue matrices
            eigenvalue_rank: Rank of eigenvalue to track (1-indexed, after optional skipping)
            skip_first_eigenvalue: If True, exclude the largest eigenvalue (default: False)
            use_log_scale: Whether to use logarithmic scale for y-axis
            output_path: Path to save the figure. If None, figure is displayed
            title_suffix: Additional text to append to the plot title
        """
        if not eigenvalue_matrices:
            raise ValueError("No eigenvalue matrices provided")
        
        # Determine eigenvalue column index
        col_index = eigenvalue_rank - 1
        if skip_first_eigenvalue:
            col_index += 1  # Skip the first eigenvalue
        
        # Check if the requested eigenvalue rank is available
        min_eigenvalues = min(matrix.shape[1] for matrix in eigenvalue_matrices.values())
        if col_index >= min_eigenvalues:
            raise ValueError(f"Eigenvalue rank {eigenvalue_rank} not available")
        
        # Determine common iteration range
        min_iterations = min(matrix.shape[0] for matrix in eigenvalue_matrices.values())
        iteration_range = np.arange(min_iterations)
        
        # Create the plot
        with plt.style.context('science'):
            plt.figure(figsize=self.figure_size)
            
            # Define plot styles
            plot_styles = {
                "FFNet": {"color": "#1f77b4", "marker": "o", "linestyle": "-"},
                "OCFFNet": {"color": "#ff7f0e", "marker": "s", "linestyle": "--"},
                "SIREN": {"color": "#2ca02c", "marker": "^", "linestyle": "-."},
                "OCSIREN": {"color": "#d62728", "marker": "D", "linestyle": ":"}
            }
            
            # Plot eigenvalue evolution for each network
            for network_name, matrix in eigenvalue_matrices.items():
                eigenvalue_evolution = matrix[:min_iterations, col_index]
                
                # Filter out NaN values for plotting
                valid_mask = np.isfinite(eigenvalue_evolution)
                if np.any(valid_mask):
                    style = plot_styles.get(network_name, {"color": "black", "marker": "o", "linestyle": "-"})
                    plt.plot(
                        iteration_range[valid_mask],
                        eigenvalue_evolution[valid_mask],
                        label=network_name,
                        marker=style["marker"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        markersize=4,
                        linewidth=2,
                        alpha=0.8,
                        markevery=max(1, len(iteration_range) // 20)  # Reduce marker density
                    )
            
            # Configure plot appearance
            if use_log_scale:
                plt.yscale("log")
                ylabel = "Eigenvalue (log scale)"
            else:
                ylabel = "Eigenvalue"
            
            plt.xlabel("Training Iteration", fontsize=10)
            plt.ylabel(ylabel, fontsize=10)
            
            # Create title
            actual_rank = eigenvalue_rank + (1 if skip_first_eigenvalue else 0)
            title = f"Evolution of Eigenvalue Rank {actual_rank}"
            if title_suffix:
                title += f" {title_suffix}"
            plt.title(title, fontsize=12, pad=10)
            
            # Add grid and legend
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
            
            plt.tight_layout()
            
            # Save or display
            if output_path is not None:
                output_path = Path(output_path)
                plt.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
                logger.info(f"Saved eigenvalue evolution to {output_path}")
                plt.close()
            else:
                plt.show()


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
        self.heatmap_plotter = NTKHeatmapPlotter()
        self.spectrum_plotter = NTKSpectrumPlotter()
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
                self.heatmap_plotter.plot_eigenvalue_heatmap(
                    eigenvalue_matrix=aligned_matrices[network_name],
                    title=plot_title,
                    top_k=self.top_k,
                    normalization=normalization,
                    output_path=output_path
                )
            else:
                logger.warning(f"No data found for {network_name}")
        
        logger.info("Heatmap visualization completed successfully")
        return aligned_matrices  # Return for potential use in spectrum plots
    
    def generate_spectrum_comparisons(
        self,
        eigenvalue_matrices: Optional[Dict[str, np.ndarray]] = None,
        iterations_to_plot: Optional[List[int]] = None,
        eigenvalue_ranks_to_track: Optional[List[int]] = None
    ) -> None:
        """
        Generate eigenvalue spectrum comparison plots.
        
        Args:
            eigenvalue_matrices: Pre-computed eigenvalue matrices. If None, will load and process data
            iterations_to_plot: List of iteration indices for spectrum comparisons. If None, uses [0, 1, 2, 3, 4, 5]
            eigenvalue_ranks_to_track: List of eigenvalue ranks to track evolution. If None, uses [1, 2, 5]
        """
        logger.info("Starting spectrum comparison generation")
        
        # Load matrices if not provided
        if eigenvalue_matrices is None:
            raw_data = self.data_loader.load_data()
            matrices = {}
            for network_name, data_array in raw_data.items():
                matrix, _ = extract_eigenvalue_matrix(data_array, top_k=self.top_k)
                matrices[network_name] = matrix
            eigenvalue_matrices = self._align_matrices_to_common_length(matrices)
        
        # Default iterations to plot (iterations 0 through 5)
        if iterations_to_plot is None:
            min_iterations = min(matrix.shape[0] for matrix in eigenvalue_matrices.values())
            max_iter = min(5, min_iterations - 1)  # Ensure we don't exceed available iterations
            iterations_to_plot = list(range(0, max_iter + 1))
        
        # Generate spectrum comparison plots for specified iterations
        for iteration_idx in iterations_to_plot:
            # Linear scale comparison
            output_path = self.output_dir / f"spectrum_comparison_iter_{iteration_idx}_linear.png"
            self.spectrum_plotter.plot_eigenvalue_spectrum_comparison(
                eigenvalue_matrices=eigenvalue_matrices,
                iteration_index=iteration_idx,
                num_eigenvalues=5,  # Show only first 5 eigenvalues
                skip_first_eigenvalue=False,
                use_log_scale=False,
                output_path=output_path,
                title_suffix="(Linear Scale)"
            )
            
            # Log scale comparison
            output_path = self.output_dir / f"spectrum_comparison_iter_{iteration_idx}_log.png"
            self.spectrum_plotter.plot_eigenvalue_spectrum_comparison(
                eigenvalue_matrices=eigenvalue_matrices,
                iteration_index=iteration_idx,
                num_eigenvalues=5,  # Show only first 5 eigenvalues
                skip_first_eigenvalue=False,
                use_log_scale=True,
                output_path=output_path,
                title_suffix="(Log Scale)"
            )
        
        # Generate eigenvalue evolution plots
        if eigenvalue_ranks_to_track is None:
            eigenvalue_ranks_to_track = [1, 2, 5]  # Track 1st, 2nd, and 5th eigenvalues
        
        for rank in eigenvalue_ranks_to_track:
            if rank < self.top_k:
                # Linear scale evolution
                output_path = self.output_dir / f"eigenvalue_rank_{rank}_evolution_linear.png"
                self.spectrum_plotter.plot_eigenvalue_evolution(
                    eigenvalue_matrices=eigenvalue_matrices,
                    eigenvalue_rank=rank,
                    skip_first_eigenvalue=False,
                    use_log_scale=False,
                    output_path=output_path,
                    title_suffix="(Linear Scale)"
                )
                
                # Log scale evolution
                output_path = self.output_dir / f"eigenvalue_rank_{rank}_evolution_log.png"
                self.spectrum_plotter.plot_eigenvalue_evolution(
                    eigenvalue_matrices=eigenvalue_matrices,
                    eigenvalue_rank=rank,
                    skip_first_eigenvalue=False,
                    use_log_scale=True,
                    output_path=output_path,
                    title_suffix="(Log Scale)"
                )
        
        logger.info("Spectrum comparison generation completed successfully")
    
    def generate_all_visualizations(
        self,
        include_spectrum_plots: bool = True,
        iterations_to_plot: Optional[List[int]] = None,
        eigenvalue_ranks_to_track: Optional[List[int]] = None
    ) -> None:
        """
        Generate all NTK eigenvalue visualizations.
        
        Args:
            include_spectrum_plots: Whether to generate spectrum comparison plots
            iterations_to_plot: Specific iterations for spectrum comparisons
            eigenvalue_ranks_to_track: Specific eigenvalue ranks to track evolution
        """
        logger.info("Starting comprehensive NTK eigenvalue visualization pipeline")
        
        # Generate heatmaps and get aligned matrices
        aligned_matrices = self.generate_heatmap_visualizations()
        
        # Generate spectrum plots if requested
        if include_spectrum_plots:
            self.generate_spectrum_comparisons(
                eigenvalue_matrices=aligned_matrices,
                iterations_to_plot=iterations_to_plot,
                eigenvalue_ranks_to_track=eigenvalue_ranks_to_track
            )
        
        logger.info("Complete visualization pipeline finished successfully")
    
    def generate_heatmap_visualizations(self) -> Dict[str, np.ndarray]:
        """Generate heatmap visualizations and return aligned matrices."""
        logger.info("Starting NTK eigenvalue heatmap generation")
        
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
                self.heatmap_plotter.plot_eigenvalue_heatmap(
                    eigenvalue_matrix=aligned_matrices[network_name],
                    title=plot_title,
                    top_k=self.top_k,
                    normalization=normalization,
                    output_path=output_path
                )
            else:
                logger.warning(f"No data found for {network_name}")
        
        logger.info("Heatmap generation completed successfully")
        return aligned_matrices


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
    
    # Visualization options
    visualization_options = {
        "include_spectrum_plots": True,  # Set False to generate only heatmaps
        "iterations_to_plot": list(range(6)),  # Iterations 0 through 5 for spectrum comparison
        "eigenvalue_ranks_to_track": [1, 2, 5]  # Track 1st, 2nd, and 5th eigenvalues
    }
    
    try:
        # Initialize and run visualization pipeline
        pipeline = NTKVisualizationPipeline(**config)
        pipeline.generate_all_visualizations(**visualization_options)
        
    except Exception as e:
        logger.error(f"Visualization pipeline failed: {e}")
        raise


def generate_heatmaps_only():
    """Convenience function to generate only heatmap visualizations."""
    config = {
        "data_dir": ".",
        "output_dir": ".",
        "top_k": DEFAULT_TOP_K,
        "use_log_scale": False,
        "match_scale": False
    }
    
    pipeline = NTKVisualizationPipeline(**config)
    pipeline.generate_heatmap_visualizations()


def generate_spectrum_plots_only():
    """Convenience function to generate only spectrum comparison plots."""
    config = {
        "data_dir": ".",
        "output_dir": ".",
        "top_k": DEFAULT_TOP_K,
        "use_log_scale": False,
        "match_scale": False
    }
    
    pipeline = NTKVisualizationPipeline(**config)
    pipeline.generate_spectrum_comparisons(
        iterations_to_plot=list(range(6)),  # Iterations 0 through 5
        eigenvalue_ranks_to_track=[1, 2, 5]
    )


if __name__ == "__main__":
    main()