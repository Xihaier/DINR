import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def visualize_sampled_data(sampled_indices, grid_size, epoch, method_name, save_dir):
    """
    Visualize sampled data points on the original grid and save the plot.

    Args:
        sampled_indices (torch.Tensor): Indices of sampled points.
        grid_size (int): Size of the 2D grid (e.g., 128).
        epoch (int): Current epoch number.
        method_name (str): Name of the sampling method.
        save_dir (str): Directory to save the plots.

    Returns:
        None: Displays and saves the plot.
    """
    # Convert 1D indices to 2D grid coordinates
    sampled_indices = sampled_indices.cpu().numpy()
    rows, cols = np.divmod(sampled_indices, grid_size)

    # Create a blank grid and mark sampled points
    sampled_grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    sampled_grid[rows, cols] = 1  # Set sampled points to 1

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Visualize the sampled points
    with plt.style.context('science'):
        plt.figure(figsize=(3.5, 3.5))
        plt.imshow(sampled_grid, cmap="GnBu", origin="lower")
        plt.title(f"{method_name} (Epoch {epoch})")

        # Remove x-ticks, y-ticks, x-label, and y-label
        plt.xticks([])
        plt.yticks([])
        plt.xlabel("")
        plt.ylabel("")

        # Save the plot
        file_name = f"{method_name}_epoch{epoch:03d}.png".replace(" ", "")
        file_path = os.path.join(save_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()  # Close the plot to free memory