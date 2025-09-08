import numpy as np
import matplotlib.pyplot as plt
import torch
import scienceplots

plt.style.use('science')


def plot_model_comparison(y, yhat1, yhat2, model1_name="FFNet", model2_name="OC-FFNet"):
    """Create 2x3 figure panel with individual colorbars for each subplot"""
    with plt.style.context(["science"]):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Define color limits for consistent scaling
        data_vmin = min(y.min(), yhat1.min(), yhat2.min())
        data_vmax = max(y.max(), yhat1.max(), yhat2.max())
        
        # Calculate errors
        error1 = np.abs(y - yhat1)
        error2 = np.abs(y - yhat2)
        
        # Define error limits (starting from 0)
        error_vmax = max(error1.max(), error2.max())
        
        # Row 1: Ground Truth, FFNet Prediction, OC-FFNet Prediction
        im1 = axes[0, 0].imshow(y, cmap="coolwarm", vmin=data_vmin, vmax=data_vmax, aspect='equal')
        axes[0, 0].set_xticks([])
        axes[0, 0].set_yticks([])
        axes[0, 0].set_title("Ground Truth", fontweight='bold')
        
        im2 = axes[0, 1].imshow(yhat1, cmap="coolwarm", vmin=data_vmin, vmax=data_vmax, aspect='equal')
        axes[0, 1].set_xticks([])
        axes[0, 1].set_yticks([])
        axes[0, 1].set_title(model1_name, fontweight='bold')
        
        im3 = axes[0, 2].imshow(yhat2, cmap="coolwarm", vmin=data_vmin, vmax=data_vmax, aspect='equal')
        axes[0, 2].set_xticks([])
        axes[0, 2].set_yticks([])
        axes[0, 2].set_title(model2_name, fontweight='bold')
        
        # Row 2: Empty, FFNet Error, OC-FFNet Error
        # Empty subplot (placeholder)
        axes[1, 0].axis('off')
        axes[1, 0].set_title("", fontweight='bold')
        
        im4 = axes[1, 1].imshow(error1, cmap="plasma", vmin=0, vmax=error_vmax, aspect='equal')
        axes[1, 1].set_xticks([])
        axes[1, 1].set_yticks([])
        axes[1, 1].set_title(f"{model1_name} Error", fontweight='bold')
        
        im5 = axes[1, 2].imshow(error2, cmap="plasma", vmin=0, vmax=error_vmax, aspect='equal')
        axes[1, 2].set_xticks([])
        axes[1, 2].set_yticks([])
        axes[1, 2].set_title(f"{model2_name} Error", fontweight='bold')
        
        # Add individual colorbars for each subplot
        # Ground Truth colorbar
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
        cbar1.ax.tick_params(labelsize=10)
        
        # FFNet Prediction colorbar
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=10)
        
        # OC-FFNet Prediction colorbar
        cbar3 = plt.colorbar(im3, ax=axes[0, 2], fraction=0.046, pad=0.04)
        cbar3.ax.tick_params(labelsize=10)
        
        # FFNet Error colorbar
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
        cbar4.ax.tick_params(labelsize=10)
        
        # OC-FFNet Error colorbar
        cbar5 = plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)
        cbar5.ax.tick_params(labelsize=10)
        
        # Calculate metrics for annotations
        def calculate_psnr(gt, pred):
            mse = np.mean((gt - pred) ** 2)
            if mse == 0:
                return float('inf')
            max_pixel = max(gt.max(), pred.max())
            return 20 * np.log10(max_pixel / np.sqrt(mse))
        
        def calculate_mae(gt, pred):
            return np.mean(np.abs(gt - pred))
        
        # Calculate metrics
        psnr_ffnet = calculate_psnr(y, yhat1)
        psnr_ocffnet = calculate_psnr(y, yhat2)
        mae_ffnet = calculate_mae(y, yhat1)
        mae_ocffnet = calculate_mae(y, yhat2)
        
        # Add corner text annotations
        # FFNet prediction
        axes[0, 1].text(0.02, 0.98, f'PSNR: {psnr_ffnet:.1f} dB\nMAE: {mae_ffnet:.3f}', 
                       transform=axes[0, 1].transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.8))
        
        # OC-FFNet prediction  
        axes[0, 2].text(0.02, 0.98, f'PSNR: {psnr_ocffnet:.1f} dB\nMAE: {mae_ocffnet:.3f}', 
                       transform=axes[0, 2].transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.8))
        
        # FFNet error
        axes[1, 1].text(0.02, 0.98, f'Mean: {error1.mean():.3f}\nMax: {error1.max():.3f}', 
                       transform=axes[1, 1].transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.8))
        
        # OC-FFNet error
        axes[1, 2].text(0.02, 0.98, f'Mean: {error2.mean():.3f}\nMax: {error2.max():.3f}', 
                       transform=axes[1, 2].transAxes, fontsize=10, fontweight='bold',
                       verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', 
                       facecolor='white', alpha=0.8))
        
        # Adjust layout with reduced vertical spacing
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, 
                           wspace=0.3, hspace=0.15)
        
        plt.savefig("comparison.png", dpi=300, bbox_inches='tight')

# Load data
y = np.load("turbulence_1024.npy")
y = (y - y.min())/(y.max() - y.min())
yhat_ffnet = np.load("test_preds_FFNet.npy").reshape(1024, 1024)
yhat_ocffnet = np.load("test_preds_OCFFNet.npy").reshape(1024, 1024)

# Create comparison plot
plot_model_comparison(y, yhat_ffnet, yhat_ocffnet)

# Print comparison statistics
print("Model Comparison Statistics:")
print(f"FFNet - Min: {yhat_ffnet.min():.3f}, Max: {yhat_ffnet.max():.3f}, Mean: {yhat_ffnet.mean():.3f}")
print(f"OCFFNet - Min: {yhat_ocffnet.min():.3f}, Max: {yhat_ocffnet.max():.3f}, Mean: {yhat_ocffnet.mean():.3f}")
print(f"Prediction Difference - Mean: {np.abs(yhat_ffnet - yhat_ocffnet).mean():.3f}, Max: {np.abs(yhat_ffnet - yhat_ocffnet).max():.3f}")

# Calculate errors vs ground truth
error_ffnet = np.abs(y - yhat_ffnet)
error_ocffnet = np.abs(y - yhat_ocffnet)
print(f"FFNet Error - Mean: {error_ffnet.mean():.3f}, Max: {error_ffnet.max():.3f}")
print(f"OCFFNet Error - Mean: {error_ocffnet.mean():.3f}, Max: {error_ocffnet.max():.3f}")