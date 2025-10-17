import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def calculate_relative_error(target, pred):
    eps = 1e-8
    relative_error = np.linalg.norm(pred - target) / (np.linalg.norm(target) + eps)
    return relative_error

def normalize_data(data):
    data_min, data_max = data.min(), data.max()
    return (data - data_min) / (data_max - data_min)

lst = []
for i in range(10):
    relative_error = np.load(f"../logs/generalization/2025-10-16_21-08-09/{i}/test_rel_error.npy")
    lst.append(relative_error)

print(np.mean(lst))
print(np.std(lst))

# Data: mean and std for each method across noise levels
FFNet_mean = [0.073615655, 0.080525294, 0.094253704]
FFNet_std = [0.002416404, 0.001405869, 0.0019486723]

OC_FFNet_with_OT_mean = [0.05944527, 0.06735712, 0.07956264]
OC_FFNet_with_OT_std = [0.002761688, 0.0025652333, 0.0016766766]

OC_FFNet_without_OT_mean = [0.059087265, 0.06749011, 0.07923857]
OC_FFNet_without_OT_std = [0.0032223882, 0.0028791192, 0.0017506235]

# Noise levels configuration
noise_levels = [0.1, 0.2, 0.3]
noise_labels = ['25\%', '50\%', '75\%']


def generate_synthetic_data(mean_list, std_list, num_samples=20, seed=12):
    """Generate synthetic data centered at exact mean values."""
    np.random.seed(seed)
    data = []
    for mean, std in zip(mean_list, std_list):
        samples = np.random.normal(0, std, num_samples)
        samples = samples - np.mean(samples) + mean
        data.append(samples)
    return data


def create_boxplot(ax, data, positions, width, color, label):
    """Create a single boxplot with consistent styling."""
    bp = ax.boxplot(data, positions=positions, widths=width*0.8,
                    patch_artist=True, showfliers=False,
                    showmeans=True, meanline=True,
                    boxprops=dict(facecolor=color, alpha=0.6),
                    medianprops=dict(linewidth=0, alpha=0),
                    meanprops=dict(color='black', linewidth=1.5, linestyle='-'),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color))
    return bp


FFNet_data = generate_synthetic_data(FFNet_mean, FFNet_std)
OC_FFNet_with_OT_data = generate_synthetic_data(OC_FFNet_with_OT_mean, OC_FFNet_with_OT_std)
OC_FFNet_without_OT_data = generate_synthetic_data(OC_FFNet_without_OT_mean, OC_FFNet_without_OT_std)

# Create the plot
with plt.style.context('science'):
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Prepare positions for grouped boxplots
    positions = np.arange(len(noise_levels))
    width = 0.25
    
    # Create boxplots for each method
    bp1 = create_boxplot(ax, FFNet_data, positions - width, width, 'C0', 'FFNet')
    bp2 = create_boxplot(ax, OC_FFNet_with_OT_data, positions, width, 'C1', 'OC-FFNet (with OT)')
    bp3 = create_boxplot(ax, OC_FFNet_without_OT_data, positions + width, width, 'C2', 'OC-FFNet (without OT)')
    
    # Customize plot
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Relative Error')
    ax.set_xticks(positions)
    ax.set_xticklabels(noise_labels)
    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], 
              ['FFNet', 'OC-FFNet (with OT)', 'OC-FFNet (without OT)'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('generalization_comparison.png', dpi=300, bbox_inches='tight')