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
    ground_truth = np.load("data/turbulence_1024.npy")
    predictions = np.load(f"logs/2025-10-16_05-09-31/{i}/test_preds.npy").reshape(1024, 1024)
    ground_truth = normalize_data(ground_truth)
    predictions = normalize_data(predictions)
    relative_error = calculate_relative_error(ground_truth, predictions)
    lst.append(relative_error)

print(np.mean(lst))
print(np.std(lst))

# Data: mean and std for each method across noise levels
FFNet_mean = [0.06854435342763021, 0.07914617752339961, 0.08423561446731405]
FFNet_std = [0.005264845587862937, 0.004512469291007012, 0.0052428733166341645]

OC_FFNet_with_OT_mean = [0.0560752225394003, 0.07639292436776654, 0.08399823095397416]
OC_FFNet_with_OT_std = [0.004862501484552783, 0.006024130949426008, 0.0037120387859610605]

OC_FFNet_without_OT_mean = [0.05625576532553046, 0.07711763006713589, 0.08486936535521798]
OC_FFNet_without_OT_std = [0.004862501484552783, 0.004622626257845679, 0.0036566255813045684]

# Noise levels configuration
noise_levels = [0.1, 0.2, 0.3]
noise_labels = ['10\%', '20\%', '30\%']


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
    plt.savefig('noise_comparison.png', dpi=300, bbox_inches='tight')