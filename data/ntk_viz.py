import numpy as np
import matplotlib.pyplot as plt

FFNet_data = np.load("ntk_analysis_FFNet.npy", allow_pickle=True)
OCFFNet_data = np.load("ntk_analysis_OCFFNet.npy", allow_pickle=True)

iter_num = 5

def extract_eigs(arr, iter_num, count=10):
    """Grab eigenvalue_1..eigenvalue_count from a given iteration."""
    item = arr[iter_num]
    return [item[f"eigenvalue_{i}"] for i in range(1, count + 1)]

ffnet_eigenvalues = extract_eigs(FFNet_data, iter_num)
ocffnet_eigenvalues = extract_eigs(OCFFNet_data, iter_num)

def plot_eigenvalues(ff_vals, ocff_vals, iter_num=None, logy=False):
    """Plot FFNet vs OCFFNet eigenvalues for visual comparison."""
    if len(ff_vals) != len(ocff_vals):
        raise ValueError("ff_vals and ocff_vals must have the same length.")
    x = np.arange(1, len(ff_vals) + 1)

    plt.figure(figsize=(7, 4.5))
    plt.plot(x, ff_vals, marker="o", label="FFNet")
    plt.plot(x, ocff_vals, marker="s", label="OCFFNet")
    if logy:
        plt.yscale("log")
    plt.xlabel("Eigenvalue index")
    plt.ylabel("Eigenvalue")
    title = f"FFNet vs. OCFFNet Eigenvalues" + (f" (iter {iter_num})" if iter_num is not None else "")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"ntk_viz_FFNet_OCFFNet_eigenvalues_{iter_num}.png")

# call it
plot_eigenvalues(ffnet_eigenvalues, ocffnet_eigenvalues, iter_num=iter_num, logy=False)
