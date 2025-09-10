import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# ---- Load data (same as before) ----
FFNet_data = np.load("ntk_analysis_FFNet.npy", allow_pickle=True)
OCFFNet_data = np.load("ntk_analysis_OCFFNet.npy", allow_pickle=True)

# ---- Helpers ----
def extract_eigen_matrix(arr, top_k=10, iters=None):
    """
    Build a (num_iters, top_k) matrix where entry [t, k-1] = eigenvalue_k at iteration t.
    If 'iters' is None, uses all iterations.
    Missing keys become NaN so plotting can mask them.
    """
    if iters is None:
        iters = range(len(arr))   # use all available iterations
    iters = range(15)
    iters = list(iters)

    mat = np.full((len(iters), top_k), np.nan, dtype=float)
    for ci, it in enumerate(iters):
        item = arr[it]
        for k in range(1, top_k + 1):
            key = f"eigenvalue_{k}"
            if key in item:
                mat[ci, k - 1] = item[key]
    return mat, np.array(iters)

def plot_eigen_heatmaps(ff_mat, oc_mat, top_k=10, use_log=False, title_suffix=""):
    """
    Plot FFNet and OCFFNet heatmaps side-by-side.
    Vertical axis: eigen index (2..top_k). Horizontal: iteration index.
    Filters out eigenvalue_1 for plotting.
    """
    # Align iteration counts
    T = min(ff_mat.shape[0], oc_mat.shape[0])
    ff_mat = ff_mat[:T, :top_k]
    oc_mat = oc_mat[:T, :top_k]

    # ---- REMOVE eigenvalue_1: keep only columns 1..top_k-1 (i.e., eigenvalues 2..top_k)
    ff_slice = ff_mat[:, 1:top_k]
    oc_slice = oc_mat[:, 1:top_k]

    # Shared color scaling computed AFTER removal of the first eigenvalue
    combined = np.concatenate([ff_slice.ravel(), oc_slice.ravel()])
    finite_vals = combined[np.isfinite(combined)]
    if finite_vals.size == 0:
        raise ValueError("No finite eigenvalues to plot (after filtering).")

    if use_log:
        positives = finite_vals[finite_vals > 0]
        if positives.size == 0:
            raise ValueError("Log scale requested but no positive eigenvalues found.")
        vmin = max(positives.min(), np.finfo(float).eps)
        vmax = finite_vals.max()
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    def _plot(ax, mat, title):
        # We want rows=eigen index, cols=iteration, so transpose
        im = ax.imshow(mat.T, aspect="auto", origin="upper", interpolation="nearest", norm=norm)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Eigen index (2..{})".format(top_k))
        # y ticks: label 2..top_k
        ax.set_yticks(np.arange(top_k - 1))
        ax.set_yticklabels([str(i) for i in range(2, top_k + 1)])
        # x ticks: space them out
        step = max(1, T // 10)
        ax.set_xticks(np.arange(0, T, step))
        ax.set_xticklabels([str(i) for i in range(0, T, step)])
        return im

    im0 = _plot(axes[0], ff_slice, f"FFNet Eigenvalues{title_suffix}")
    im1 = _plot(axes[1], oc_slice, f"OCFFNet Eigenvalues{title_suffix}")

    # Shared colorbar
    cbar = fig.colorbar(im1, ax=axes.ravel().tolist())
    cbar.set_label("Eigenvalue" + (" (log scale)" if use_log else ""))

    plt.savefig(f"ntk_heatmap_filtered{title_suffix}.png", dpi=200, bbox_inches="tight")
    plt.close()

# ---- Build matrices (top 10 by default) ----
top_k = 10
ff_mat, ff_iters = extract_eigen_matrix(FFNet_data, top_k=top_k)
oc_mat, oc_iters = extract_eigen_matrix(OCFFNet_data, top_k=top_k)

# ---- Plot: linear scale, eigenvalues 2..10 only ----
plot_eigen_heatmaps(ff_mat, oc_mat, top_k=top_k, use_log=False, title_suffix="")

# ---- Optional: log-intensity version ----
# plot_eigen_heatmaps(ff_mat, oc_mat, top_k=top_k, use_log=True, title_suffix=" (log)")
