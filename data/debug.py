import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import scienceplots

# ----------------------------
# Load data
# ----------------------------
FFNet_data = np.load("ntk_analysis_FFNet.npy", allow_pickle=True)
OCFFNet_data = np.load("ntk_analysis_OCFFNet.npy", allow_pickle=True)
SIREN_data = np.load("ntk_analysis_SIREN.npy", allow_pickle=True)
OCSIREN_data = np.load("ntk_analysis_OCSIREN.npy", allow_pickle=True)

# ----------------------------
# Helpers
# ----------------------------
def extract_eigen_matrix(arr, top_k=10, iters=None):
    """
    Return a matrix M of shape (num_iters, top_k), where
      M[t, k-1] = eigenvalue_k at iteration t.
    If 'iters' is None, use all iterations [0..len(arr)-1].
    Missing keys are filled with NaN.
    """
    if iters is None:
        iters = range(len(arr))
    # If you intend to cap to first 15 iterations, keep it safe:
    iters = range(min(15, len(arr)))
    iters = list(iters)
    iters = list(range(30, 60, 2))

    M = np.full((len(iters), top_k), np.nan, dtype=float)
    for ci, it in enumerate(iters):
        item = arr[it]
        for k in range(1, top_k + 1):
            key = f"eigenvalue_{k}"
            if key in item:
                M[ci, k - 1] = item[key]
    return M, np.array(iters)

def make_shared_norm(mats, top_k, use_log=False, match_scale=True):
    """
    Build a Normalize/LogNorm over the union of all matrices *after*
    removing eigenvalue_1 (i.e., using columns 1..top_k-1) so colors
    are comparable across separate figures.
    If match_scale=False, return None (each figure will autoscale).
    """
    if not match_scale:
        return None

    vals = []
    for m in mats:
        slice_ = m[:, 1:top_k]  # eigenvalues 2..top_k
        vals.append(slice_.ravel())

    vals = np.concatenate(vals)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return None

    if use_log:
        pos = vals[vals > 0]
        if pos.size == 0:
            raise ValueError("Log scale requested but no positive eigenvalues found.")
        return LogNorm(vmin=max(pos.min(), np.finfo(float).eps), vmax=vals.max())
    else:
        return Normalize(vmin=vals.min(), vmax=vals.max())

def plot_single_heatmap(mat, title, top_k=10, use_log=False, norm=None, savepath=None):
    """
    Plot one heatmap as its own figure.
    Rows: eigen index 2..top_k (top to bottom), relabeled as 1..top_k-1.
    Cols: iteration 0..last (left to right).
    """
    with plt.style.context('science'):
        # Remove the first eigenvalue (keep 2..top_k) and transpose so rows=eigen, cols=iteration
        data = (mat[:, 1:top_k]).T   # shape: (top_k-1, T)
        rows, cols = data.shape

        plt.figure(figsize=(5, 3.42))
        im = plt.imshow(
            data,
            cmap="PiYG_r",
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            norm=norm  # shared or None for autoscale
        )
        plt.title(title)
        plt.xlabel("Training Step")
        # ---- Y label and ticks start at 1 and end at top_k-1 ----
        plt.ylabel("NTK Eigenvalues")
        plt.yticks(np.arange(rows), [str(i) for i in range(1, top_k)])

        # x ticks across iterations (spread ~10 ticks max)
        step = max(1, cols // 10)
        plt.xticks(np.arange(0, cols, step), [str(i) for i in range(0, cols, step)])

        cbar = plt.colorbar(im)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=200, bbox_inches="tight")
        plt.close()

# ----------------------------
# Build matrices (top 11 eigenvalues to get 10 after dropping the first)
# ----------------------------
top_k = 10  # plotting will show indices 1..(top_k-1) i.e., 10 rows
ff_mat, ff_iters = extract_eigen_matrix(FFNet_data, top_k=top_k)
oc_mat, oc_iters = extract_eigen_matrix(OCFFNet_data, top_k=top_k)
siren_mat, siren_iters = extract_eigen_matrix(SIREN_data, top_k=top_k)
ocsiren_mat, ocsiren_iters = extract_eigen_matrix(OCSIREN_data, top_k=top_k)
# Align to the same iteration count if arrays differ in length
T = min(ff_mat.shape[0], oc_mat.shape[0])
ff_mat = ff_mat[:T, :top_k]
oc_mat = oc_mat[:T, :top_k]
siren_mat = siren_mat[:T, :top_k]
ocsiren_mat = ocsiren_mat[:T, :top_k]

# ----------------------------
# Plot separately (one-by-one)
# ----------------------------
use_log = False       # set True if values span orders of magnitude
match_scale = False   # set True to use the same color scale across both figures

shared_norm = make_shared_norm([ff_mat, oc_mat, siren_mat, ocsiren_mat], top_k=top_k, use_log=use_log, match_scale=match_scale)

# 1) FFNet (its own figure)
plot_single_heatmap(
    ff_mat,
    title=f"(a) FFNet",
    top_k=top_k,
    use_log=use_log,
    norm=shared_norm,
    savepath="ffnet_eigs_heatmap.png"
)

# 2) OCFFNet (its own figure)
plot_single_heatmap(
    oc_mat,
    title=f"(b) OCFFNet",
    top_k=top_k,
    use_log=use_log,
    norm=shared_norm,
    savepath="ocffnet_eigs_heatmap.png"
)

# 3) SIREN (its own figure)
plot_single_heatmap(
    siren_mat,
    title=f"(c) SIREN",
    top_k=top_k,
    use_log=use_log,
    norm=shared_norm,
    savepath="siren_eigs_heatmap.png"
)

# 4) OCSIREN (its own figure)
plot_single_heatmap(
    ocsiren_mat,
    title=f"(d) OCSIREN",
    top_k=top_k,
    use_log=use_log,
    norm=shared_norm,
    savepath="ocsiren_eigs_heatmap.png"
)