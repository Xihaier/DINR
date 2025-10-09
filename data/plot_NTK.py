import re
import numpy as np
import matplotlib.pyplot as plt
import scienceplots


def extract_ntk_lists(path):
    arr = np.load(path, allow_pickle=True)
    entries = list(arr)

    # Keep chronological order; if 'epoch' exists, sort by it
    if entries and all('epoch' in e for e in entries):
        entries = sorted(entries, key=lambda d: d['epoch'])

    # Scalars as lists (length = #entries, e.g., 151)
    effective_rank    = [float(e.get('effective_rank', np.nan)) for e in entries]
    condition_number  = [float(e.get('condition_number', np.nan)) for e in entries]
    spectrum_decay    = [float(e.get('spectrum_decay', np.nan)) for e in entries]
    trace             = [float(e.get('trace', np.nan)) for e in entries]

    # Determine how many eigenvalue_i keys exist (e.g., up to 1024)
    max_idx = 0
    for e in entries:
        for k in e.keys():
            m = re.match(r'^eigenvalue_(\d+)$', k)
            if m:
                max_idx = max(max_idx, int(m.group(1)))

    # List-of-lists: one list per epoch, each length = max_idx (e.g., 1024)
    eigenvalues = []
    for e in entries:
        vals = [float(e.get(f'eigenvalue_{i}', np.nan)) for i in range(1, max_idx + 1)]
        eigenvalues.append(vals)

    return effective_rank, condition_number, spectrum_decay, trace, eigenvalues

# ---- usage ----
effective_rank_ffnet, condition_number_ffnet, spectrum_decay_ffnet, trace_ffnet, eigenvalues_ffnet = extract_ntk_lists("ntk_analysis_FFNet.npy")
effective_rank_ocffnet, condition_number_ocffnet, spectrum_decay_ocffnet, trace_ocffnet, eigenvalues_ocffnet = extract_ntk_lists("ntk_analysis_OCFFNet.npy")

# with plt.style.context('science'):
#     plt.figure(figsize=(6, 4))
#     plt.plot(effective_rank_ffnet, label="FFNet")
#     plt.plot(effective_rank_ocffnet, label="OCFFNet")
#     plt.legend()
#     plt.title("effective_rank")
#     plt.savefig("effective_rank.png", dpi=300)
#     plt.close()

#     plt.figure(figsize=(6, 4))
#     plt.plot(condition_number_ffnet, label="FFNet")
#     plt.plot(condition_number_ocffnet, label="OCFFNet")
#     plt.legend()
#     plt.title("condition_number")
#     plt.savefig("condition_number.png", dpi=300)
#     plt.close()

#     plt.figure(figsize=(6, 4))
#     plt.plot(spectrum_decay_ffnet, label="FFNet")
#     plt.plot(spectrum_decay_ocffnet, label="OCFFNet")
#     plt.legend()
#     plt.title("spectrum_decay")
#     plt.savefig("spectrum_decay.png", dpi=300)
#     plt.close()

#     plt.figure(figsize=(6, 4))
#     plt.plot(trace_ffnet, label="FFNet")
#     plt.plot(trace_ocffnet, label="OCFFNet")
#     plt.legend()
#     plt.title("trace")
#     plt.savefig("trace.png", dpi=300)
#     plt.close()

#     for i in range(50):
#         plt.figure(figsize=(6, 4))
#         plt.plot(eigenvalues_ffnet[i], label="FFNet")
#         plt.plot(eigenvalues_ocffnet[i], label="OCFFNet")
#         plt.yscale("log") 
#         plt.legend()
#         plt.title(f"eigenvalues at epoch {2*i}")
#         plt.savefig(f"eigenvalues_{i}.png", dpi=300)
#         plt.close()


eigenvalues_ffnet = eigenvalues_ffnet[100]
eigenvalues_ocffnet = eigenvalues_ocffnet[100]

ratio_ffnet = np.sum(eigenvalues_ffnet[:250]) / np.sum(eigenvalues_ffnet)
ratio_ocffnet = np.sum(eigenvalues_ocffnet[:250]) / np.sum(eigenvalues_ocffnet)

print(f"FFNet ratio: {ratio_ffnet}")
print(f"OCFFNet ratio: {ratio_ocffnet}")