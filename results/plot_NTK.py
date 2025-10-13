import re
import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from scipy import stats
from scipy.optimize import curve_fit


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
effective_rank_ocffnet_withoutOT, condition_number_ocffnet_withoutOT, spectrum_decay_ocffnet_withoutOT, trace_ocffnet_withoutOT, eigenvalues_ocffnet_withoutOT = extract_ntk_lists("ntk_analysis_OCFFNet_withoutOT.npy")
epochs = list(2*np.arange(151))
epochs = epochs[1:]

with plt.style.context('science'):
    plt.figure(figsize=(5, 4))
    plt.plot(epochs, effective_rank_ffnet, linewidth=2.5, label="FFNet")
    plt.plot(epochs, effective_rank_ocffnet, linewidth=2.5, label="OCFFNet")
    plt.plot(epochs, effective_rank_ocffnet_withoutOT, linewidth=2.5, label="OCFFNet without OT")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.legend()
    plt.title("Effective Rank")
    plt.savefig("effective_rank.png", dpi=300)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.plot(epochs, condition_number_ffnet, linewidth=2.5, label="FFNet")
    plt.plot(epochs, condition_number_ocffnet, linewidth=2.5, label="OCFFNet")
    plt.plot(epochs, condition_number_ocffnet_withoutOT, linewidth=2.5, label="OCFFNet without OT")
    plt.grid(True, alpha=0.3)
    plt.yscale("log")
    plt.legend()
    plt.title("Condition Number")
    plt.savefig("condition_number.png", dpi=300)
    plt.close()

    # Select a few representative epochs for detailed analysis
    select_epochs = [0, 25, 50, 100]  # epochs 0, 50, 100, 300
    
    # Distribution types to fit
    distributions = {
        'lognormal': ('Log-normal', lambda data: stats.lognorm.fit(data, floc=0), 
                     lambda x, params: stats.lognorm.pdf(x, *params)),
        'weibull': ('Weibull', lambda data: stats.weibull_min.fit(data, floc=0), 
                   lambda x, params: stats.weibull_min.pdf(x, *params))
    }
    
    for epoch_idx in select_epochs:
        models_data = [
            (eigenvalues_ffnet[epoch_idx], "FFNet"),
            (eigenvalues_ocffnet[epoch_idx], "OCFFNet"),
            (eigenvalues_ocffnet_withoutOT[epoch_idx], "OCFFNet without OT")
        ]

        # Determine global x range for all models
        all_eigenvals = []
        for eigenvals, _ in models_data:
            eigenvals_clean = np.array([x for x in eigenvals if not np.isnan(x) and x > 0])
            if len(eigenvals_clean) > 0:
                all_eigenvals.extend(eigenvals_clean)
        
        if len(all_eigenvals) == 0:
            continue
        
        x_range = np.logspace(np.log10(min(all_eigenvals)), 
                             np.log10(max(all_eigenvals)), 1000)
        
        # Create a separate plot for each distribution type
        for dist_key, (dist_name, fit_func, pdf_func) in distributions.items():
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            
            for eigenvals, model_name in models_data:
                # Remove NaN and filter positive values
                eigenvals_clean = np.array([x for x in eigenvals if not np.isnan(x) and x > 0])
                
                if len(eigenvals_clean) == 0:
                    continue
                
                # Fit distribution
                try:
                    params = fit_func(eigenvals_clean)
                    pdf_vals = pdf_func(x_range, params)
                    
                    # Plot fitted distribution only
                    ax.plot(x_range, pdf_vals, linestyle='-', 
                           linewidth=2.5, label=f'{model_name}')
                except Exception as e:
                    print(f"Failed to fit {dist_name} for {model_name} at epoch {epoch_idx}: {e}")
                    continue
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Eigenvalue', fontsize=12)
            ax.set_ylabel('Probability Density', fontsize=12)
            ax.set_title(f'NTK Eigenvalue Distribution at Epoch {2*epoch_idx}', fontsize=14)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f"eigenvalues_{dist_key}_epoch_{epoch_idx}.png", dpi=300)
            plt.close()