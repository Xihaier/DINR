import pandas as pd
from typing import Dict, List, Optional, Tuple

def get_rel_error_lists(
    csv_path: str,
    include_steps: bool = True
) -> Tuple[Optional[List[float]], Dict[str, List[float]]]:
    """
    Reads the CSV and returns:
      - steps: list of x-values from 'trainer/global_step' (or None if not present/asked)
      - rel_errors: dict with keys 'fresh_sun_3', 'royal_fire_2', 'floral_breeze_1'
        mapping to lists of float values for each run's train/rel_error.

    If include_steps is True and the step column exists, rows are aligned and any rows
    with missing rel_error values are dropped consistently across all three series.
    """
    rel_cols_exact = {
        "fresh_sun_3": "fresh-sun-3 - train/rel_error",
        "royal_fire_2": "royal-fire-2 - train/rel_error",
        "floral_breeze_1": "floral-breeze-1 - train/rel_error",
    }

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    have_cols = [c for c in rel_cols_exact.values() if c in df.columns]
    if not have_cols:
        raise ValueError("None of the expected rel_error columns were found.")

    # Ensure numeric dtypes
    for c in have_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    steps_col = "trainer/global_step" if include_steps and "trainer/global_step" in df.columns else None
    if steps_col:
        # Keep rows where all available rel_error columns are present
        subset_cols = [steps_col] + have_cols
        sub = df[subset_cols].dropna(subset=have_cols)
        steps = pd.to_numeric(sub[steps_col], errors="coerce").dropna().tolist()
        rel_errors = {
            k: sub[v].astype(float).tolist() if v in sub.columns else []
            for k, v in rel_cols_exact.items()
        }
        return steps, rel_errors
    else:
        # No steps requested or available; just return lists per series (dropping NaNs individually)
        steps = None
        rel_errors = {
            k: df[v].dropna().astype(float).tolist() if v in df.columns else []
            for k, v in rel_cols_exact.items()
        }
        return steps, rel_errors

# ---- Example usage ----
steps, rel = get_rel_error_lists("data.csv", include_steps=True)
# Now you can plot:
import matplotlib.pyplot as plt
import scienceplots


with plt.style.context('science'):
    plt.figure(figsize=(6, 4))
    plt.plot(steps[250:], rel["fresh_sun_3"][250:])
    plt.plot(steps[250:], rel["royal_fire_2"][250:])
    plt.plot(steps[250:], rel["floral_breeze_1"][250:])
    plt.yscale("log")
    plt.legend(["OC-FFNet without OT", "OC-FFNet with OT", "FFNet"])
    plt.title("Relative Error")
    plt.xlabel("Steps")
    plt.ylabel("Relative Error")
    plt.savefig("relative_error.png", dpi=300)
    plt.close()
