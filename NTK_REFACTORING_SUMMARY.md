# NTK Computation Pipeline Refactoring Summary

## Overview
This document summarizes the comprehensive refactoring of the NTK (Neural Tangent Kernel) computation pipeline to improve correctness, determinism, and usability.

## Changes Made

### 1. DataModule: Fixed NTK Coordinate Subset (`src/data/datamodule.py`)

**New Parameters:**
- `ntk_subset_mode: str = "subgrid"` - Mode for NTK subset selection ("subgrid" or "all")
- `ntk_subgrid_g: int = 32` - Grid size per dimension for uniform subgrid sampling

**New Methods:**
- `_build_ntk_subset()` - Builds a fixed, deterministic nD uniform subgrid from the canonical grid
  - For 2D: creates g×g grid; for D-dim: creates g^D grid
  - Uses row-major indexing for consistency
  - Supports "all" mode to use entire dataset
  
- `get_ntk_coords()` - Returns fixed NTK coordinates (CPU tensor)
  - Called by training modules to get consistent coordinates
  - Returns cloned tensor to prevent accidental modifications

**Behavior:**
- NTK coordinates are computed **once** during `setup()` and reused for all analyses
- Ensures determinism across epochs and runs
- Coordinates are sampled uniformly from the canonical rectilinear grid

### 2. NTK Analyzer: Bug Fixes & Improvements (`src/utils/ntk.py`)

#### Critical Multi-Output NTK Bug Fix
**Problem:** The original code incorrectly summed **all** output blocks (c,c′) when computing the NTK, leading to incorrect kernel values for multi-output networks.

**Solution:** Fixed the reduction to sum **only diagonal blocks** (c=c′):
```python
# OLD (INCORRECT):
K = J @ J.t()  # (N×C, N×C)
K = K.view(N, C, N, C).sum(dim=(1, 3))  # Wrong: sums all blocks

# NEW (CORRECT):
J = J_flat.view(N, C, P)  # Reshape to (N, C, P)
K = torch.einsum('icp,jcp->ij', J, J)  # Correct: sum over c and p
```

This implements the correct empirical NTK: K_ij = Σ_c ⟨∂f_c(x_i)/∂θ, ∂f_c(x_j)/∂θ⟩

#### Chunked Computation
- Removed incomplete chunked implementation
- Raises clear `NotImplementedError` with guidance when `chunk_size` is set
- Directs users to configure `ntk_subgrid_g` instead

#### Determinism & Sanity Checks
- Model automatically set to `eval()` mode during NTK computation
- Training state restored after computation
- Added eigenvalue count validation
- Added trace normalization sanity check (warns if trace ≠ 1.0 when using trace normalization)

#### API Clarity
- `return_all_eigenvalues` parameter documented as always True (kept for backward compatibility)
- Full spectrum always computed and returned in `NTKResult.eigenvalues`

### 3. Training Modules: Refactored NTK Integration (`src/models/modelmodule.py`)

**Applies to both `INRTraining` and `OCINRTraining` classes.**

#### Deprecated Parameters
- `ntk_subset_stride` - Now deprecated with warning
- `ntk_subset_size` - Now deprecated with warning
- Users should configure `ntk_subgrid_g` in DataModule instead

#### Refactored Methods

**`_capture_training_inputs()`:**
- Now a no-op (deprecated functionality)
- No longer accumulates training batches

**`_get_ntk_inputs()`:**
- Fetches fixed coordinates from `trainer.datamodule.get_ntk_coords()`
- Emits deprecation warnings once if old parameters are used
- Validates that DataModule and coordinates are available

**`_setup_ntk_analysis()`:**
- Validates DataModule provides NTK coordinates
- Prints informative message with coordinate count

**`_perform_ntk_analysis()`:**
- Uses fixed coordinates from DataModule
- **Logs only top-k eigenvalues** to prevent metric explosion
- **Saves full spectrum** to disk in `ntk_results_history`
- Model automatically in eval mode (handled by NTK analyzer)

**`training_step()`:**
- Removed call to `_capture_training_inputs()` (no longer needed)

## Migration Guide

### For Existing Configs

**Old Configuration (Deprecated):**
```yaml
model:
  ntk_analysis: true
  ntk_top_k: 10
  ntk_subset_stride: 10  # Deprecated
  ntk_subset_size: 1024  # Deprecated
```

**New Configuration:**
```yaml
data:
  ntk_subset_mode: "subgrid"  # or "all"
  ntk_subgrid_g: 32  # 32×32 grid for 2D, 32^D for D-dim

model:
  ntk_analysis: true
  ntk_top_k: 10  # Number of eigenvalues to log (full spectrum still saved)
```

### Key Benefits

1. **Determinism:** Fixed coordinates ensure reproducible NTK analysis across runs
2. **Correctness:** Multi-output NTK bug fixed
3. **Performance:** No batch accumulation overhead
4. **Usability:** Clear error messages and deprecation warnings
5. **Scalability:** Logs only top-k eigenvalues while saving full spectrum to disk
6. **Robustness:** Sanity checks catch common issues

## Logging Behavior

### Logger Metrics (Real-time)
- `ntk_effective_rank`
- `ntk_condition_number`
- `ntk_spectrum_decay`
- `ntk_eigenvalue_1` through `ntk_eigenvalue_k` (top-k only)

### Disk Storage (Complete)
- Full spectrum saved to `ntk_analysis.npy` in log directory
- Contains all eigenvalues for every epoch analyzed
- Format: List of dicts with keys: `epoch`, `effective_rank`, `condition_number`, `spectrum_decay`, `trace`, `total_eigenvalues`, `eigenvalue_1`, `eigenvalue_2`, ...

## Testing Recommendations

1. **Verify determinism:** Run same experiment twice, compare NTK results
2. **Check coordinates:** Ensure `get_ntk_coords()` returns expected grid
3. **Validate spectrum:** Check that eigenvalue count matches coordinate count
4. **Test multi-output:** Verify NTK values are correct for multi-output networks
5. **Performance:** Benchmark with different `ntk_subgrid_g` values

## Breaking Changes

- **Must update config:** Add `ntk_subgrid_g` to DataModule configuration
- **Old parameters ignored:** `ntk_subset_stride` and `ntk_subset_size` will emit warnings
- **Coordinate source changed:** NTK now uses DataModule's fixed grid instead of training batches

## Backward Compatibility

- Old parameters still accepted (with deprecation warnings)
- API signature unchanged for external callers
- Disk format for `ntk_analysis.npy` unchanged

