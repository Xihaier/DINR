# DINR: Dynamical Implicit Neural Representations

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This is the offical PyTorch implementation of **DINR**  Dynamical Implicit Neural Representations for learning continuous representations of complex scientific data.

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Configuration](#️-configuration)
- [Citation](#-citation)
- [License](#-license)

---

## 🔧 Installation

### Prerequisites

- Python 3.12+
- CUDA 11.8+ (for GPU support)
- conda or mamba (recommended for environment management)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/DINR.git
   cd DINR
   ```

2. **Create conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate DINR
   ```

3. **Prepare data**
   Place your `.npy` data files in the `data/` directory:
   ```
   data/
   ├── turbulence_1024.npy
   ├── ctbl3d.npy
   ├── water_vapor.npy
   └── emd_32218.npy
   ```

---

## 🚀 Quick Start

### Basic Training

Train a Fourier Feature Network on turbulence data:
```bash
python src/train.py data=turbulence model=FFNet
```

Train a Dynamical FFNet:
```bash
python src/train.py data=turbulence model=DynamicalFFNet
```

### Run All Experiments

Use the provided script to train all model variants:
```bash
bash scripts/run.sh
```

### Evaluation

Evaluate a trained model:
```bash
python src/eval.py \
  data=turbulence \
  model=FFNet \
  ckpt_path=logs/ntk/FFNet/checkpoints/best.ckpt
```

---

## 📁 Project Structure

```
DINR/
├── configs/                    # Hydra configuration files
│   ├── callbacks/             # Training callbacks (checkpointing, early stopping)
│   ├── data/                  # Dataset configurations
│   ├── model/                 # Model architecture configs
│   │   ├── FFNet.yaml
│   │   ├── SIREN.yaml
│   │   ├── DynamicalFFNet.yaml
│   │   └── DynamicalSIREN.yaml
│   ├── trainer/               # PyTorch Lightning trainer configs
│   ├── logger/                # Logging configurations (W&B)
│   ├── train.yaml             # Main training configuration
│   └── eval.yaml              # Evaluation configuration
│
├── data/                       # Data directory (*.npy files, gitignored)
│   ├── turbulence_1024.npy
│   ├── ctbl3d.npy
│   └── ...
│
├── src/                        # Source code
│   ├── data/
│   │   └── datamodule.py      # Lightning DataModule with NTK subset support
│   ├── models/
│   │   ├── components/        # Model architectures
│   │   │   ├── FFNet.py              # Fourier Feature Network
│   │   │   ├── SIRENNet.py           # SIREN Network
│   │   │   ├── Dynamical_FFNet.py    # OC-FFNet
│   │   │   └── Dynamical_SIRENNet.py # OC-SIREN
│   │   └── modelmodule.py     # Lightning modules (INRTraining, DINRTraining)
│   ├── utils/
│   │   ├── ntk.py             # Neural Tangent Kernel analysis
│   │   ├── metrics.py         # Loss and error metrics
│   │   ├── viz.py             # Visualization utilities
│   │   └── ...                # Various utilities
│   ├── train.py               # Training entry point
│   └── eval.py                # Evaluation entry point
│
├── scripts/
│   └── run.sh                 # Batch training script
│
├── logs/                       # Training outputs (gitignored)
│   └── ntk/                   # Organized by experiment name
│
├── environment.yml             # Conda environment specification
├── .gitignore
├── .project-root              # Root marker for rootutils
└── README.md
```

---

## ⚙️ Configuration

DINR uses [Hydra](https://hydra.cc/) for configuration management. All configurations are in the `configs/` directory.

### Key Configuration Files

#### Model Configuration (`configs/model/`)

**FFNet.yaml** - Traditional Fourier Feature Network
```yaml
net:
  _target_: src.models.components.FFNet.FourierFeatureNetwork
  input_dim: 2
  mapping_size: 256      # Fourier feature dimension
  hidden_dim: 256
  num_layers: 5
  output_dim: 1
  sigma: 10.0           # Fourier feature scale
  dropout_rate: 0.1
  activation: "GELU"
  use_residual: true
```

**DynamicalFFNet.yaml** - Dynamical FFNet
```yaml
net:
  _target_: src.models.components.Dynamical_FFNet.DynamicalFourierFeatureNetwork
  input_dim: 2
  mapping_size: 256
  hidden_dim: 256
  num_layers: 3          # ODE function layers
  num_steps: 12          # ODE integration steps
  total_time: 1.0        # Integration time horizon
  ot_lambda: 0.1         # Optimal transport weight
  block_type: "residual"
```

#### Data Configuration (`configs/data/turbulence.yaml`)

```yaml
_target_: src.data.datamodule.DataModule
data_dir: ${paths.data_dir}turbulence_1024.npy
in_features: 2
normalization: min-max
data_shape: [1024, 1024]
batch_size: [65536, 65536, 65536]  # [train, val, test]
ntk_subset_mode: subgrid           # NTK coordinate sampling
ntk_subgrid_g: 32                  # NTK grid resolution
generalization_test: false
```

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@software{dinr2025,
  title={coming soon}
}
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- PyTorch Lightning team for the excellent training framework
- Hydra team for flexible configuration management
- Authors of FFNet and SIREN for foundational INR architectures
- The neural ODE community for continuous-depth architecture inspiration

---

## 📞 Contact

- **Email**: xluo@bnl.gov

---

**Note**: This project is under active development. Star ⭐ the repository to stay updated!