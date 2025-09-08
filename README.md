# OC-INR: Optimal Control Implicit Neural Representations

A PyTorch Lightning implementation of optimal control-regularized implicit neural representations (INRs) for learning continuous function approximations from discrete data. This repository implements both standard INR architectures and their optimal control (OC) variants with transport regularization.

## 🚀 Features

- **Multiple INR Architectures**:
  - **FFNet**: Random Fourier feature mappings for high-frequency learning
  - **SIREN**: Sinusoidal representation networks with periodic activations  
  - **OC-FFNet**: Optimal control variant of Fourier Feature Networks with transport regularization
  - **OC-SIREN**: Optimal control variant of SIREN networks

- **Optimal Transport Regularization**: ODE-based dynamics with transport cost minimization for improved training stability and generalization

## 📋 Requirements

Based on the codebase analysis, the following dependencies are required:

```bash
# Core ML framework
torch>=1.9.0
lightning>=2.0.0
torchmetrics

# Configuration and utilities  
hydra-core>=1.3.0
omegaconf
rootutils

# Data processing and visualization
numpy
matplotlib
scienceplots
pandas

# Logging
wandb  # for experiment tracking
lightning-utilities
```

## 🛠️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/OCINR.git
cd OCINR
```

2. **Install dependencies**:
```bash
pip install torch lightning torchmetrics hydra-core omegaconf rootutils numpy matplotlib scienceplots pandas wandb lightning-utilities
```

3. **Set up the project root**:
```bash
export PROJECT_ROOT=$(pwd)
```

## 📊 Dataset

The project uses four scientific datasets:

- **Turbulence Data**: `data/turbulence_1024.npy` - 1024×1024 turbulence field
- **Time Projection Chamber Data**: 
- **Cryo-EM Data**:
- **Black Sea Data**:
- **ERA5 Data**:

## 🎯 Quick Start

### Training Models

Train different INR architectures using the provided scripts:

```bash
# Train standard Fourier Feature Network
python src/train.py model=FFNet

# Train SIREN network
python src/train.py model=SIREN

# Train OC-Fourier Feature Network  
python src/train.py model=OCFFNet

# Train OC-SIREN network
python src/train.py model=OCSIREN
```
## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- **SIREN**: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- **Fourier Features**: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
- **Optimal Control**: [Optimal Control for Transformer Architectures: Enhancing Generalization, Robustness and Efficiency](https://arxiv.org/abs/2505.13499)

## 🙏 Acknowledgments

Built with:
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) - Training framework
- [Hydra](https://hydra.cc/) - Configuration management  
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) - Project template
