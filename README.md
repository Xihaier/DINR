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

The project uses turbulence flow data stored as NumPy arrays:

- **Training data**: `data/turbulence_1024.npy` - 1024×1024 turbulence field
- **Data format**: 2D spatial coordinates → scalar field values
- **Normalization**: Min-max normalization applied automatically
- **Batch size**: Configurable per split (default: 65536 for training)

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

### Hyperparameter Sweeps

Run systematic hyperparameter exploration:

```bash
# Run the provided hyperparameter sweep for OC-SIREN
bash scripts/run.sh
```

### Model Evaluation

Evaluate trained models:

```bash
# Evaluate a specific checkpoint
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

### Visualization and Comparison

Compare model predictions:

```bash
# Generate comparison plots (requires prediction files)
python data/test.py
```

## 🏗️ Architecture Overview

### Standard INR Models

1. **FourierFeatureNetwork** (`src/models/components/FFNet.py`):
   - Random Fourier feature mapping for input coordinates
   - MLP with residual connections and configurable activations
   - Optimized for high-frequency function learning

2. **SIREN** (`src/models/components/SIRENNet.py`):
   - Sinusoidal activation functions throughout the network
   - Specialized weight initialization (ω₀ parameterization)
   - Effective for learning smooth, continuous functions

### Optimal Control Variants

3. **OCFourierFeatureNetwork** (`src/models/components/OC_FFNet.py`):
   - Fourier feature embedding + ODE dynamics
   - Transport cost regularization via Euler integration
   - Multiple time conditioning modes: concatenation, FiLM, time embeddings

4. **OCSIREN** (`src/models/components/OC_SIRENNet.py`):
   - SIREN-based ODE dynamics with sine activations
   - Optimal transport regularization with configurable λ parameter
   - Maintains SIREN initialization schemes within ODE framework

### Training Modules

- **INRTraining**: Standard supervised learning for FFNet and SIREN
- **OCINRTraining**: Multi-objective training with data loss + transport regularization

## ⚙️ Configuration

The project uses Hydra for configuration management. Key configuration files:

### Model Configurations (`configs/model/`)

- `FFNet.yaml`: Standard Fourier Feature Network settings
- `SIREN.yaml`: SIREN network with ω₀ parameters  
- `OCFFNet.yaml`: OC-Fourier network with ODE parameters
- `OCSIREN.yaml`: OC-SIREN with transport regularization

### Training Configuration (`configs/train.yaml`)

```yaml
defaults:
  - data: turbulence      # Data configuration
  - model: OCFFNet        # Model architecture  
  - trainer: default      # Lightning trainer settings
  - logger: wandb         # Experiment logging
  - callbacks: default    # Training callbacks

train: true               # Enable training
test: true               # Enable testing  
seed: 3407               # Reproducibility seed
```

### Key Hyperparameters

**Standard Models**:
- `hidden_dim`: Network width (default: 256)
- `num_layers`: Network depth (default: 5-7)
- `dropout_rate`: Dropout probability
- `activation`: Activation function ("GELU", "ReLU", etc.)

**OC Models**:
- `num_steps`: ODE integration steps (default: 12)
- `total_time`: Integration time horizon (default: 1.0)
- `ot_lambda`: Transport regularization weight (default: 0.001)
- `fusion_mode`: Time conditioning ("cat", "film", "film_time_embed")

**SIREN-specific**:
- `omega_0`: First layer frequency (default: 30.0)
- `omega_0_hidden`: Hidden layer frequency (default: 30.0)

## 📈 Monitoring and Logging

The project integrates with Weights & Biases for experiment tracking:

- **Metrics tracked**: Train/test loss, relative error, transport cost
- **Model info**: Parameter counts, architecture details
- **Artifacts**: Model checkpoints, predictions, visualizations

Configure logging in `configs/logger/wandb.yaml`:

```yaml
wandb:
  project: "OC-INR"           # W&B project name
  offline: False              # Enable online logging
  log_model: False           # Upload checkpoints
```

## 📁 Project Structure

```
OCINR/
├── configs/                 # Hydra configuration files
│   ├── model/              # Model architecture configs
│   ├── data/               # Dataset configurations  
│   ├── trainer/            # Lightning trainer settings
│   └── logger/             # Logging configurations
├── src/
│   ├── data/
│   │   └── datamodule.py   # Lightning data module
│   ├── models/
│   │   ├── components/     # Network architectures
│   │   │   ├── FFNet.py           # Fourier Feature Network
│   │   │   ├── SIRENNet.py        # SIREN implementation
│   │   │   ├── OC_FFNet.py        # OC-Fourier Feature Network
│   │   │   └── OC_SIRENNet.py     # OC-SIREN implementation
│   │   └── modelmodule.py  # Lightning training modules
│   ├── utils/              # Utility functions and metrics
│   ├── train.py           # Training script
│   └── eval.py            # Evaluation script
├── data/                   # Dataset and results
│   ├── turbulence_1024.npy       # Training data
│   ├── test_preds_FFNet.npy      # Model predictions
│   ├── test_preds_OCFFNet.npy    # OC model predictions  
│   └── test.py                   # Visualization script
├── scripts/
│   └── run.sh             # Hyperparameter sweep scripts
└── logs/                  # Training logs and checkpoints
```

## 🔬 Experiments and Results

### Model Comparison

The repository includes tools for systematic model comparison:

1. **Prediction Quality**: PSNR and MAE metrics on test data
2. **Training Dynamics**: Loss curves and convergence analysis  
3. **Visual Comparison**: Side-by-side prediction visualizations

### Hyperparameter Studies

Example sweep configurations in `scripts/run.sh`:

- **OC-SIREN frequency sweep**: ω₀ ∈ [1, 15] with different architectures
- **Architecture comparison**: MLP vs. residual blocks
- **Integration steps**: Effect of ODE discretization on performance

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- **SIREN**: [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661)
- **Fourier Features**: [Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains](https://arxiv.org/abs/2006.10739)
- **Optimal Transport**: Neural ODE and optimal transport theory applications

## 🙏 Acknowledgments

Built with:
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/) - Training framework
- [Hydra](https://hydra.cc/) - Configuration management  
- [Weights & Biases](https://wandb.ai/) - Experiment tracking
- [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) - Project template
