
# JKSPLICEAI-PYTORCH

## Description

`jkspliceai-pytorch` is a PyTorch implementation of the SpliceAI module, optimized for efficiency and flexibility. It is separated from the original `jkspliceai` package to resolve dependency conflicts and provide a pure PyTorch environment.

## Features

- **PyTorch Native**: Optimized for PyTorch environments with support for CUDA and MPS (Apple Silicon).
- **Modular Design**: Refactored into clean modular components for layers, models, and utilities.
- **Multiple Architectures**: Supports standard SpliceAI (80nt, 400nt, 2k, 10k), UNet-based, and Transformer-based variants.
- **Easy Inference**: Simple `spliceAI` wrapper for quick predictions.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch
- pandas
- numpy
- h5py
- einops

### Installing from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/Planet-KIM/jkspliceai-pytorch.git
   cd jkspliceai-pytorch
   ```

2. Install dependencies:
   ```bash
   pip install -r config/requirements.txt
   ```
   *Note: Ensure `torch` is installed according to your system configuration (CUDA/CPU).*

3. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

## Directory Structure

- **`jkspliceai_pytorch/`**: Main package source.
  - **`layers/`**: Basic building blocks (Residual blocks, SE blocks, Attention).
  - **`models/`**: Model architecture definitions (Standard, UNet, Transformer).
  - **`utils/`**: Helper functions for data processing and device management.
  - **`wrapper/`**: High-level API for running SpliceAI.
  - **`models_data/`**: Stores model weights (`.pth` files).

## Usage

### Basic Usage

```python
from jkspliceai_pytorch import spliceAI
from jklib.genome import locus

# Define variant location
loc = locus("chr1:925952-925952")
ref = "G"
alt = "A"

# Run prediction
# use_gpu=True will automatically use CUDA or MPS if available
result = spliceAI(
    loc=loc, 
    ref=ref, 
    alt=alt, 
    view=5, 
    use_gpu=True
)

print(result)
```

### Running Tests

To verify the installation and device configuration:

```bash
python script/test_spliceai_pytorch.py
```

## License

[License Information]
