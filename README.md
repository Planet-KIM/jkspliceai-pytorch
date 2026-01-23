
# JKSPLICEAI-PYTORCH

## Description

`jkspliceai-pytorch` is a PyTorch implementation of the SpliceAI module, optimized for efficiency and flexibility. It is separated from the original `jkspliceai` package to resolve dependency conflicts and provide a pure PyTorch environment.

## Features

- **Refactored Structure**: Clean separation of layers, models, utilities, and wrappers.
- **PyTorch Integration**: Full support for PyTorch models including standard SpliceAI and custom variants (UNet, Transformer).
- **Multiple Models**: Supports standard `10k` and `10k_drop` (with dropout) models.
- **GPU/MPS Support**: Automatically detects CUDA (NVIDIA) or MPS (Apple Silicon), falling back to CPU.
- **Easy Inference**: Wrapper function `spliceAI` for easy prediction from genomic loci.

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
   pip install -r requirements.txt
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

# Run prediction with standard model (default)
result = spliceAI(loc=loc, ref=ref, alt=alt, use_gpu=True)
print("Standard Result:", result)

# Run prediction with 10k_drop model
result_drop = spliceAI(loc=loc, ref=ref, alt=alt, model='10k_drop', use_gpu=True)
print("Drop Model Result:", result_drop)
```

### Running Tests

To verify the installation and device configuration:

```bash
python script/test_spliceai_pytorch.py
```

## License

[License Information]
