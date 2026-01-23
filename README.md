
# JKSPLICEAI-PYTORCH

## Description

`jkspliceai-pytorch` is a PyTorch implementation of the SpliceAI module, separated from the original `jkspliceai` package to resolve dependency conflicts with TensorFlow. This module is designed to run SpliceAI models using PyTorch, supporting both CPU and GPU (CUDA/MPS) environments.

## Features

- **PyTorch Native**: Optimized for PyTorch environments.
- **GPU Support**: Supports NVIDIA CUDA and Apple Metal Performance Shaders (MPS).
- **Simplified API**: Easy-to-use wrapper `spliceAI` for running predictions.

## Installation

### Prerequisites

- Python >= 3.8
- PyTorch
- pandas
- numpy
- h5py

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

## Usage

### Basic Usage

```python
from jkspliceai_pytorch.spliceAI import spliceAI
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

To verify the installation and device configuration, run the provided test script:

```bash
python script/test_spliceai_pytorch.py
```

## Structure

- `jkspliceai_pytorch/`: Main package source code.
- `script/`: Helper and test scripts.
- `setup.py`: Package installation script.

## License

[License Information]
