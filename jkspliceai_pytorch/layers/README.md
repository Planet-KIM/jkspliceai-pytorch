
# Layers Directory

This directory contains the fundamental building blocks used to construct SpliceAI models.

## Files

- **`blocks.py`**: 
    - `ResidualBlock`, `ResidualBlock2`: Standard SpliceAI residual blocks (with optional Dropout).
    - `SEBlock`: Squeeze-and-Excitation block for channel attention.
    - `SelfAttnBlock`: Transformer-based self-attention block.
    - `MultiScaleBlock`: Module for parallel multi-dilation convolutions.
