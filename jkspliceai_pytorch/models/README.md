
# Models Directory

This directory contains the definitions of various SpliceAI model architectures.

## Files

- **`spliceai.py`**: 
    - Standard SpliceAI architectures (`80nt`, `400nt`, `2k`, `10k`).
    - `SpliceAI_10k_drop`: 10k model with Dropout layers.
    - `SpliceAI`: Factory class for instantiating models by name.
- **`spliceai_unet.py`**: 
    - UNet-style implementations (`SpliceAI_10k_UNet`).
    - `SpliceAI_80nt_UNet`: Smaller UNet variant.
- **`spliceai_trans.py`**: 
    - `SpliceAI_10k_Transformer`: 10k model integrated with Self-Attention blocks.
    - `SpliceAI_10k_Transformer_SE`: 10k model with both Self-Attention and SE blocks.
