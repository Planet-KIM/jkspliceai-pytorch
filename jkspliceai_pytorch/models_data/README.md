
# Models Data Directory

This directory is the storage location for pre-trained model weights.

## Structure

It is recommended to organize weights by model type or version.

- **`pytorch_10k/`**: Standard 10k model weights (e.g., `10k_1_retry.pth`).
- **`pytorch/`**: Weights for drop-out models (e.g., `10k_1_drop_retry.pth`).

*Note: Large weight files (`.h5`, `.pth`) are typically ignored by git (via `.gitignore`) to prevent repository bloating.*
