
# Utils Directory

This directory contains utility functions for data processing, helper methods, and device management.

## Files

- **`data.py`**: 
    - `one_hot_encode_torch`: Converts DNA sequences to one-hot encoded PyTorch tensors.
    - `replace_dna_ref_to_alt2`: Handles the substitution of reference alleles with alternative alleles in sequences.
    - `custom_dataframe`: Helpers for formatting output DataFrames.
- **`helpers.py`**: 
    - `get_device`: Automatically detects and returns the best available PyTorch device (CUDA > MPS > CPU).
