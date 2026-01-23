
import torch

def get_device():
    """
    Priority:
    1. CUDA (RTX 3090, etc.)
    2. Apple MPS (M1/M2/M3)
    3. CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
