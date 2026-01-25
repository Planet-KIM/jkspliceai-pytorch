
import sys
import os

# Add parent directory to path to ensure we import the local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jkspliceai_pytorch import spliceAI as spliceAI_torch
from jklib.genome import locus
import torch

def test_10k_drop():
    print(f"\n{'='*20}")
    print(f"Running Case: SpliceAI 10k_drop Model")
    print(f"{'='*20}")
    
    # Example Location
    loc, ref, alt = locus("chr1:925952-925952"), "G", "A"
    
    try:
        # Calling spliceAI wrapper with model='10k_drop'
        datas = spliceAI_torch(
            loc=loc, 
            ref=ref, 
            alt=alt, 
            model='10k_drop',
            view=5,
            verbose=True,
            use_gpu=True
        )
        print("Success! Result head:")
        if hasattr(datas, 'head'):
            print(datas.head())
        else:
            print(datas)
            
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        print("Using MPS")
    else:
        print("Using CPU")
        
    test_10k_drop()
