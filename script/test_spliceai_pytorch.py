
import sys
import os

# Add parent directory to path to ensure we import the local package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jkspliceai_pytorch import spliceAI
from jklib.genome import locus
import torch

def run_test(case_name, use_gpu):
    print(f"\n{'='*20}")
    print(f"Running Case: {case_name}")
    print(f"Config: use_gpu={use_gpu}")
    print(f"{'='*20}")
    
    # Example Location
    loc, ref, alt = locus("chr1:925952-925952"), "G", "A"
    
    try:
        # Calling spliceAI wrapper which now supports use_gpu
        # Note: max_distance defaults to 5000 in both wrapper and model usage
        datas = spliceAI(
            loc=loc, 
            ref=ref, 
            alt=alt, 
            view=5,
            use_gpu=use_gpu
        )
        print("Success! Result head:")
        # Attempt to access head if dataframe, else print
        if hasattr(datas, 'head'):
            print(datas.head())
        else:
            print(datas)
            
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print(f"Torch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"MPS available: {torch.backends.mps.is_available()}")

    # Case 1: CPU
    run_test("PyTorch CPU", use_gpu=False)

    # Case 2: GPU (Will use MPS on Mac if available and use_gpu=True)
    run_test("PyTorch GPU", use_gpu=True)
