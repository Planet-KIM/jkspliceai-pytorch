import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jkspliceai_pytorch import spliceAI
from jklib.genome import locus

loc, ref, alt = locus("chr1:925952-925952"), "G", "A"
datas = spliceAI(loc=loc, ref=ref, alt=alt, model='10k', use_gpu=True)
print("That result is prediction of spliceAI using gpu (10k)")
print(datas)
idatas = spliceAI(loc=loc, ref=ref, alt=alt, model='10k', use_gpu=False)
print("That result is prediction of spliceAI using cpu (10k)")
print(datas)
