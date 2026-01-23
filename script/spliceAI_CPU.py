import os
import pathlib

from jkspliceai_pytorch.spliceAI import spliceAI_model
from jklib.genome import locus

loc, ref, alt = locus("chr1:925952-925952"), "G", "A"
datas = spliceAI_model(loc=loc, ref=ref, alt=alt, model='spliceai_torch', use_gpu=True)
print("That result is prediction of spliceAI using gpu")
print(datas)
idatas = spliceAI_model(loc=loc, ref=ref, alt=alt, model='spliceai_torch', use_gpu=False)
print("That result is prediction of spliceAI using cpu")
print(datas)
