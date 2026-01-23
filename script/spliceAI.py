import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from jkspliceai_pytorch import spliceAI
from jklib.genome import locus
from jklib.utils.basic import rc

import time
import torch

# This script demonstrates manual usage or specific dev usage
# Pre-configured models are now available via model='10k' or '10k_drop'



# Using standard wrapper for demo
loc=locus("chr1:925952-925952")
ref="G"
alt="A"

print(loc, ref, alt)
# Example using 10k_drop model which corresponds to 'spliceai_torch' key in new logic if swapped, 
# or specific keys. Let's use the standard wrapper.
df = spliceAI(loc=loc, ref=ref, alt=alt, max_distance=5000,
        model='10k', view=10, assembly='hg38', verbose=True, todict=False)
print("Standard 10k results:")
print(df.iloc[0] if not df.empty else df)

df_drop = spliceAI(loc=loc, ref=ref, alt=alt, max_distance=5000,
        model='10k_drop', view=10, assembly='hg38', verbose=True, todict=False)
print("10k_drop results:")
print(df_drop.iloc[0] if not df_drop.empty else df_drop)

print(df.iloc[0])
print(time.time()-current_time)
print(df)
#df= spliceAI_model(loc=loc, ref=ref, alt=alt, max_distance=1000,
#        model='ensemble', view=5, assembly='hg38', verbose=True, todict=False)
#print(df)
