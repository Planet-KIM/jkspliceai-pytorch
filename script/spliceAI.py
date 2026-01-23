from jkspliceai_pytorch.spliceAI import *
from jklib.genome import locus
from jklib.utils.basic import rc

import time

import torch
from spliceai_pytorch import SpliceAI   

from variants import Variants

model_dict = {}
for m in range(5):
    #model_path = f"/home/jkportal/portal/renewal/jkportal/configs/data/models/pytorch/10k_{m+1}_drop_retry.pth"
    model_path = f"/home/jkportal/portal/renewal/jkportal/configs/data/models/pytorch_only_10k/10k_{m+1}_retry.pth"
    #model = SpliceAI.from_preconfigured('10k_drop')
    model = SpliceAI.from_preconfigured('10k')

    # weights_only=True 옵션 사용 시:
    state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()  # 평가 모드

    model_dict[m] = model


current_time= time.time()
#loc=locus('chr10:71791123-71791123+')
#ref='G'
#alt='A'

#loc= locus("chr1:925952-925952")
#ref= "G"
#alt="A"

variantStr = "NM_001371596.2(MFSD8):c.525T>A"
locStr, ref, alt = Variants(variantStr).variant2locus()
loc = locus(f"{locStr}-")
ref, alt = rc(ref), rc(alt)
#loc=locus("1:930130-930130")
#ref="C"
#alt="G"


print(locStr, ref, alt)
df= spliceAI_model(loc=loc, ref=ref, alt=alt, max_distance=5000,
        model=model_dict, view=10, assembly='hg38', verbose=True, todict=False)
#print(df[df.columns.tolist()[8:11]])
print(df.iloc[0])
print(time.time()-current_time)
print(df)
#df= spliceAI_model(loc=loc, ref=ref, alt=alt, max_distance=1000,
#        model='ensemble', view=5, assembly='hg38', verbose=True, todict=False)
#print(df)
