#from jkspliceai_pytorch.spliceAI import spliceAI_dataportal
import sys
sys.path.append('/home/jkportal/portal/renewal/jkportal/jkrun/modules/third_party')
from jktpm import spliceAI2portal

variantStr='chr10:71791123-71791123 G>A'   

df = spliceAI2portal(variantStr=variantStr, strand='+', distance=5000, view=5, verbose=True)#, post_value=False)
print(df)
