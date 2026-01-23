import os
import pickle
import tensorflow as tf
from jkspliceai.spliceAI import spliceAI_model
from jklib.genome import locus

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("TF GPUs:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if not gpus:
    raise RuntimeError("TensorFlow GPU not available")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

loc = locus("chr1:925952-925952")
ref = "G"
alt = "A"

datas = spliceAI_model(
    loc=loc,
    ref=ref,
    alt=alt,
    model="ensemble",
    use_gpu=True
)

with open("workdir/result.pkl", "wb") as f:
    pickle.dump(datas, f)

print("Result saved to workdir/result.pkl")
