import subprocess
import time
import re
import pickle
from pathlib import Path


# =========================================================
# 사용자 입력
# =========================================================
LOC = "chr1:925952-925952"
REF = "G"
ALT = "A"

JOB_NAME = "spliceai_auto"
BASE_DIR = Path("workdir")
SCRIPT_PY = BASE_DIR / "run_spliceai.py"
SCRIPT_SBATCH = BASE_DIR / "run_spliceai.sbatch"
RESULT_FILE = BASE_DIR / "result.pkl"
LOG_DIR = BASE_DIR / "logs"

BASE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)


# =========================================================
# 1. GPU 노드에서 실행될 Python 스크립트 생성
# =========================================================
SCRIPT_PY.write_text(
f"""import os
import pickle
import torch
from jkspliceai_pytorch.spliceAI import spliceAI_model
from jklib.genome import locus

print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Torch CUDA available:", torch.cuda.is_available())
print("Torch MPS available:", torch.backends.mps.is_available())

if torch.cuda.is_available():
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("Using MPS device")
else:
    print("No GPU detected")

loc = locus("{LOC}")
ref = "{REF}"
alt = "{ALT}"

datas = spliceAI_model(
    loc=loc,
    ref=ref,
    alt=alt,
    model="spliceai_torch",
    use_gpu=True
)

with open("{RESULT_FILE}", "wb") as f:
    pickle.dump(datas, f)

print("Result saved to {RESULT_FILE}")
"""
)

print("✔ run_spliceai.py generated")


# =========================================================
# 2. sbatch 스크립트 생성 (⚠️ 공백/개행 주의)
# =========================================================
SCRIPT_SBATCH.write_text(
f"""#!/bin/bash
#SBATCH --job-name={JOB_NAME}
#SBATCH --partition=jklab-gpu-A6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output={LOG_DIR}/%j.out
#SBATCH --error={LOG_DIR}/%j.err

source /home/dwkim/anaconda3/etc/profile.d/conda.sh
conda activate portal

python {SCRIPT_PY}
"""
)

# 실행 권한 (환경에 따라 필요)
SCRIPT_SBATCH.chmod(0o755)

print("✔ run_spliceai.sbatch generated")


# =========================================================
# 3. sbatch 제출 (stderr 반드시 출력)
# =========================================================
proc = subprocess.run(
    ["sbatch", str(SCRIPT_SBATCH)],
    capture_output=True,
    text=True
)

print("----- sbatch STDOUT -----")
print(proc.stdout.strip())
print("----- sbatch STDERR -----")
print(proc.stderr.strip())

proc.check_returncode()

job_id = re.search(r"Submitted batch job (\d+)", proc.stdout).group(1)
print("✔ Job ID:", job_id)


# =========================================================
# 4. Job 종료 대기
# =========================================================
print("⏳ Waiting for job to finish...")

while True:
    q = subprocess.run(
        ["squeue", "-j", job_id],
        capture_output=True,
        text=True
    )
    if job_id not in q.stdout:
        break
    time.sleep(10)

print("✔ Job finished")


# =========================================================
# 5. 결과 로드
# =========================================================
if not RESULT_FILE.exists():
    raise RuntimeError("Result file not found. Check Slurm logs.")

with open(RESULT_FILE, "rb") as f:
    result = pickle.load(f)

print("\n=== SpliceAI Result ===")
print(result)

