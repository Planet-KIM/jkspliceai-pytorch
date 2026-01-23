import torch

# GPU 사용 가능 여부 체크
gpu_available = torch.cuda.is_available()
print("GPU 사용 가능 여부 for cuda:", gpu_available)

# GPU 사용 가능 여부 체크 for mac device
gpu_available_mac =  torch.backends.mps.is_available()
print("GPU 사용 가능 여부 for mac:", gpu_available_mac)

# 사용 가능한 GPU의 개수 확인
gpu_count = torch.cuda.device_count()
print("사용 가능한 GPU 개수:", gpu_count)

# 만약 GPU가 사용 가능하다면, 첫번째 GPU의 이름 출력
if gpu_available:
    print("첫번째 GPU 이름:", torch.cuda.get_device_name(0))
