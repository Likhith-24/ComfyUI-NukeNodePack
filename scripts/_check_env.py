import sys
import torch
print("python:", sys.executable)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
