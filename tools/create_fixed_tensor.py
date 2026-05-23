import torch
import numpy as np

# 使用NumPy确定性随机生成，保证跨设备跨版本一致
rng = np.random.RandomState(42)
arr = rng.randn(1500, 7) * 0.001
fixed_tensor = torch.tensor(arr, dtype=torch.float32)

# 保存为文件
torch.save(fixed_tensor, '/data/bevperception/data/nuscenes/fixed_tensor.pt')