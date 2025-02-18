import torch
from my_hidden_state_diffusion import HiddenStateDiffusionModel

# 初始化一个模型
model = HiddenStateDiffusionModel(hidden_dim=256, time_embed_dim=128)
model.train()

# 模拟一个 batch
batch_size = 4
hidden_dim = 256
sample = torch.randn(batch_size, hidden_dim)    # [4, 256]
timestep = torch.randint(0, 1000, size=(batch_size,))  # 随机时间步

# 前向
output = model(sample, timestep)
print(output.sample.shape)  # [4, 256]
