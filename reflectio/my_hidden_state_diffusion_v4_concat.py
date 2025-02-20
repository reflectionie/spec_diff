import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

# 假设已有 HiddenStateDiffusionOutput 定义，如下简单包装：
class HiddenStateDiffusionOutput:
    def __init__(self, sample):
        self.sample = sample

class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.activation = nn.SiLU()

    def forward(self, x):
        return self.activation(x + self.block(x))

class HiddenStateDiffusionModel(ModelMixin, ConfigMixin):
    """
    改进版“条件扩散”模型：
      - sample: 加噪后的 gt, 形状 [B, hidden_dim]
      - draft: 条件信息, 形状 [B, hidden_dim]
      - timestep: 时间步 (整数或形状 [B])
    目标是预测 gt 与 draft 的残差，并最终得到 gt = draft + residual。
    """
    @register_to_config
    def __init__(self, hidden_dim: int = 128, time_embed_dim: int = 64):
        super().__init__()

        # 时间步嵌入（简单实现）
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 直接拼接 sample、draft 和时间嵌入，输入维度为 2*hidden_dim + time_embed_dim
        input_dim = 2 * hidden_dim + time_embed_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim)
        )


    def forward(
        self,
        sample: torch.Tensor,          # 加噪后的 gt, shape [B, hidden_dim]
        timestep: torch.Tensor,        # 时间步, shape [B] 或 标量
        draft: torch.Tensor = None,    # 条件 draft, shape [B, hidden_dim]
        return_dict: bool = True,
        **kwargs
    ):
        # 时间步嵌入
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        t_emb = self.time_embedding(timestep.reshape(-1, 1).float())  # 形状 [B, time_embed_dim]

        # 如果没有提供条件 draft，则默认为全零向量
        if draft is None:
            draft = torch.zeros_like(sample)

        # 将 sample、draft 与时间嵌入直接拼接
        x = torch.cat([sample, draft, t_emb], dim=-1)  # 形状 [B, 2*hidden_dim + time_embed_dim]

        # 网络预测残差 (即 gt 与 draft 的差)
        residual = self.model(x)  # 输出 shape 为 [B, hidden_dim]

        # 最终预测 gt: draft + residual
        pred = draft + residual

        if not return_dict:
            return (pred,)
        return HiddenStateDiffusionOutput(sample=pred)
