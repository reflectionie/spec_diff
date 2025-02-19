import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from dataclasses import dataclass
from diffusers.utils import BaseOutput

@dataclass
class HiddenStateDiffusionOutput(BaseOutput):
    """
    自定义隐藏状态扩散模型的输出。
    Args:
        sample (`torch.Tensor` of shape `(batch_size, hidden_dim)`):
            最后输出的预测结果，例如预测的噪声、或去噪后的隐藏向量等。
    """
    sample: torch.Tensor = None


class ResidualMLPBlock(nn.Module):
    """
    一个简单的 MLP 残差块示例：
    1) LayerNorm -> Linear -> 激活 -> LayerNorm -> Linear
    2) 与输入相加做残差
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.norm2 = nn.LayerNorm(dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.norm2(x)
        x = self.linear2(x)
        return x + residual


class HiddenStateDiffusionModel(ModelMixin, ConfigMixin):
    """
    一个改进版本的“条件扩散”模型示例：
      - 在原有的两层 MLP 基础上，增加了多层残差 MLP 块。
      - 仍然使用 (sample - draft) + time_embed 的思路进行输入。
    """

    @register_to_config
    def __init__(
        self,
        hidden_dim: int = 128,        # sample 和 draft 的原始维度
        time_embed_dim: int = 64,     # 时间步嵌入维度
        num_layers: int = 4,         # 残差层数，可根据需要增减
    ):
        super().__init__()

        # 时间步嵌入（与原逻辑相同）
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 先把 (diff + time_embed) 投影到 hidden_dim
        # 输入维度 = hidden_dim + time_embed_dim
        input_dim = hidden_dim + time_embed_dim
        self.input_layer = nn.Linear(input_dim, hidden_dim)

        # 多层残差 MLP 块
        self.res_blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim) for _ in range(num_layers)
        ])

        # 输出层，将 hidden_dim 投影回到原始维度
        self.output_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        sample: torch.Tensor,          # 加噪后的 gt, shape [B, hidden_dim]
        timestep: torch.Tensor,        # 时间步, shape [B]
        draft: torch.Tensor = None,    # 作为条件, shape [B, hidden_dim]
        return_dict: bool = True,
        **kwargs
    ):
        # 1. 时间步嵌入
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        t_emb = self.time_embedding(timestep.reshape(-1, 1).float())  # [B, time_embed_dim]

        # 2. 若无 draft 则填 0
        if draft is None:
            draft = torch.zeros_like(sample)

        # 3. 计算 diff，并与时间步嵌入拼接
        diff = sample - draft  # [B, hidden_dim]
        x = torch.cat([diff, t_emb], dim=-1)  # 形状 [B, hidden_dim + time_embed_dim]

        # 4. 投影到 hidden_dim
        x = self.input_layer(x)

        # 5. 依次经过多个残差块
        for block in self.res_blocks:
            x = block(x)

        # 6. 输出层
        residual = self.output_layer(x)

        # 7. 最终预测结果：draft + residual
        pred = draft + residual

        if not return_dict:
            return (pred,)

        return HiddenStateDiffusionOutput(sample=pred)
