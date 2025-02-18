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


class HiddenStateDiffusionModel(ModelMixin, ConfigMixin):
    """
    一个示例性的“条件扩散”模型：
      - `sample` 表示加噪后的 gt (形状 [B, hidden_dim])
      - `draft` 表示条件 (形状同 [B, hidden_dim])
      - `timestep` 表示时间步 (整数)
    """
    @register_to_config
    def __init__(
        self,
        hidden_dim: int = 128,      # sample 和 draft 的原始维度
        time_embed_dim: int = 64,   # 时间步嵌入维度
    ):
        super().__init__()

        # 时间步嵌入（简单版）
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )


        input_dim = hidden_dim + time_embed_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(
        self,
        sample: torch.Tensor,          # 加噪后的 gt, shape [B, hidden_dim]
        timestep: torch.Tensor,        # 时间步, shape [B]
        draft: torch.Tensor = None,    # 作为条件, shape [B, hidden_dim]
        return_dict: bool = True,
        **kwargs
    ):
        # 时间步嵌入
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        t_emb = self.time_embedding(timestep.reshape(-1, 1).float())  # [B, time_embed_dim]

        if draft is None:
            draft = torch.zeros_like(sample)

        # 在残差预测中，我们可以先计算出网络的输入差异
        # 例如，这里我们直接计算 sample 与 draft 的差值
        diff = sample - draft  # [B, hidden_dim]

        # 将 diff 与时间步嵌入拼接
        x = torch.cat([diff, t_emb], dim=-1)  # 形状 [B, hidden_dim + time_embed_dim]

        # 主体网络预测残差（也就是 gt 与 draft 的差）
        residual = self.model(x)  # 输出 shape 为 [B, hidden_dim]

        # 最后输出 gt 的预测：draft + residual
        pred = draft + residual

        if not return_dict:
            return (pred,)
        
        return HiddenStateDiffusionOutput(sample=pred)
