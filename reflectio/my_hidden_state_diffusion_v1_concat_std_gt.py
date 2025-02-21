import torch
import torch.nn as nn
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
"""
共处理文件数：68001 (总文件数：68001)
真实 hidden state 的处理元素数：356853231616
真实 hidden state 的均值：0.002686
真实 hidden state 的标准差：2.436462
----------------------------------------
草稿 hidden state 的处理元素数：356853231616
草稿 hidden state 的均值：0.003213
草稿 hidden state 的标准差：2.042219
"""
# 简单包装的输出类
class HiddenStateDiffusionOutput:
    def __init__(self, sample):
        self.sample = sample

class HiddenStateDiffusionModel(ModelMixin, ConfigMixin):
    """
    改进版“条件扩散”模型：
      - sample: 加噪后的 gt, 形状 [B, hidden_dim]
      - draft: 条件信息, 形状 [B, hidden_dim]
      - timestep: 时间步 (整数或形状 [B])
    目标是预测 gt 与 draft 的残差，并最终得到 gt = draft + residual。
    
    新增：
      - data_mean: 数据均值，用于归一化
      - data_std: 数据标准差，用于归一化
      - use_normalization: 是否在 forward 时使用归一化/反归一化操作
    """
    @register_to_config
    def __init__(
        self, 
        hidden_dim: int = 128, 
        time_embed_dim: int = 64, 
        data_mean: float = 0.002686, 
        data_std: float = 2.436462
    ):
        super().__init__()
        self.data_mean = data_mean
        self.data_std = data_std

        # 时间步嵌入（简单实现）
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # 拼接 sample、draft 和时间嵌入，输入维度为 2*hidden_dim + time_embed_dim
        input_dim = 2 * hidden_dim + time_embed_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """将数据归一化到均值0、标准差1的分布"""
        return (x - self.data_mean) / self.data_std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """将归一化后的数据恢复到原始分布"""
        return x * self.data_std + self.data_mean

    def forward(
        self,
        sample: torch.Tensor,          # 加噪后的 gt, shape [B, hidden_dim]
        timestep: torch.Tensor,        # 时间步, shape [B] 或 标量
        draft: torch.Tensor = None,    # 条件 draft, shape [B, hidden_dim]
        use_normalization: bool = True,  # 控制是否使用归一化/反归一化
        return_dict: bool = True,
        **kwargs
    ):
        # 如果启用归一化，则对 sample 和 draft 进行归一化
        if use_normalization:
            sample = self.normalize(sample)
            if draft is not None:
                draft = self.normalize(draft)
            else:
                draft = torch.zeros_like(sample)
        else:
            if draft is None:
                draft = torch.zeros_like(sample)

        # 时间步嵌入
        if len(timestep.shape) == 0:
            timestep = timestep[None].to(sample.device)
        t_emb = self.time_embedding(timestep.reshape(-1, 1).float())  # shape [B, time_embed_dim]

        # 拼接归一化后的 sample、draft 和时间嵌入
        x = torch.cat([sample, draft, t_emb], dim=-1)  # shape [B, 2*hidden_dim + time_embed_dim]

        # 网络预测残差 (即 gt 与 draft 的差)
        residual = self.model(x)  # 输出 shape 为 [B, hidden_dim]

        # 预测 gt: draft + residual
        pred = draft + residual

        # 如果启用归一化，则对预测结果反归一化，恢复到原始尺度
        if use_normalization:
            pred = self.denormalize(pred)

        if not return_dict:
            return (pred,)
        return HiddenStateDiffusionOutput(sample=pred)