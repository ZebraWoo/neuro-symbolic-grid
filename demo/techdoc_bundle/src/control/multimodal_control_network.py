"""
多模态嵌入 — MultimodalEmbedding
为 4 个模态创建独立编码器 → 门控加权融合。
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple

from .neuron_models import TemporalLIF


class MultimodalEmbedding(nn.Module):
    """多模态嵌入层 — 统一不同来源数据到共同特征空间。

    Args:
        modalities: {modality_name: input_dim}
        hidden_dim: 编码器隐藏维度
        embedding_dim: 输出嵌入维度
    """

    def __init__(
        self,
        modalities: Dict[str, int],
        hidden_dim: int = 64,
        embedding_dim: int = 32,
    ):
        super().__init__()
        self.modalities = modalities
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.encoders = nn.ModuleDict()
        for name, dim in modalities.items():
            self.encoders[name] = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embedding_dim),
            )

        n_mod = len(modalities)
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * n_mod, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_mod),
            nn.Softmax(dim=1),
        )

    def forward_temporal(
        self, modalities_data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量时间维前向 (避免逐时刻 Python 循环).

        Args:
            modalities_data: {name: (B, T, d_m)}
        Returns:
            fused: (B, T, embedding_dim)
            weights: (B, num_modalities)
        """
        first = next(iter(modalities_data.values()))
        B, T, _ = first.shape
        embeddings = []

        for name in self.modalities:
            x = modalities_data[name].reshape(B * T, -1)
            emb = self.encoders[name](x).reshape(B, T, self.embedding_dim)
            embeddings.append(emb)

        concat = torch.cat(embeddings, dim=-1)  # (B, T, M*E)
        flat = concat.reshape(B * T, -1)
        weights = self.fusion_gate(flat).reshape(B, T, -1)

        stacked = torch.stack(embeddings, dim=2)  # (B, T, M, E)
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=2)
        return fused, weights[:, 0, :]
