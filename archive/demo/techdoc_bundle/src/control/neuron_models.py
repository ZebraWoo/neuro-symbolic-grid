"""
神经元模型 — TemporalLIF (batch×time 并行)
用于 multimodal_control_network 的脉冲 Transformer block。
"""

import torch
import torch.nn as nn


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward + triangular surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        th = ctx.threshold
        grad_x = grad_output * torch.clamp(1.0 - (x - th).abs(), min=0.0)
        return grad_x, None


class TemporalLIF(nn.Module):
    """Batch×time LIF: (B, T, D) -> (B, T, D) spikes."""

    def __init__(self, threshold: float = 1.0, leak: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        mem = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(T):
            mem = self.leak * mem + x[:, t, :]
            spk = SurrogateSpike.apply(mem, self.threshold)
            mem = mem * (1.0 - spk)
            spikes.append(spk.unsqueeze(1))
        return torch.cat(spikes, dim=1)
