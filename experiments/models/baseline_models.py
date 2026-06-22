"""
Baseline models for comparison experiments.

Implements:
  - LSTM Baseline
  - Transformer Baseline
  - TCN Baseline
  - SNN-LIF Baseline (single-compartment LIF)
  - SNN-Izhikevich Baseline (single-compartment Izhikevich)

All baselines accept the same input format (flattened multimodal tensor)
and output [B, 5] decision logits.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.models.decision_head import DecisionIntentHead


# ---------------------------------------------------------------------------
# Helper: flatten multimodal dict to tensor
# ---------------------------------------------------------------------------

def flatten_modalities(
    modalities: Dict[str, torch.Tensor],
    feature_dims: Dict[str, int],
) -> torch.Tensor:
    """
    Convert modality dict to flat tensor [B, T, total_dim].
    """
    tensors = []
    for name, dim in feature_dims.items():
        if name in modalities:
            tensors.append(modalities[name])
        else:
            B = list(modalities.values())[0].shape[0]
            T = list(modalities.values())[0].shape[1]
            tensors.append(torch.zeros(B, T, dim, device=list(modalities.values())[0].device))
    return torch.cat(tensors, dim=-1)


# ---------------------------------------------------------------------------
# LSTM Baseline
# ---------------------------------------------------------------------------

class LSTMBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_intents: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decision_head = DecisionIntentHead(hidden_dim, hidden_dim, num_intents, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: [B, T, 11] flattened multimodal tensor"""
        lstm_out, _ = self.lstm(x)                    # [B, T, hidden_dim]
        final_repr = lstm_out[:, -1, :]               # [B, hidden_dim]
        logits = self.decision_head(final_repr)        # [B, 5]
        return {"decision_logits": logits, "final_representation": final_repr}


# ---------------------------------------------------------------------------
# Transformer Baseline
# ---------------------------------------------------------------------------

class TransformerBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_blocks: int = 4,
        num_heads: int = 4,
        num_intents: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoding = self._make_pe(1024, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dropout=dropout,
            batch_first=True, dim_feedforward=hidden_dim * 4,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_blocks)
        self.decision_head = DecisionIntentHead(hidden_dim, hidden_dim, num_intents, dropout)

    @staticmethod
    def _make_pe(max_len: int, d_model: int) -> nn.Parameter:
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.input_proj(x) + self.pos_encoding[:, :x.shape[1], :]
        encoded = self.encoder(x, mask=nn.Transformer.generate_square_subsequent_mask(
            x.shape[1], device=x.device
        ))
        final_repr = encoded[:, -1, :]
        logits = self.decision_head(final_repr)
        return {"decision_logits": logits, "final_representation": final_repr}


# ---------------------------------------------------------------------------
# TCN Baseline (Temporal Convolutional Network)
# ---------------------------------------------------------------------------

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation,
                               padding=(kernel_size - 1) * dilation)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x)
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.norm2(self.conv2(out))
        # Trim padding to match residual length
        if out.shape[-1] > residual.shape[-1]:
            out = out[:, :, :residual.shape[-1]]
        return F.relu(out + residual)


class TCNBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_intents: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = [input_dim, 64, 128, hidden_dim]
        self.blocks = nn.ModuleList([
            TCNBlock(channels[i], channels[i + 1], kernel_size=3, dilation=2**i, dropout=dropout)
            for i in range(len(channels) - 1)
        ])
        self.decision_head = DecisionIntentHead(hidden_dim, hidden_dim, num_intents, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.transpose(1, 2)  # [B, C, T]
        for block in self.blocks:
            x = block(x)
        # Remove padding and pool
        x = x[:, :, -1]  # Last timestep
        logits = self.decision_head(x)
        return {"decision_logits": logits, "final_representation": x}


# ---------------------------------------------------------------------------
# SNN-LIF Baseline (Single-compartment LIF)
# ---------------------------------------------------------------------------

class SNNLIFBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_intents: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.lif_tau = nn.Parameter(torch.tensor(2.0))
        self.lif_threshold = 1.0

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(3)
        ])
        self.decision_head = DecisionIntentHead(hidden_dim, hidden_dim, num_intents, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, D = x.shape

        # Spike encoding: rate-code the continuous inputs
        encoded = self.input_proj(x)  # [B, T, hidden_dim]

        # LIF dynamics over time
        mem = torch.zeros(B, encoded.shape[-1], device=x.device)
        spikes = []
        for t in range(T):
            inp = encoded[:, t, :]
            mem = mem * torch.sigmoid(-1.0 / self.lif_tau) + inp
            spike = (mem >= self.lif_threshold).float()
            mem = mem * (1.0 - spike)  # reset
            spikes.append(spike.unsqueeze(1))
        spike_seq = torch.cat(spikes, dim=1)  # [B, T, hidden_dim]

        # Process through blocks
        out = spike_seq
        for block in self.blocks:
            out = block(out) + out  # residual

        final_repr = out.mean(dim=1)  # mean pooling over time
        logits = self.decision_head(final_repr)
        return {"decision_logits": logits, "final_representation": final_repr}


# ---------------------------------------------------------------------------
# SNN-Izhikevich Baseline (Single-compartment Izhikevich)
# ---------------------------------------------------------------------------

class SNNIzhikevichBaseline(nn.Module):
    def __init__(
        self,
        input_dim: int = 11,
        hidden_dim: int = 128,
        num_intents: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Izhikevich RS parameters
        self.register_buffer("a", torch.tensor(0.02))
        self.register_buffer("b", torch.tensor(0.2))
        self.register_buffer("c", torch.tensor(-65.0))
        self.register_buffer("d", torch.tensor(8.0))
        self.v_peak = 30.0

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(3)
        ])
        self.decision_head = DecisionIntentHead(hidden_dim, hidden_dim, num_intents, dropout)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, D = x.shape

        encoded = self.input_proj(x)  # [B, T, hidden_dim]

        V = torch.full((B, self.hidden_dim), self.c.item(), device=x.device)
        u = torch.zeros(B, self.hidden_dim, device=x.device)
        spikes = []

        for t in range(T):
            I_in = encoded[:, t, :]
            dV = 0.04 * V * V + 5.0 * V + 140.0 - u + I_in
            du = self.a * (self.b * V - u)
            V = V + dV
            u = u + du

            spike_mask = (V >= self.v_peak).float()
            V = torch.where(spike_mask.bool(), self.c, V)
            u = torch.where(spike_mask.bool(), u + self.d, u)
            spikes.append(spike_mask.unsqueeze(1))

        spike_seq = torch.cat(spikes, dim=1)  # [B, T, hidden_dim]

        out = spike_seq
        for block in self.blocks:
            out = block(out) + out

        final_repr = out.mean(dim=1)
        logits = self.decision_head(final_repr)
        return {"decision_logits": logits, "final_representation": final_repr}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_MODEL_REGISTRY = {
    "lstm": LSTMBaseline,
    "transformer": TransformerBaseline,
    "tcn": TCNBaseline,
    "snn_lif": SNNLIFBaseline,
    "snn_izh": SNNIzhikevichBaseline,
}


def build_baseline(model_name: str, **kwargs) -> nn.Module:
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown baseline: {model_name}. Options: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[model_name](**kwargs)
