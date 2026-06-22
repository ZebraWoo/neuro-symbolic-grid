"""
Multi-compartment LIF for GORS.

Key difference from old multi-comp Izhikevich:
  - Dendrites: leaky integration ONLY (no spike, no reset) — temporal smoothing
  - Soma: full LIF with spike — receives direct grid input + dendritic coupling currents
  - No Izhikevich dynamics → stable training, fast convergence

This preserves the "multi-compartment spatial decoupling" narrative
while being as trainable as a flat LIF.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward + triangular surrogate gradient backward."""

    @staticmethod
    def forward(ctx, x, threshold=1.0):
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        th = ctx.threshold
        grad_input = grad_output * torch.clamp(1.0 - torch.abs(x - th), 0.0)
        return grad_input, None


def lif_spike(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return SurrogateSpike.apply(x, threshold)


class LeakyDendrite(nn.Module):
    """
    Dendrite compartment — leaky integration ONLY, no spike.

    Dynamics:
      v_d(t) = tau * v_d(t-1) + I_input(t)

    Purpose: smooth meteorological noise before coupling to soma.
    Acts as a modality-specific temporal filter.

    Args:
        input_dim: dimension of input features to this dendrite
        output_dim: dimension of coupling current sent to soma
        tau: decay constant (0 < tau < 1, higher = slower decay = smoother)
    """

    def __init__(self, input_dim: int, output_dim: int, tau: float = 0.9):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, output_dim)
        self.register_buffer("tau", torch.tensor(tau))
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] modality features
        Returns:
            coupling: [B, T, output_dim] smoothed coupling current to soma
        """
        B, T, _ = x.shape
        device = x.device

        v = torch.zeros(B, self.output_dim, device=device)
        outputs = []

        for t in range(T):
            I_in = self.input_proj(x[:, t, :])  # [B, D]
            v = self.tau * v + (1 - self.tau) * I_in  # leaky integration
            outputs.append(v.unsqueeze(1))

        return torch.cat(outputs, dim=1)  # [B, T, D]


class LIFSoma(nn.Module):
    """
    Soma compartment — full LIF with spike, receiving:
      1. Direct grid electrical input
      2. Coupling currents from dendrites

    Dynamics:
      v_s(t) = tau_s * v_s(t-1) * (1 - s(t-1)) + I_direct(t) + Σ g_c * v_d(t)
      s(t) = 1 if v_s(t) >= V_th else 0
      After spike: v_s ← 0 (hard reset)
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_dendrites: int = 2,
                 tau: float = 0.9, v_threshold: float = 1.0):
        super().__init__()
        self.direct_proj = nn.Linear(input_dim, hidden_dim)
        self.coupling_fusion = nn.Linear(hidden_dim * num_dendrites, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.register_buffer("tau", torch.tensor(tau))
        self.v_threshold = v_threshold
        self.hidden_dim = hidden_dim
        self.num_dendrites = num_dendrites

    def forward(self, x_direct: torch.Tensor,
                dendritic_currents: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_direct: [B, T, input_dim] direct grid input (load + renewable)
            dendritic_currents: list of [B, T, hidden_dim] from each dendrite
        Returns:
            spike_seq: [B, T, hidden_dim] spike-encoded output
            spike_rate: [B] average firing rate
        """
        B, T, _ = x_direct.shape
        device = x_direct.device

        # Project direct input
        I_direct = self.direct_proj(x_direct)  # [B, T, H]

        # Fuse dendritic currents
        dend_cat = torch.cat(dendritic_currents, dim=-1)  # [B, T, H*N]
        I_dend = self.coupling_fusion(dend_cat)            # [B, T, H]

        # Total input
        I_total = I_direct + I_dend  # [B, T, H]

        # LIF dynamics over time
        v = torch.zeros(B, self.hidden_dim, device=device)
        spikes = []
        total_spikes = torch.zeros(B, device=device)

        for t in range(T):
            I_t = I_total[:, t, :]
            # Membrane update with reset
            v = self.tau * v + (1 - self.tau) * I_t

            # Spike detection + surrogate gradient
            s = lif_spike(v, self.v_threshold)
            total_spikes += s.sum(dim=1)

            # Hard reset
            v = v * (1 - s)

            # Output: spike-gated
            out = s * self.output_proj(v)
            spikes.append(out.unsqueeze(1))

        spike_seq = torch.cat(spikes, dim=1)  # [B, T, H]
        spike_rate = total_spikes / (T * self.hidden_dim)

        return spike_seq, spike_rate


# ---------------------------------------------------------------------------
# Helper: build multi-comp LIF from fused embeddings
# ---------------------------------------------------------------------------

def multi_comp_lif_forward(
    fused: torch.Tensor,                      # [B, T, E] modality-fused embedding
    encoded: dict[str, torch.Tensor],          # per-modality embeddings
    embedding_dim: int,
    hidden_dim: int,
    dendrite_0: LeakyDendrite,
    dendrite_1: LeakyDendrite,
    soma: LIFSoma,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Multi-compartment LIF forward pass.

    Dendrite 0: irradiance + weather (meteorological modalities) → smoothed coupling
    Dendrite 1: load + renewable (electrical modalities) → smoothed coupling
    Soma: direct load+renewable + dendritic coupling currents → LIF spikes
    """
    B, T, _ = fused.shape
    device = fused.device

    # Build dendrite inputs
    concat_d0 = torch.cat([
        encoded.get("irradiance", torch.zeros(B, T, embedding_dim, device=device)),
        encoded.get("weather", torch.zeros(B, T, embedding_dim, device=device)),
    ], dim=-1)  # [B, T, 2*E]

    concat_d1 = torch.cat([
        encoded.get("load", torch.zeros(B, T, embedding_dim, device=device)),
        encoded.get("renewable", torch.zeros(B, T, embedding_dim, device=device)),
    ], dim=-1)  # [B, T, 2*E]

    # Dendritic coupling currents (leak-only, no spike)
    I_d0 = dendrite_0(concat_d0)  # [B, T, H]
    I_d1 = dendrite_1(concat_d1)  # [B, T, H]

    # Soma: direct grid input (load+renewable) + dendritic currents → LIF spikes
    x_direct = torch.cat([
        encoded.get("load", torch.zeros(B, T, embedding_dim, device=device)),
        encoded.get("renewable", torch.zeros(B, T, embedding_dim, device=device)),
    ], dim=-1)  # [B, T, 2*E]

    spike_output, spike_rate = soma(x_direct, [I_d0, I_d1])

    return spike_output, spike_rate
