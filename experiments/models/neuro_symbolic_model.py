"""
Neuro-Symbolic Decision Model for Power Grid Decision Support.

Architecture:

  PSML Multimodal Input (load, renewable, irradiance, weather)
       │
       ▼
  Multi-modal Spike Encoding (rate + temporal coding)
       │
       ▼
  Multi-compartment SNN:
    ├─ Dendrite 0: load + renewable modalities → LIF encoding
    ├─ Dendrite 1: irradiance + weather modalities → LIF encoding
    └─ Soma: Izhikevich dynamics integrating dendritic inputs
       │
       ▼
  SpikeFormer Blocks × N (self-attention + LIF gating)
       │
       ▼
  Final Representation [B, hidden_dim]
       │
       ├──► Decision Intent Head → [B, 5] logits
       └──► Confidence Head → [B, 1]

The symbolic rule layer and physics constraints are applied as losses
during training (see experiments/losses/), NOT inside the model forward().
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Surrogate Gradient for Spike Function
# ---------------------------------------------------------------------------

class SurrogateSpikeFunction(torch.autograd.Function):
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
        # Triangular surrogate: max(0, 1 - |x - th|)
        grad_input = grad_output * torch.clamp(1.0 - torch.abs(x - th), 0.0)
        return grad_input, None


def spike_fn(x: torch.Tensor, threshold: float = 1.0) -> torch.Tensor:
    return SurrogateSpikeFunction.apply(x, threshold)


# ---------------------------------------------------------------------------
# Multi-compartment Neuron Components
# ---------------------------------------------------------------------------

class DendriteCompartment(nn.Module):
    """
    A single dendritic compartment receiving synaptic input from one modality group.

    Dynamics:
      C_j * dV_dend,j/dt = g_L * (E_L - V_dend,j) + I_syn,j + g_c * (V_soma - V_dend,j)
    """

    def __init__(self, input_dim: int, hidden_dim: int, tau: float = 2.0):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.tau = tau
        self.hidden_dim = hidden_dim

        # Learnable parameters
        self.g_L = nn.Parameter(torch.tensor(0.1))      # leak conductance
        self.E_L = nn.Parameter(torch.tensor(-70.0))     # resting potential (mV)
        self.g_c = nn.Parameter(torch.tensor(0.5))       # coupling to soma

        # Membrane state (reset each forward)
        self.register_buffer("V_dend", torch.zeros(1, hidden_dim), persistent=False)
        self.register_buffer("V_soma_cache", torch.zeros(1, hidden_dim), persistent=False)

    def forward(self, x: torch.Tensor, V_soma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, T, input_dim] input features
            V_soma: [B, hidden_dim] current soma membrane potential
        Returns:
            I_coupling: [B, T, hidden_dim] coupling current to soma
            V_dend: [B, hidden_dim] final dendritic potential
        """
        B, T, _ = x.shape
        device = x.device

        # Reset membrane
        V_dend = torch.zeros(B, self.hidden_dim, device=device)

        I_coupling_list = []

        for t in range(T):
            x_t = x[:, t, :]                         # [B, input_dim]
            I_syn = F.relu(self.input_proj(x_t))     # [B, hidden_dim]

            # Dendrite dynamics (Euler integration)
            I_coupling = self.g_c * (V_soma - V_dend)  # [B, hidden_dim]
            dV = (self.g_L * (self.E_L - V_dend) + I_syn + I_coupling) / self.tau
            V_dend = V_dend + dV

            I_coupling_list.append(I_coupling.unsqueeze(1))

        I_coupling_seq = torch.cat(I_coupling_list, dim=1)  # [B, T, hidden_dim]
        return I_coupling_seq, V_dend


class IzhikevichSoma(nn.Module):
    """
    Soma compartment with Izhikevich dynamics.

    dV_s/dt = 0.04*V_s² + 5*V_s + 140 - u + I_total
    du/dt   = a * (b*V_s - u)

    When V_s >= V_peak (30 mV): spike, reset V_s ← c, u ← u + d
    """

    def __init__(
        self,
        hidden_dim: int,
        neuron_type: str = "RS",
        spike_threshold: float = 30.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.spike_threshold = spike_threshold

        # Izhikevich parameters per neuron type
        neuron_params = {
            "RS":  (0.02, 0.2,  -65.0, 8.0),    # Regular Spiking
            "FS":  (0.1,  0.2,  -65.0, 2.0),    # Fast Spiking
            "LTS": (0.02, 0.25, -65.0, 2.0),    # Low-Threshold Spiking
            "IB":  (0.02, 0.2,  -55.0, 4.0),    # Intrinsically Bursting
        }
        a, b, c, d = neuron_params.get(neuron_type, neuron_params["RS"])
        self.register_buffer("a", torch.tensor(a))
        self.register_buffer("b", torch.tensor(b))
        self.register_buffer("c", torch.tensor(c))
        self.register_buffer("d", torch.tensor(d))

        # Learnable projection of dendritic coupling current
        self.coupling_proj = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        dendritic_currents: torch.Tensor,  # [B, T, hidden_dim] summed coupling from all dendrites
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            spike_output: [B, T, hidden_dim] spike-encoded output sequence
            V_soma: [B, hidden_dim] final soma potential
            spike_rate: [B] average firing rate
        """
        B, T, H = dendritic_currents.shape
        device = dendritic_currents.device

        V_s = torch.full((B, H), self.c.item(), device=device)  # init at reset potential
        u = torch.zeros(B, H, device=device)  # recovery variable

        spikes_list = []
        total_spikes = torch.zeros(B, device=device)

        for t in range(T):
            I_total = self.coupling_proj(dendritic_currents[:, t, :])

            # Izhikevich dynamics (Euler step, dt=1)
            dV = 0.04 * V_s * V_s + 5.0 * V_s + 140.0 - u + I_total
            du = self.a * (self.b * V_s - u)

            V_s = V_s + dV
            u = u + du

            # Spike detection
            spike_mask = (V_s >= self.spike_threshold).float()
            total_spikes += spike_mask.sum(dim=1)

            # Reset spiked neurons
            V_s = torch.where(spike_mask.bool(), self.c, V_s)
            u = torch.where(spike_mask.bool(), u + self.d, u)

            # Output: spike-gated [B, H]
            spike_out = spike_mask * self.output_proj(V_s)
            spikes_list.append(spike_out)

        spike_output = torch.stack(spikes_list, dim=1)  # [B, T, H]
        spike_rate = total_spikes / (T * H)              # [B]

        return spike_output, V_s, spike_rate


# ---------------------------------------------------------------------------
# SpikeFormer Block (adapted from existing code)
# ---------------------------------------------------------------------------

class SpikeFormerBlock(nn.Module):
    """
    Self-attention + FFN + optional LIF spike gating.
    Adapted from src/control/multimodal_control_network.py.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, use_lif: bool = True):
        super().__init__()
        self.use_lif = use_lif

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

        if use_lif:
            self.lif_weight = nn.Parameter(torch.tensor(0.3))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: [B, T, D]
        Returns:
            output: [B, T, D]
            spike_rate: [B] or None
        """
        # Self-attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.dropout(attn_out)

        # FFN
        residual = x
        x = residual + self.dropout(self.ffn(self.norm2(x)))

        # LIF gating
        spike_rate = None
        if self.use_lif:
            spikes = spike_fn(x)
            spike_rate = spikes.mean(dim=(1, 2))  # [B]
            x = x + self.lif_weight * spikes

        return x, spike_rate


# ---------------------------------------------------------------------------
# Neuro-Symbolic Decision Model
# ---------------------------------------------------------------------------

class NeuroSymbolicDecisionModel(nn.Module):
    """
    Full model: Multi-modal encoding → Multi-compartment SNN →
    SpikeFormer blocks → Decision head.

    The symbolic and physics layers are applied as loss functions externally.
    """

    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 64,
        embedding_dim: int = 32,
        num_blocks: int = 4,
        num_heads: int = 4,
        num_decision_intents: int = 5,
        seq_len: int = 96,
        dropout: float = 0.1,
        use_lif: bool = True,
        use_multi_comp: bool = True,
        soma_neuron_type: str = "RS",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_decision_intents = num_decision_intents
        self.seq_len = seq_len
        self.use_multi_comp = use_multi_comp
        self.use_multi_comp_lif = False  # set externally for multi-comp LIF variant
        self.use_spike = True             # set False for dense mode
        self.modality_dims = modality_dims

        # ---- Multi-modal Embedding ----
        self.modality_encoders = nn.ModuleDict()
        total_input_dim = 0
        for name, dim in modality_dims.items():
            self.modality_encoders[name] = nn.Sequential(
                nn.Linear(dim, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.ReLU(),
                nn.Linear(embedding_dim, embedding_dim),
            )
            total_input_dim += dim

        # Fusion gate: learn modality importance
        self.fusion_gate = nn.Sequential(
            nn.Linear(embedding_dim * len(modality_dims), len(modality_dims)),
            nn.Softmax(dim=-1),
        )

        # ---- Multi-compartment (Izhikevich, kept for ablation) ----
        self.dendrite_0 = DendriteCompartment(embedding_dim * 2, hidden_dim)
        self.dendrite_1 = DendriteCompartment(embedding_dim * 2, hidden_dim)
        self.soma = IzhikevichSoma(hidden_dim, neuron_type=soma_neuron_type)

        # ---- Multi-compartment LIF (stable, recommended) ----
        from experiments.models.multi_comp_lif import LeakyDendrite, LIFSoma
        self.dendrite_0_lif = LeakyDendrite(embedding_dim * 2, hidden_dim, tau=0.9)
        self.dendrite_1_lif = LeakyDendrite(embedding_dim * 2, hidden_dim, tau=0.9)
        self.soma_lif = LIFSoma(embedding_dim * 2, hidden_dim, num_dendrites=2, tau=0.9)

        # ---- Single-compartment fallback ----
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)

        # ---- Positional Encoding ----
        self.register_buffer(
            "pos_encoding",
            self._make_positional_encoding(seq_len, hidden_dim),
            persistent=False,
        )

        # ---- SpikeFormer Blocks ----
        self.blocks = nn.ModuleList([
            SpikeFormerBlock(hidden_dim, num_heads, dropout, use_lif=use_lif)
            for _ in range(num_blocks)
        ])

        # ---- Decision Head ----
        from experiments.models.decision_head import DecisionIntentHead, ConfidenceHead
        self.decision_head = DecisionIntentHead(hidden_dim, hidden_dim, num_decision_intents, dropout)
        self.confidence_head = ConfidenceHead(hidden_dim, hidden_dim)

    @staticmethod
    def _make_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]

    def _multi_compartment_forward(
        self, fused_embeddings: torch.Tensor, modality_embeddings: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-compartment SNN forward pass.

        Args:
            fused_embeddings: [B, T, embedding_dim] modality-fused embeddings
            modality_embeddings: dict of [B, T, embedding_dim] per modality
        Returns:
            spike_output: [B, T, hidden_dim]
            spike_rates: averaged firing rate across compartments
        """
        B, T, _ = fused_embeddings.shape
        device = fused_embeddings.device

        # Group modalities for dendrites
        # Dendrite 0: load + renewable
        concat_0 = torch.cat([
            modality_embeddings.get("load", torch.zeros(B, T, self.embedding_dim, device=device)),
            modality_embeddings.get("renewable", torch.zeros(B, T, self.embedding_dim, device=device)),
        ], dim=-1)

        # Dendrite 1: irradiance + weather
        concat_1 = torch.cat([
            modality_embeddings.get("irradiance", torch.zeros(B, T, self.embedding_dim, device=device)),
            modality_embeddings.get("weather", torch.zeros(B, T, self.embedding_dim, device=device)),
        ], dim=-1)

        # Initialize soma potential (at reset value)
        V_soma = torch.full((B, self.hidden_dim), self.soma.c.item(), device=device)

        # Forward through dendrites (coupling current to soma)
        I_d0, _ = self.dendrite_0(concat_0, V_soma)
        I_d1, _ = self.dendrite_1(concat_1, V_soma)

        # Sum dendritic coupling currents
        I_total = (I_d0 + I_d1) / 2.0  # [B, T, hidden_dim]

        # Soma Izhikevich dynamics
        spike_output, V_soma, spike_rate = self.soma(I_total)

        return spike_output, spike_rate

    def _single_compartment_forward(
        self, fused_embeddings: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-compartment LIF with proper membrane dynamics."""
        B, T, _ = fused_embeddings.shape
        device = fused_embeddings.device
        x = self.input_proj(fused_embeddings)  # [B, T, hidden_dim]
        H = x.shape[-1]

        # Proper LIF: leaky integration → fire → reset over time
        v = torch.zeros(B, H, device=device)
        tau = torch.sigmoid(torch.tensor(-0.5, device=device))  # decay factor
        spikes = []
        total_spikes = torch.zeros(B, device=device)

        for t in range(T):
            I_t = x[:, t, :]
            # Leaky integration
            v = tau * v + (1 - tau) * I_t
            # Fire
            s = spike_fn(v)
            total_spikes += s.sum(dim=1)
            # Hard reset
            v = v * (1 - s)
            spikes.append(s.unsqueeze(1))

        spike_output = torch.cat(spikes, dim=1)  # [B, T, H]
        spike_rate = total_spikes / (T * H)       # [B]
        return spike_output, spike_rate

    def forward(
        self, modalities_data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            modalities_data: Dict mapping modality name to tensor [B, seq_len, dim]
                             e.g. {"load": [B, 96, 1], "renewable": [B, 96, 2], ...}
        Returns:
            Dictionary with:
              - decision_logits: [B, 5] raw logits (use with BCEWithLogitsLoss)
              - confidence: [B, 1]
              - final_representation: [B, hidden_dim] for analysis/closed-loop
              - modality_weights: [B, num_modalities]
              - spike_rates: [B] average firing rate
        """
        B = list(modalities_data.values())[0].shape[0]
        seq_len = list(modalities_data.values())[0].shape[1]
        device = list(modalities_data.values())[0].device

        # Step 1: Encode each modality
        encoded = {}
        for name, encoder in self.modality_encoders.items():
            if name in modalities_data:
                x = modalities_data[name]  # [B, T, dim]
                encoded[name] = encoder(x)  # [B, T, embedding_dim]
            else:
                encoded[name] = torch.zeros(B, seq_len, self.embedding_dim, device=device)

        # Step 2: Modality fusion with learned gate
        stacked = torch.cat(list(encoded.values()), dim=-1)  # [B, T, E*M]
        gate_input = stacked.mean(dim=1)                      # [B, E*M]
        modality_weights = self.fusion_gate(gate_input)       # [B, M]

        fused = sum(
            w[:, None, None] * encoded[name]
            for name, w in zip(self.modality_dims.keys(), modality_weights.unbind(-1))
        )  # [B, T, E]

        # Step 3: SNN forward (multi-comp LIF / multi-comp Izhikevich / single-comp / dense)
        all_spike_rates = []
        if not self.use_spike:
            # Dense mode: skip spike encoding, feed continuous features directly
            snn_output = self.input_proj(fused)  # [B, T, hidden_dim]
            spike_rate = torch.zeros(B, device=device)
            all_spike_rates.append(spike_rate)
        elif self.use_multi_comp_lif:
            # Multi-compartment LIF (stable) — dendrites leak-only, soma LIF+spike
            from experiments.models.multi_comp_lif import multi_comp_lif_forward
            snn_output, spike_rate = multi_comp_lif_forward(
                fused, encoded, self.embedding_dim, self.hidden_dim,
                self.dendrite_0_lif, self.dendrite_1_lif, self.soma_lif,
            )
            all_spike_rates.append(spike_rate)
        elif self.use_multi_comp:
            # Multi-compartment Izhikevich (legacy, unstable)
            snn_output, spike_rate = self._multi_compartment_forward(fused, encoded)
            all_spike_rates.append(spike_rate)
        else:
            # Single-compartment LIF (fast, stable)
            snn_output, spike_rate = self._single_compartment_forward(fused)
            all_spike_rates.append(spike_rate)

        # Step 4: Positional encoding + SpikeFormer blocks
        if snn_output.shape[-1] != self.pos_encoding.shape[-1]:
            raise RuntimeError(
                f"Dimension mismatch: snn_output {snn_output.shape} vs "
                f"pos_encoding {self.pos_encoding.shape}. "
                f"snn_output dim={snn_output.shape[-1]}, pos dim={self.pos_encoding.shape[-1]}. "
                f"hidden_dim={self.hidden_dim}, embedding_dim={self.embedding_dim}"
            )
        if snn_output.dim() == 4:
            # Squeeze any spurious dimension (e.g., from old soma bug)
            snn_output = snn_output.squeeze(2)
        x = snn_output + self.pos_encoding[:, :seq_len, :].to(device)

        for block in self.blocks:
            x, sr = block(x)
            if sr is not None:
                all_spike_rates.append(sr)

        # Step 5: Pool to single representation
        # Use last timestep + mean pooling
        final_repr = x[:, -1, :] + 0.3 * x.mean(dim=1)  # [B, hidden_dim]

        # Step 6: Decision head
        decision_logits = self.decision_head(final_repr)       # [B, 5]
        confidence = self.confidence_head(final_repr)          # [B, 1]

        # Aggregate spike rates
        avg_spike_rate = torch.stack(all_spike_rates).mean(dim=0) if all_spike_rates else torch.zeros(B, device=device)

        return {
            "decision_logits": decision_logits,
            "confidence": confidence,
            "final_representation": final_repr,
            "modality_weights": modality_weights,
            "spike_rate": avg_spike_rate,
        }
