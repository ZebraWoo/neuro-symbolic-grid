"""
Decision Intent Head.

Maps the SNN's final representation to 5 binary decision intents.
Output is passed through sigmoid → each intent is independent (multi-label).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DecisionIntentHead(nn.Module):
    """
    MLP head that maps [B, hidden_dim] → [B, 5] with sigmoid.
    Each of the 5 outputs is an independent probability.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_intents: int = 5, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_intents),
        )
        self.num_intents = num_intents

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, D] representation vector
        Returns:
            logits: [B, 5] raw logits (use BCEWithLogitsLoss)
        """
        return self.net(x)


class ConfidenceHead(nn.Module):
    """Estimates decision confidence ∈ [0, 1]."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns [B, 1] confidence scores."""
        return self.net(x)
