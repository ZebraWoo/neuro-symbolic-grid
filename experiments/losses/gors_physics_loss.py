"""
GORS Physics-Constrained Loss.

Constraints on the risk score's consistency with physical boundaries:
  1. Power Balance (L_pb): penalizes risk that ignores supply-demand mismatch
  2. Ramp Constraint (L_rp): restricts inter-step risk volatility
  3. Renewable Capacity (L_re): ensures risk reflects renewable penetration limits

Consistency residual r_phy = L_phy for closed-loop feedback.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GORSPhysicsLoss(nn.Module):
    def __init__(
        self,
        w_balance: float = 1.5,
        w_ramp: float = 1.0,
        w_capacity: float = 0.5,
        lambda_physics: float = 0.25,
    ):
        super().__init__()
        self.w_balance = w_balance
        self.w_ramp = w_ramp
        self.w_capacity = w_capacity
        self.lambda_physics = lambda_physics

    def forward(
        self,
        gors_pred: torch.Tensor,              # [B, 1] or [B]
        modalities: Dict[str, torch.Tensor],
        prev_gors: torch.Tensor = None,       # [B] previous-step GORS
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Returns:
            L_phy: scalar physics loss
            r_phy: consistency residual (scalar, detached)
            metrics: per-constraint diagnostics
        """
        if gors_pred.dim() == 2:
            gors_pred = gors_pred.squeeze(-1)  # [B]
        B = gors_pred.shape[0]
        device = gors_pred.device

        # ---- 1) Power Balance Constraint ----
        # High risk should correlate with load-generation imbalance
        load = modalities.get("load")
        renewable = modalities.get("renewable")
        if load is not None and renewable is not None:
            load_val = load[:, -1, 0].abs() if load.dim() == 3 else load.abs()
            ren_val = renewable[:, -1, :].sum(dim=1).abs() if renewable.dim() == 3 else renewable.abs()
            imbalance = torch.abs(load_val - ren_val) / (load_val.abs().mean() + 1e-6)
            # Risk should be HIGH when imbalance is HIGH
            balance_gap = torch.abs(gors_pred - torch.sigmoid(imbalance))
            L_balance = self.w_balance * (balance_gap ** 2).mean()
        else:
            L_balance = torch.tensor(0.0, device=device)

        # ---- 2) Ramp Constraint ----
        # Risk should not change too abruptly between steps
        if prev_gors is not None:
            delta_gors = torch.abs(gors_pred - prev_gors)
            ramp_violation = F.relu(delta_gors - 0.3)  # max 30% risk change per step
            L_ramp = self.w_ramp * (ramp_violation ** 2).mean()
        else:
            L_ramp = torch.tensor(0.0, device=device)

        # ---- 3) Renewable Capacity Constraint ----
        # Risk should reflect renewable penetration limits
        if renewable is not None:
            ren_val = renewable[:, -1, :].sum(dim=1).abs() if renewable.dim() == 3 else renewable.abs()
            ren_penetration = ren_val / (load_val.abs().mean() + 1e-6)
            # High penetration → elevated risk (but bounded)
            ren_risk_gap = torch.abs(gors_pred - torch.sigmoid(2.0 * (ren_penetration - 0.5)))
            L_capacity = self.w_capacity * (ren_risk_gap ** 2).mean()
        else:
            L_capacity = torch.tensor(0.0, device=device)

        # ---- Total Physics Loss ----
        L_phy_raw = L_balance + L_ramp + L_capacity
        L_phy = self.lambda_physics * L_phy_raw

        # Consistency residual
        r_phy = L_phy_raw.detach()

        metrics = {
            "L_balance": L_balance.item(),
            "L_ramp": L_ramp.item(),
            "L_capacity": L_capacity.item(),
            "L_phy": L_phy.item(),
            "r_phy": r_phy.item(),
        }
        return L_phy, r_phy, metrics
