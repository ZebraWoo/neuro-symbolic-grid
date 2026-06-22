"""
Physics Constraint Loss for Power Grid Decision Support.

Constraints (adapted from patent CN26CC0077A for PSML data without IEEE 39-bus):

  1. Ramp Rate:     |ΔP| < R_max
  2. SOC Bounds:    SOC_min < SOC < SOC_max
  3. Power Balance: |Gen + ESS - NetLoad| < ε
  4. Curtailment:   Curtail ≤ Curtail_max

All constraints are converted to differentiable penalty terms.

L_physics = w_ramp * L_ramp + w_soc * L_soc + w_balance * L_balance + w_curtail * L_curtail
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PhysicsConstraintLoss(nn.Module):
    """
    Differentiable physics constraint loss.

    Does NOT require actual grid simulation. All constraints are computed
    from the model's decision intents and the input features.
    """

    def __init__(
        self,
        ramp_limit: float = 0.15,
        soc_min: float = 0.10,
        soc_max: float = 0.90,
        balance_tolerance: float = 0.05,
        curtail_max: float = 0.20,
        w_ramp: float = 1.0,
        w_soc: float = 1.0,
        w_balance: float = 1.5,
        w_curtail: float = 0.5,
        lambda_physics: float = 0.25,
    ):
        super().__init__()
        self.ramp_limit = ramp_limit
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.balance_tolerance = balance_tolerance
        self.curtail_max = curtail_max

        self.w_ramp = w_ramp
        self.w_soc = w_soc
        self.w_balance = w_balance
        self.w_curtail = w_curtail
        self.lambda_physics = lambda_physics

    def forward(
        self,
        decision_logits: torch.Tensor,     # [B, 5]
        features: Dict[str, torch.Tensor],
        prev_decision_logits: torch.Tensor = None,  # [B, 5] from previous timestep
        soc_proxy: torch.Tensor = None,     # [B]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics constraint violations as differentiable penalties.
        """
        B = decision_logits.shape[0]
        device = decision_logits.device

        decision_probs = torch.sigmoid(decision_logits)
        inc_gen = decision_probs[:, 0]
        chg_ess = decision_probs[:, 1]
        dis_ess = decision_probs[:, 2]
        volt_sup = decision_probs[:, 3]
        risk_warn = decision_probs[:, 4]

        # ---- 1) Ramp Rate Constraint ----
        # |ΔP| < R_max where ΔP is change in generation decision
        if prev_decision_logits is not None:
            prev_probs = torch.sigmoid(prev_decision_logits)
            prev_gen = prev_probs[:, 0]
            delta_gen = inc_gen - prev_gen
            ramp_violation = F.relu(torch.abs(delta_gen) - self.ramp_limit)
            L_ramp = (ramp_violation ** 2).mean()
        else:
            L_ramp = torch.tensor(0.0, device=device)

        # Also check implicit ramp: ESS charge <-> discharge switching
        # Rapid switch from charge to discharge (or vice versa) is a ramp violation
        if prev_decision_logits is not None:
            prev_probs = torch.sigmoid(prev_decision_logits)
            prev_ess_net = prev_probs[:, 1] - prev_probs[:, 2]  # charge - discharge
            ess_net = chg_ess - dis_ess
            delta_ess = ess_net - prev_ess_net
            ess_ramp_violation = F.relu(torch.abs(delta_ess) - self.ramp_limit)
            L_ramp = L_ramp + 0.5 * (ess_ramp_violation ** 2).mean()

        # ---- 2) SOC Bounds Constraint ----
        # Simplified: use decision-level proxy
        # High charge + high SOC → violation, High discharge + low SOC → violation
        if soc_proxy is not None:
            soc_charge_violation = F.relu(soc_proxy - self.soc_max) * chg_ess
            soc_discharge_violation = F.relu(self.soc_min - soc_proxy) * dis_ess
            L_soc = (soc_charge_violation ** 2).mean() + (soc_discharge_violation ** 2).mean()
        else:
            # Without SOC proxy, penalize simultaneous charge + discharge
            L_soc = (chg_ess * dis_ess).mean()  # shouldn't do both at once

        # ---- 3) Power Balance Constraint ----
        # Net decision should roughly match net load change
        delta_net_load = features.get("delta_net_load", torch.zeros(B, device=device))
        # Generation + ESS contribution should balance net load change
        gen_contrib = inc_gen * 0.5  # scaled
        ess_contrib = (chg_ess - dis_ess) * 0.3  # ESS charging reduces net, discharging increases
        decision_balance = gen_contrib + ess_contrib

        # Normalize delta_net_load to comparable scale
        delta_nl_norm = torch.tanh(delta_net_load / (delta_net_load.std() + 1e-6)) * 0.5
        balance_error = decision_balance - delta_nl_norm
        balance_violation = F.relu(torch.abs(balance_error) - self.balance_tolerance)
        L_balance = (balance_violation ** 2).mean()

        # ---- 4) Curtailment Constraint ----
        # Renewable curtailment should be limited
        # Proxy: if renewable is high but charge is low, potential curtailment
        delta_ren = features.get("delta_renewable", torch.zeros(B, device=device))
        renewable_surge = F.relu(delta_ren)
        # When renewable surges but ESS doesn't charge enough → curtailment risk
        curtail_proxy = F.relu(renewable_surge - chg_ess * renewable_surge.std())
        curtail_violation = F.relu(curtail_proxy / (renewable_surge.std() + 1e-6) - self.curtail_max)
        L_curtail = (curtail_violation ** 2).mean()

        # ---- Total ----
        L_physics_raw = (
            self.w_ramp * L_ramp +
            self.w_soc * L_soc +
            self.w_balance * L_balance +
            self.w_curtail * L_curtail
        )
        L_physics = self.lambda_physics * L_physics_raw

        metrics = {
            "L_ramp": L_ramp.item(),
            "L_soc": L_soc.item(),
            "L_balance": L_balance.item(),
            "L_curtail": L_curtail.item(),
            "L_physics": L_physics.item(),
            "ramp_violations": int((ramp_violation > 0).sum().item()) if prev_decision_logits is not None else 0,
            "soc_violations": int((soc_charge_violation > 0).sum().item() + (soc_discharge_violation > 0).sum().item()) if soc_proxy is not None else 0,
            "balance_violations": int((balance_violation > 0).sum().item()),
            "curtail_violations": int((curtail_violation > 0).sum().item()),
        }

        return L_physics, metrics
