"""
Symbolic Rule Layer — Differentiable Soft Logic for Power Grid Decision Support.

Key idea from patent CN26CC0077A:
  - Expert rules (qualitative) → Expert targets (quantitative, via tanh mapping)
  - Action-vs-target gap → Soft truth value (via sigmoid with temperature)
  - All truth values → Product t-norm → Comprehensive trust
  - Trust → Regularization penalty in total loss

Rules (R1-R6):
  R1: IF ΔRenewable > δ THEN Charge ESS
  R2: IF ΔLoad > δ THEN Increase Generation
  R3: IF Wind Speed > τ THEN Risk Warning
  R4: IF SOC > SOC_max THEN Prohibit Charge ESS
  R5: IF SOC < SOC_min THEN Prohibit Discharge ESS
  R6: IF |ΔP| > Ramp Limit THEN Voltage Support
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicNeuron(nn.Module):
    """
    Maps continuous margin → soft boolean truth value ∈ [0, 1].

    truth = σ(temperature * margin)

    truth ≈ 1: constraint/rule is well-satisfied
    truth ≈ 0: constraint/rule is violated
    """

    def __init__(self, temperature: float = 20.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, margin: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.temperature * margin)


class SymbolicRuleLoss(nn.Module):
    """
    Differentiable symbolic rule loss.

    Computes:
      1. Expert target for each rule
      2. Soft truth value via LogicNeuron
      3. Comprehensive trust = product(truth_m)  (differentiable t-norm)
      4. Symbolic penalty = λ * (1 - trust)
    """

    def __init__(
        self,
        temperature: float = 20.0,
        rule_weights: Optional[list] = None,
        lambda_symbolic: float = 0.15,
    ):
        super().__init__()
        self.logic = LogicNeuron(temperature)
        self.num_rules = 6
        self.lambda_symbolic = lambda_symbolic

        if rule_weights is None:
            rule_weights = [1.0] * self.num_rules
        self.register_buffer("rule_weights", torch.tensor(rule_weights))

    def forward(
        self,
        decision_logits: torch.Tensor,    # [B, 5]
        features: Dict[str, torch.Tensor],  # Raw features for rule evaluation
        soc_proxy: Optional[torch.Tensor] = None,  # [B]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            decision_logits: [B, 5] model output logits
            features: Dict containing at least:
              - "delta_renewable": [B] change in renewable
              - "delta_load": [B] change in load
              - "delta_net_load": [B] change in net load
              - "wind_speed": [B] current wind speed
              - "delta_power": [B] power change magnitude
            soc_proxy: [B] estimated SOC (0-1)
        Returns:
            symbolic_loss: scalar
            metrics: Dict with per-rule truth values and total trust
        """
        B = decision_logits.shape[0]
        device = decision_logits.device

        # Decode decision intents
        decision_probs = torch.sigmoid(decision_logits)  # [B, 5]
        inc_gen = decision_probs[:, 0]       # Increase Generation
        chg_ess = decision_probs[:, 1]       # Charge ESS
        dis_ess = decision_probs[:, 2]       # Discharge ESS
        volt_sup = decision_probs[:, 3]      # Voltage Support
        risk_warn = decision_probs[:, 4]     # Risk Warning

        truths = []

        # ---- R1: IF ΔRenewable > δ THEN Charge ESS ----
        delta_ren = features.get("delta_renewable", torch.zeros(B, device=device))
        # Expert target: when renewable rises, should charge
        # target = tanh(delta_ren_normalized) → maps to [-1, 1], positive = charge
        expert_target_r1 = torch.tanh(delta_ren / (delta_ren.std() + 1e-6))
        gap_r1 = chg_ess - expert_target_r1  # small gap = follows rule
        margin_r1 = 0.15 - torch.abs(gap_r1)
        truth_r1 = self.logic(margin_r1)

        # ---- R2: IF ΔLoad > δ THEN Increase Generation ----
        delta_load = features.get("delta_load", torch.zeros(B, device=device))
        expert_target_r2 = torch.tanh(delta_load / (delta_load.std() + 1e-6))
        gap_r2 = inc_gen - expert_target_r2
        margin_r2 = 0.15 - torch.abs(gap_r2)
        truth_r2 = self.logic(margin_r2)

        # ---- R3: IF Wind Speed > τ THEN Risk Warning ----
        wind_speed = features.get("wind_speed", torch.zeros(B, device=device))
        tau = features.get("wind_threshold", torch.tensor(10.0, device=device))
        # Expert: risk should increase with wind speed above threshold
        wind_anomaly = F.relu(wind_speed - tau) / (tau + 1e-6)
        expert_target_r3 = torch.tanh(wind_anomaly)
        gap_r3 = risk_warn - expert_target_r3
        margin_r3 = 0.15 - torch.abs(gap_r3)
        truth_r3 = self.logic(margin_r3)

        # ---- R4: IF SOC > SOC_max THEN NOT Charge ESS ----
        if soc_proxy is not None:
            soc_max = features.get("soc_max", torch.tensor(0.90, device=device))
            soc_over = F.relu(soc_proxy - soc_max) / 0.10  # normalized overflow
            # Expert: when SOC is high, charging probability should be low
            expert_target_r4 = -torch.tanh(soc_over * 2.0)  # strongly discourage charging
            gap_r4 = chg_ess - expert_target_r4
        else:
            gap_r4 = torch.zeros(B, device=device)
        margin_r4 = 0.15 - torch.abs(gap_r4)
        truth_r4 = self.logic(margin_r4)

        # ---- R5: IF SOC < SOC_min THEN NOT Discharge ESS ----
        if soc_proxy is not None:
            soc_min = features.get("soc_min", torch.tensor(0.10, device=device))
            soc_under = F.relu(soc_min - soc_proxy) / 0.10
            expert_target_r5 = -torch.tanh(soc_under * 2.0)
            gap_r5 = dis_ess - expert_target_r5
        else:
            gap_r5 = torch.zeros(B, device=device)
        margin_r5 = 0.15 - torch.abs(gap_r5)
        truth_r5 = self.logic(margin_r5)

        # ---- R6: IF |ΔP| > Ramp Limit THEN Voltage Support ----
        delta_p = features.get("delta_power", torch.zeros(B, device=device))
        ramp_limit = features.get("ramp_limit", torch.tensor(0.15, device=device))
        ramp_anomaly = F.relu(torch.abs(delta_p) - ramp_limit) / (ramp_limit + 1e-6)
        expert_target_r6 = torch.tanh(ramp_anomaly)
        gap_r6 = volt_sup - expert_target_r6
        margin_r6 = 0.15 - torch.abs(gap_r6)
        truth_r6 = self.logic(margin_r6)

        # Collect all truths
        truth_list = [truth_r1, truth_r2, truth_r3, truth_r4, truth_r5, truth_r6]
        truth_tensor = torch.stack(truth_list, dim=0)  # [6, B]

        # Differentiable t-norm: product (from patent)
        comprehensive_trust = truth_tensor.prod(dim=0)  # [B]

        # Weighted symbolic penalty
        weighted_trust = (truth_tensor * self.rule_weights.unsqueeze(1)).mean(dim=0)  # [B]
        symbolic_loss = self.lambda_symbolic * (1.0 - weighted_trust).mean()

        # Per-rule penalty for diagnostics
        per_rule_penalty = {
            f"penalty_r{i+1}": (1.0 - truth_tensor[i].mean()).item()
            for i in range(self.num_rules)
        }

        metrics = {
            "symbolic_loss": symbolic_loss.item(),
            "comprehensive_trust": comprehensive_trust.mean().item(),
            "weighted_trust": weighted_trust.mean().item(),
            **{f"truth_r{i+1}": truth_tensor[i].mean().item() for i in range(self.num_rules)},
            **per_rule_penalty,
        }

        return symbolic_loss, metrics
