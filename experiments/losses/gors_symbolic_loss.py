"""
GORS Symbolic Rule Layer — Differentiable Soft Logic for Risk Compliance.

Refactored per advisor feedback:
  - Rules evaluate whether GORS properly reflects risk factors
  - Expert targets auto-generated from raw features (no manual annotation)
  - Consistency residual r_rule = 1 - T (product t-norm of rule truths)

Rules:
  R1: High temperature → elevated weather risk  (e_temp = σ(W_temp))
  R2: High wind speed → elevated weather risk   (e_wind = σ(W_wind))
  R3: Large net load change → imbalance risk    (e_load = σ(|ΔL|))
  R4: Large renewable change → volatility risk  (e_ren  = σ(|ΔR|))
  R5: Combined extreme → systemic risk          (e_sys  = σ(combined))
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LogicNeuron(nn.Module):
    """Maps continuous margin → soft truth ∈ [0,1]."""
    def __init__(self, temperature: float = 20.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, margin: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.temperature * margin)


class GORSSymbolicLoss(nn.Module):
    """
    Differentiable symbolic risk compliance loss.

    Computes:
      1. Expert risk target e_k for each rule (auto-derived from features)
      2. Gap g_k = |y_pred - e_k|  (how much GORS deviates from expert target)
      3. Soft truth T_k = σ(α · (δ - g_k))  (δ = tolerance margin)
      4. Comprehensive trust T = Π T_k  (product t-norm)
      5. Symbolic risk violation loss L_rule = -ln(T)
      6. Consistency residual r_rule = 1 - T  (for closed-loop feedback)
    """

    def __init__(
        self,
        temperature: float = 20.0,
        tolerance: float = 0.15,
        lambda_symbolic: float = 0.15,
    ):
        super().__init__()
        self.logic = LogicNeuron(temperature)
        self.tolerance = tolerance
        self.lambda_symbolic = lambda_symbolic

    def forward(
        self,
        gors_pred: torch.Tensor,          # [B, 1] or [B]
        modalities: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            L_rule: scalar symbolic loss
            metrics: per-rule truth values and consistency residual
        """
        if gors_pred.dim() == 2:
            gors_pred = gors_pred.squeeze(-1)  # [B]
        B = gors_pred.shape[0]
        device = gors_pred.device

        # Extract features from modalities
        feat = self._extract_rule_features(modalities)

        # ---- R1: High Temperature → Elevated Risk ----
        # Expert: risk should increase with temperature anomaly
        temp = feat["temperature"]  # [B]
        e_temp = torch.sigmoid(2.0 * temp)  # [B,1] mapped to [0,1]
        g_temp = torch.abs(gors_pred - e_temp)
        m_temp = self.tolerance - g_temp
        T1 = self.logic(m_temp)

        # ---- R2: High Wind Speed → Elevated Risk ----
        wind = feat["wind_speed"]  # [B]
        e_wind = torch.sigmoid(2.0 * wind)
        g_wind = torch.abs(gors_pred - e_wind)
        m_wind = self.tolerance - g_wind
        T2 = self.logic(m_wind)

        # ---- R3: Large Net Load Change → Imbalance Risk ----
        delta_load = feat["delta_load"]  # [B]
        e_load = torch.sigmoid(3.0 * torch.abs(delta_load))
        g_load = torch.abs(gors_pred - e_load)
        m_load = self.tolerance - g_load
        T3 = self.logic(m_load)

        # ---- R4: Large Renewable Change → Volatility Risk ----
        delta_ren = feat["delta_renewable"]  # [B]
        e_ren = torch.sigmoid(3.0 * torch.abs(delta_ren))
        g_ren = torch.abs(gors_pred - e_ren)
        m_ren = self.tolerance - g_ren
        T4 = self.logic(m_ren)

        # ---- R5: Combined Extreme → Systemic Risk ----
        combined = 0.3 * temp + 0.3 * wind + 0.2 * torch.abs(delta_load) + 0.2 * torch.abs(delta_ren)
        e_sys = torch.sigmoid(2.0 * combined)
        g_sys = torch.abs(gors_pred - e_sys)
        m_sys = self.tolerance - g_sys
        T5 = self.logic(m_sys)

        # Mean t-norm: comprehensive trust (stable, no exponential collapse)
        truth_tensor = torch.stack([T1, T2, T3, T4, T5], dim=0)  # [5, B]
        T_comprehensive = truth_tensor.mean(dim=0)  # [B]

        # Symbolic loss: MSE between current trust and perfect trust (1.0)
        L_rule = self.lambda_symbolic * ((1.0 - T_comprehensive) ** 2).mean()

        # Consistency residual (for closed-loop feedback)
        r_rule = (1.0 - T_comprehensive).mean().detach()

        metrics = {
            "L_rule": L_rule.item(),
            "r_rule": r_rule.item(),
            "trust_comprehensive": T_comprehensive.mean().item(),
            "T1_temp": T1.mean().item(),
            "T2_wind": T2.mean().item(),
            "T3_load": T3.mean().item(),
            "T4_ren": T4.mean().item(),
            "T5_sys": T5.mean().item(),
        }
        return L_rule, r_rule, metrics

    @staticmethod
    def _extract_rule_features(modalities: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Extract rule-relevant features from modality tensors."""
        device = list(modalities.values())[0].device
        B = list(modalities.values())[0].shape[0]

        # Temperature from weather modality (index 3 in weather: [Dew, Wind, Hum, Temp])
        weather = modalities.get("weather")
        if weather is not None and weather.dim() == 3:
            temp = weather[:, -1, 3]  # last timestep, temperature column
        else:
            temp = torch.zeros(B, device=device)

        # Wind speed from weather (index 1)
        if weather is not None and weather.dim() == 3:
            wind = weather[:, -1, 1]
        else:
            wind = torch.zeros(B, device=device)

        # Load change proxy: last timestep load value
        load = modalities.get("load")
        if load is not None and load.dim() == 3:
            load_val = load[:, -1, 0]
            delta_load = load_val - load_val.mean()
        else:
            delta_load = torch.zeros(B, device=device)

        # Renewable change proxy
        renewable = modalities.get("renewable")
        if renewable is not None and renewable.dim() == 3:
            ren_val = renewable[:, -1, :].sum(dim=1)
            delta_ren = ren_val - ren_val.mean()
        else:
            delta_ren = torch.zeros(B, device=device)

        return {
            "temperature": temp,
            "wind_speed": wind,
            "delta_load": delta_load,
            "delta_renewable": delta_ren,
        }
