"""
Shared evaluation utilities for all experiments.

Metrics:
  - Decision Accuracy (per-class F1, Macro F1, Hamming Loss)
  - Rule Satisfaction Rate (RSR)
  - Physics Violation Count
  - Closed-loop Convergence
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Decision Accuracy Metrics
# ---------------------------------------------------------------------------

def compute_decision_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    intent_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute multi-label classification metrics.

    Args:
        logits: [N, 5] raw logits
        labels: [N, 5] binary ground truth
    Returns:
        Dict with macro_f1, micro_f1, hamming_loss, per_class_f1, accuracy
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    N, C = labels.shape

    per_class = {}
    f1s = []
    for i in range(C):
        tp = ((preds[:, i] == 1) & (labels[:, i] == 1)).sum().item()
        fp = ((preds[:, i] == 1) & (labels[:, i] == 0)).sum().item()
        fn = ((preds[:, i] == 0) & (labels[:, i] == 1)).sum().item()
        tn = ((preds[:, i] == 0) & (labels[:, i] == 0)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        name = intent_names[i] if intent_names else f"class_{i}"
        per_class[f"{name}_precision"] = round(precision, 4)
        per_class[f"{name}_recall"] = round(recall, 4)
        per_class[f"{name}_f1"] = round(f1, 4)
        f1s.append(f1)

    # Micro F1 (global)
    tp_micro = ((preds == 1) & (labels == 1)).sum().item()
    fp_micro = ((preds == 1) & (labels == 0)).sum().item()
    fn_micro = ((preds == 0) & (labels == 1)).sum().item()
    precision_micro = tp_micro / max(tp_micro + fp_micro, 1)
    recall_micro = tp_micro / max(tp_micro + fn_micro, 1)
    micro_f1 = 2 * precision_micro * recall_micro / max(precision_micro + recall_micro, 1e-8)

    macro_f1 = float(np.mean(f1s))
    hamming_loss = (preds != labels).float().mean().item()
    subset_accuracy = (preds == labels).all(dim=1).float().mean().item()

    return {
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "hamming_loss": round(hamming_loss, 4),
        "subset_accuracy": round(subset_accuracy, 4),
        **per_class,
    }


# ---------------------------------------------------------------------------
# Rule Satisfaction Rate (RSR)
# ---------------------------------------------------------------------------

def compute_rule_satisfaction(
    decision_probs: torch.Tensor,   # [N, 5]
    features: Dict[str, torch.Tensor],
    soc_proxy: Optional[torch.Tensor] = None,
    temperature: float = 20.0,
) -> Dict:
    """
    Compute per-rule satisfaction rate.

    A rule is "satisfied" when its soft truth value > 0.5.
    """
    B = decision_probs.shape[0]
    device = decision_probs.device

    inc_gen = decision_probs[:, 0]
    chg_ess = decision_probs[:, 1]
    dis_ess = decision_probs[:, 2]
    volt_sup = decision_probs[:, 3]
    risk_warn = decision_probs[:, 4]

    results = {}

    # R1: ΔRenewable↑ → Charge ESS
    delta_ren = features.get("delta_renewable", torch.zeros(B, device=device))
    expert_r1 = torch.tanh(delta_ren / (delta_ren.std() + 1e-6))
    gap_r1 = chg_ess - expert_r1
    truth_r1 = torch.sigmoid(temperature * (0.15 - torch.abs(gap_r1)))
    results["R1_charge_ess"] = (truth_r1 > 0.5).float().mean().item()

    # R2: ΔLoad↑ → Increase Generation
    delta_load = features.get("delta_load", torch.zeros(B, device=device))
    expert_r2 = torch.tanh(delta_load / (delta_load.std() + 1e-6))
    gap_r2 = inc_gen - expert_r2
    truth_r2 = torch.sigmoid(temperature * (0.15 - torch.abs(gap_r2)))
    results["R2_inc_gen"] = (truth_r2 > 0.5).float().mean().item()

    # R3: Wind Speed↑ → Risk Warning
    wind_speed = features.get("wind_speed", torch.zeros(B, device=device))
    tau = features.get("wind_threshold", torch.tensor(10.0, device=device))
    wind_anomaly = F.relu(wind_speed - tau) / (tau + 1e-6)
    expert_r3 = torch.tanh(wind_anomaly)
    gap_r3 = risk_warn - expert_r3
    truth_r3 = torch.sigmoid(temperature * (0.15 - torch.abs(gap_r3)))
    results["R3_risk_warn"] = (truth_r3 > 0.5).float().mean().item()

    # R4: SOC > SOC_max → NOT Charge
    if soc_proxy is not None:
        soc_over = F.relu(soc_proxy - 0.90) / 0.10
        expert_r4 = -torch.tanh(soc_over * 2.0)
        gap_r4 = chg_ess - expert_r4
        truth_r4 = torch.sigmoid(temperature * (0.15 - torch.abs(gap_r4)))
        results["R4_soc_max"] = (truth_r4 > 0.5).float().mean().item()
    else:
        results["R4_soc_max"] = 1.0

    # R5: SOC < SOC_min → NOT Discharge
    if soc_proxy is not None:
        soc_under = F.relu(0.10 - soc_proxy) / 0.10
        expert_r5 = -torch.tanh(soc_under * 2.0)
        gap_r5 = dis_ess - expert_r5
        truth_r5 = torch.sigmoid(temperature * (0.15 - torch.abs(gap_r5)))
        results["R5_soc_min"] = (truth_r5 > 0.5).float().mean().item()
    else:
        results["R5_soc_min"] = 1.0

    # R6: |ΔP| > Ramp → Voltage Support
    delta_p = features.get("delta_power", torch.zeros(B, device=device))
    ramp_anomaly = F.relu(torch.abs(delta_p) - 0.15) / 0.15
    expert_r6 = torch.tanh(ramp_anomaly)
    gap_r6 = volt_sup - expert_r6
    truth_r6 = torch.sigmoid(temperature * (0.15 - torch.abs(gap_r6)))
    results["R6_volt_sup"] = (truth_r6 > 0.5).float().mean().item()

    # Average RSR
    values = list(results.values())
    results["avg_rsr"] = round(float(np.mean(values)), 4)

    return {k: round(v, 4) for k, v in results.items()}


# ---------------------------------------------------------------------------
# Physics Violation Count
# ---------------------------------------------------------------------------

def count_physics_violations(
    decision_probs: torch.Tensor,
    prev_probs: Optional[torch.Tensor],
    features: Dict[str, torch.Tensor],
    soc_proxy: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Count hard violation instances across the dataset.
    """
    B = decision_probs.shape[0]

    inc_gen = decision_probs[:, 0]
    chg_ess = decision_probs[:, 1]
    dis_ess = decision_probs[:, 2]

    violations = {}

    # Ramp violation
    if prev_probs is not None:
        delta_gen = inc_gen - prev_probs[:, 0]
        violations["ramp"] = int((torch.abs(delta_gen) > 0.15).sum().item())
    else:
        violations["ramp"] = 0

    # SOC violation
    if soc_proxy is not None:
        soc_charge_v = (soc_proxy > 0.90) & (chg_ess > 0.5)
        soc_discharge_v = (soc_proxy < 0.10) & (dis_ess > 0.5)
        violations["soc"] = int((soc_charge_v | soc_discharge_v).sum().item())
    else:
        violations["soc"] = 0

    # Power balance violation
    # Simplified check: extreme simultaneous charge+discharge
    violations["balance"] = int(((chg_ess > 0.5) & (dis_ess > 0.5)).sum().item())

    # Curtailment violation
    delta_ren = features.get("delta_renewable", torch.zeros(B))
    renewable_surge = F.relu(delta_ren)
    curtail_proxy = F.relu(renewable_surge - chg_ess * renewable_surge.std())
    curtail_ratio = curtail_proxy / (renewable_surge.std() + 1e-6)
    violations["curtail"] = int((curtail_ratio > 0.20).sum().item())

    violations["total"] = sum(violations.values())
    return violations


# ---------------------------------------------------------------------------
# Closed-loop Convergence
# ---------------------------------------------------------------------------

def evaluate_closed_loop(
    model, features, symbolic_loss_fn, physics_loss_fn,
    max_iter: int = 5, fast_lr: float = 0.01,
) -> Dict:
    """
    Run closed-loop iteration and track convergence.
    """
    from experiments.losses.closed_loop_loss import ClosedLoopFeedback

    cl = ClosedLoopFeedback(
        max_iterations=max_iter,
        fast_lr=fast_lr,
    )

    result = cl.closed_loop_iterate(
        model, features, symbolic_loss_fn, physics_loss_fn,
        track_history=True,
    )

    return result


# ---------------------------------------------------------------------------
# Format results as LaTeX table rows
# ---------------------------------------------------------------------------

def format_latex_table(results: Dict[str, Dict], metric: str = "macro_f1") -> str:
    """Format results dictionary as LaTeX table."""
    models = list(results.keys())
    header = " & ".join(["Model"] + models) + " \\\\"
    lines = [header, "\\hline"]

    # Collect metrics
    values = [f"{results[m].get(metric, 0):.4f}" for m in models]
    lines.append(f"{metric} & " + " & ".join(values) + " \\\\")

    return "\n".join(lines)
