"""
Closed-loop Feedback Mechanism.

Key idea from patent CN26CC0077A:
  Decision → Constraint Check → Residual → Feedback Injection → Corrected Decision

Two correction loops:
  Fast loop (output-level):  ẑ = z - η_fast * ∇_z L_physics
  Slow loop (parameter-level): W ← W - η_slow * ∇_W L_physics

The closed-loop operates WITHOUT real grid feedback. Instead:
  - "Feedback" = residual between decision and constraint-satisfying target
  - The residual is computed by checking each rule/constraint
  - Iterative correction converges toward compliant decisions
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ClosedLoopFeedback(nn.Module):
    """
    Wraps the decision model with a closed-loop feedback mechanism.

    At inference time, iteratively:
      1. Forward pass → decision_logits
      2. Check physics constraints → residual vector
      3. Fast correction: adjust decision_logits toward compliance
      4. Repeat until convergence or max_iterations
    """

    def __init__(
        self,
        max_iterations: int = 5,
        fast_lr: float = 0.01,
        convergence_threshold: float = 0.001,
    ):
        super().__init__()
        self.max_iterations = max_iterations
        self.fast_lr = fast_lr
        self.convergence_threshold = convergence_threshold

    def compute_residual(
        self,
        decision_logits: torch.Tensor,          # [B, 5]
        symbolic_loss_fn: nn.Module,
        physics_loss_fn: nn.Module,
        features: Dict[str, torch.Tensor],
        soc_proxy: Optional[torch.Tensor] = None,
        prev_decision_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the combined constraint violation residual.

        Returns:
            residual: [B, 5] per-intent violation signal
            metrics: Dict of violation statistics
        """
        B, D = decision_logits.shape
        device = decision_logits.device

        # Symbolic rule check
        sym_loss, sym_metrics = symbolic_loss_fn(
            decision_logits, features, soc_proxy,
        )

        # Physics constraint check
        phys_loss, phys_metrics = physics_loss_fn(
            decision_logits, features, prev_decision_logits, soc_proxy,
        )

        # Each rule/constraint produces a per-sample penalty signal
        # Aggregate into a residual vector of shape [B, 5]
        # Map rule violations to specific intents:
        #   R1 violation → Charge ESS correction needed (index 1)
        #   R2 violation → Increase Generation correction (index 0)
        #   R3 violation → Risk Warning correction (index 4)
        #   R4 violation → Charge ESS correction (index 1)
        #   R5 violation → Discharge ESS correction (index 2)
        #   R6 violation → Voltage Support correction (index 3)

        # For now, use gradient-based residual:
        # residual = -fast_lr * ∇_z (L_symbolic + L_physics)
        total_violation = sym_loss + phys_loss

        residual = torch.zeros(B, D, device=device)
        if decision_logits.requires_grad:
            grad = torch.autograd.grad(
                total_violation, decision_logits,
                retain_graph=True, create_graph=False,
                only_inputs=True,
            )[0]
            if grad is not None:
                residual = -self.fast_lr * grad

        metrics = {
            **sym_metrics,
            **phys_metrics,
            "total_violation": total_violation.item(),
            "residual_norm": residual.norm(dim=1).mean().item(),
        }

        return residual, metrics

    def closed_loop_iterate(
        self,
        model: nn.Module,
        features: Dict[str, torch.Tensor],
        symbolic_loss_fn: nn.Module,
        physics_loss_fn: nn.Module,
        soc_proxy: Optional[torch.Tensor] = None,
        track_history: bool = True,
    ) -> Dict:
        """
        Run closed-loop iteration: forward → check → correct → repeat.

        Args:
            model: NeuroSymbolicDecisionModel
            features: input features
            symbolic_loss_fn: SymbolicRuleLoss
            physics_loss_fn: PhysicsConstraintLoss
            soc_proxy: optional SOC estimates
            track_history: if True, record per-iteration metrics

        Returns:
            Dict with final decision, iteration history, and convergence stats
        """
        model.eval()
        device = next(model.parameters()).device

        # Move features to device
        features = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in features.items()}
        if soc_proxy is not None:
            soc_proxy = soc_proxy.to(device)

        # Initial forward pass
        with torch.no_grad():
            output = model(features)
            decision_logits = output["decision_logits"]  # [B, 5]

        history = []
        prev_logits = decision_logits.clone()
        violation_counts = []

        for iteration in range(self.max_iterations):
            # Make logits require grad for residual computation
            decision_logits = decision_logits.detach().requires_grad_(True)

            # Compute residual
            residual, metrics = self.compute_residual(
                decision_logits, symbolic_loss_fn, physics_loss_fn,
                features, soc_proxy, prev_logits,
            )

            # Apply correction (fast loop)
            corrected_logits = decision_logits + residual

            # Track changes
            delta = (corrected_logits - decision_logits).norm(dim=1).mean().item()

            history.append({
                "iteration": iteration,
                "delta": delta,
                **metrics,
            })

            # Check convergence
            if delta < self.convergence_threshold:
                break

            prev_logits = decision_logits.detach()
            decision_logits = corrected_logits.detach()

        # Final output
        final_probs = torch.sigmoid(decision_logits)

        return {
            "final_decision_logits": decision_logits.detach(),
            "final_decision_probs": final_probs,
            "num_iterations": len(history),
            "converged": history[-1]["delta"] < self.convergence_threshold if history else False,
            "iteration_history": history,
            "initial_logits": output["decision_logits"],
        }
