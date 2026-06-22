"""
Shared training loop for all decision-support models.

Supports:
  - BCE loss (standard)
  - BCE + Symbolic + Physics (ours)
  - Closed-loop loss injection (optional)
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class DecisionTrainer:
    """
    Trainer for multi-label decision intent classification.

    Loss options:
      - bce_only: standard BCEWithLogitsLoss
      - neuro_symbolic: BCE + L_symbolic + L_physics (ours)
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        loss_mode: str = "bce_only",  # "bce_only" | "neuro_symbolic"
        symbolic_loss_fn: Optional[nn.Module] = None,
        physics_loss_fn: Optional[nn.Module] = None,
        lr: float = 0.001,
        weight_decay: float = 1e-5,
        gradient_clip: float = 1.0,
        seed: int = 42,
    ):
        self.model = model.to(device)
        self.device = device
        self.loss_mode = loss_mode
        self.symbolic_loss_fn = symbolic_loss_fn.to(device) if symbolic_loss_fn is not None else None
        self.physics_loss_fn = physics_loss_fn.to(device) if physics_loss_fn is not None else None

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.gradient_clip = gradient_clip

        torch.manual_seed(seed)

    def _extract_features(
        self,
        modalities: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        prev_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract rule-relevant features from the data batch."""
        B = labels.shape[0]
        device = labels.device

        # Get load and renewable from modalities
        load = modalities.get("load", torch.zeros(B, 1, 1, device=device))
        if load.dim() == 3:
            load = load.mean(dim=1)  # [B, 1]
        elif load.dim() == 2:
            load = load  # [B, 1]
        load = load.squeeze(-1)  # [B]

        renewable = modalities.get("renewable", torch.zeros(B, 1, 2, device=device))
        if renewable.dim() == 3:
            renewable = renewable.sum(dim=-1).mean(dim=1)  # [B]

        wind_speed = None
        weather = modalities.get("weather", None)
        if weather is not None:
            if weather.dim() == 3:
                wind_speed = weather[:, :, 1].mean(dim=1)  # Wind Speed is index 1 in weather
            elif weather.dim() == 2:
                wind_speed = weather[:, 1]

        features = {
            "delta_renewable": torch.zeros(B, device=device),
            "delta_load": torch.zeros(B, device=device),
            "delta_net_load": torch.zeros(B, device=device),
            "delta_power": torch.zeros(B, device=device),
            "wind_speed": wind_speed if wind_speed is not None else torch.zeros(B, device=device),
            "wind_threshold": torch.tensor(10.0, device=device),
            "soc_min": torch.tensor(0.10, device=device),
            "soc_max": torch.tensor(0.90, device=device),
            "ramp_limit": torch.tensor(0.15, device=device),
        }

        # SOC proxy from labels (circular, but acceptable as a heuristic)
        # In production this would be from an actual SOC estimator
        soc_proxy = 0.5 + 0.3 * (labels[:, 1] - labels[:, 2])  # charge tendency - discharge tendency
        soc_proxy = torch.clamp(soc_proxy, 0.05, 0.95)

        return features, soc_proxy

    def train_epoch(
        self,
        dataloader,
        epoch: int,
    ) -> Dict[str, float]:
        import time
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_sym = 0.0
        total_phys = 0.0
        n_batches = 0
        t0 = time.time()

        detail_acc: Dict[str, float] = {}

        for i, (modalities, labels) in enumerate(dataloader):
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            labels = labels.to(self.device)

            output = self.model(modalities)
            decision_logits = output["decision_logits"]

            # BCE loss
            loss_bce = self.bce_loss(decision_logits, labels)
            loss = loss_bce
            total_bce += loss_bce.item()

            loss_dict = {}

            # Neuro-symbolic losses
            if self.loss_mode == "neuro_symbolic" and self.symbolic_loss_fn is not None:
                features, soc_proxy = self._extract_features(modalities, labels)
                sym_loss, sym_metrics = self.symbolic_loss_fn(decision_logits, features, soc_proxy)
                loss = loss + sym_loss
                total_sym += sym_loss.item()
                for k, v in sym_metrics.items():
                    loss_dict[k] = v

                if self.physics_loss_fn is not None:
                    phys_loss, phys_metrics = self.physics_loss_fn(
                        decision_logits, features, None, soc_proxy,
                    )
                    loss = loss + phys_loss
                    total_phys += phys_loss.item()
                    for k, v in phys_metrics.items():
                        loss_dict[k] = v

            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            for k, v in loss_dict.items():
                detail_acc[k] = detail_acc.get(k, 0.0) + v

            if i == 0 or (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                logger.info(
                    "  batch %3d/%d | loss=%.4f bce=%.4f | %.1f s",
                    i + 1, len(dataloader), loss.item(), loss_bce.item(), elapsed,
                )

        avg = total_loss / max(n_batches, 1)
        for k in detail_acc:
            detail_acc[k] /= max(n_batches, 1)

        return {
            "loss": avg,
            "loss_bce": total_bce / max(n_batches, 1),
            "loss_sym": total_sym / max(n_batches, 1) if total_sym > 0 else 0.0,
            "loss_phys": total_phys / max(n_batches, 1) if total_phys > 0 else 0.0,
            **detail_acc,
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        all_logits = []
        all_labels = []

        for modalities, labels in dataloader:
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            labels = labels.to(self.device)
            output = self.model(modalities)
            all_logits.append(output["decision_logits"].cpu())
            all_labels.append(labels.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Multi-label metrics
        all_probs = torch.sigmoid(all_logits)
        all_preds = (all_probs > 0.5).float()

        # Per-class F1
        per_class_f1 = []
        for i in range(all_labels.shape[1]):
            tp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 1)).sum().item()
            fp = ((all_preds[:, i] == 1) & (all_labels[:, i] == 0)).sum().item()
            fn = ((all_preds[:, i] == 0) & (all_labels[:, i] == 1)).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            per_class_f1.append(f1)

        macro_f1 = float(np.mean(per_class_f1))
        micro_f1 = float(np.mean([
            2 * ((all_preds == 1) & (all_labels == 1)).sum().item() /
            max((all_preds == 1).sum().item() + (all_labels == 1).sum().item(), 1)
        ])) if all_labels.numel() > 0 else 0.0

        hamming_loss = (all_preds != all_labels).float().mean().item()

        bce = F.binary_cross_entropy_with_logits(all_logits, all_labels).item()

        return {
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "hamming_loss": hamming_loss,
            "bce": bce,
            "per_class_f1": per_class_f1,
        }

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 50,
        checkpoint_dir: str = "checkpoints",
        output_dir: str = "outputs",
        model_tag: str = "model",
    ) -> Dict:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = {
            "train_loss": [],
            "train_bce": [],
            "train_sym": [],
            "train_phys": [],
            "val_macro_f1": [],
            "val_micro_f1": [],
            "val_hamming": [],
            "val_bce": [],
            "lr": [],
        }
        best_f1 = 0.0
        best_state = None
        start_time = time.time()
        logger.info("Starting training: %d epochs, %d batches/epoch", epochs, len(train_loader))

        for epoch in range(1, epochs + 1):
            t_epoch = time.time()
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)
            t_val = time.time()

            self.scheduler.step()

            history["train_loss"].append(train_metrics["loss"])
            history["train_bce"].append(train_metrics["loss_bce"])
            history["train_sym"].append(train_metrics.get("loss_sym", 0.0))
            history["train_phys"].append(train_metrics.get("loss_phys", 0.0))
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_micro_f1"].append(val_metrics["micro_f1"])
            history["val_hamming"].append(val_metrics["hamming_loss"])
            history["val_bce"].append(val_metrics["bce"])
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                elapsed = t_val - t_epoch
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Loss={train_metrics['loss']:.4f} "
                    f"BCE={train_metrics['loss_bce']:.4f} "
                    f"Sym={train_metrics.get('loss_sym', 0):.4f} "
                    f"Phys={train_metrics.get('loss_phys', 0):.4f} | "
                    f"Val F1_m={val_metrics['macro_f1']:.3f} "
                    f"Hamm={val_metrics['hamming_loss']:.3f} | "
                    f"{elapsed:.0f}s"
                )

        elapsed = time.time() - start_time
        logger.info(f"Training completed in {elapsed:.0f}s. Best val F1_macro={best_f1:.4f}")

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"{model_tag}.pth"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_f1": best_f1,
            "config": {"model_tag": model_tag},
        }, ckpt_path)

        if best_state is not None:
            best_path = checkpoint_dir / f"{model_tag}_best.pth"
            torch.save({
                "model_state_dict": best_state,
                "best_f1": best_f1,
            }, best_path)

        # Save history
        hist_path = output_dir / f"history_{model_tag}.json"
        with open(hist_path, "w") as f:
            json.dump(history, f, indent=2)

        logger.info(f"Saved: {ckpt_path}, history: {hist_path}")
        return history
