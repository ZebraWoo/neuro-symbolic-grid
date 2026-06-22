#!/usr/bin/env python3
"""
Train the GORS (Grid Operational Risk Score) framework.

Key differences from previous version:
  - Output: single scalar y ∈ [0,1] (not 5 binary intents)
  - Labels: derived from data volatility (not fake decisions)
  - Loss:   MSE + L_rule + L_phy + L_fb (self-supervised)
  - Closed-loop: r_t = r_rule + r_phy → feedback current → soma injection

Usage:
  # Quick test
  python experiments/train_gors.py --epochs 5 --zones-per-split 2

  # Full training
  python experiments/train_gors.py --epochs 50 --batch-size 32

  # Ablation: no symbolic
  python experiments/train_gors.py --no-symbolic --tag gors_no_sym

  # Ablation: no physics
  python experiments/train_gors.py --no-physics --tag gors_no_phys

  # Ablation: no closed-loop feedback
  python experiments/train_gors.py --no-feedback --tag gors_no_fb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.gors_config import gors_cfg
from experiments.gors_label import generate_gors_labels, GORSDataset
from experiments.models.neuro_symbolic_model import NeuroSymbolicDecisionModel
from experiments.losses.gors_symbolic_loss import GORSSymbolicLoss
from experiments.losses.gors_physics_loss import GORSPhysicsLoss
from src.data.multimodal_psml_dataset import MODALITY_DIMS, multimodal_psml_collate, load_psml_zone_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_gors")


# ---------------------------------------------------------------------------
# GORS Model Wrapper
# ---------------------------------------------------------------------------

class GORSModel(nn.Module):
    """
    Wraps NeuroSymbolicDecisionModel for GORS output (scalar risk score).
    The underlying SNN backbone is unchanged; only the output head is modified.
    """

    def __init__(self, model_cfg=None, seq_len: int = 96):
        super().__init__()
        if model_cfg is None:
            model_cfg = gors_cfg.model

        self.snn = NeuroSymbolicDecisionModel(
            modality_dims=MODALITY_DIMS,
            hidden_dim=model_cfg.hidden_dim,
            embedding_dim=model_cfg.embedding_dim,
            num_blocks=model_cfg.num_blocks,
            num_heads=model_cfg.num_heads,
            num_decision_intents=model_cfg.output_dim,  # 1 for GORS
            seq_len=seq_len,
            dropout=model_cfg.dropout,
            use_lif=model_cfg.use_lif_in_blocks,
            use_multi_comp=model_cfg.use_multi_comp,
            soma_neuron_type=model_cfg.soma_neuron_type,
        )

    def forward(self, modalities_data):
        output = self.snn(modalities_data)
        gors = torch.sigmoid(output["decision_logits"])  # [B, 1] ∈ [0,1]
        return {
            "gors": gors,
            "final_representation": output["final_representation"],
            "spike_rate": output.get("spike_rate", None),
            "modality_weights": output.get("modality_weights", None),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class GORSTrainer:
    def __init__(self, model, device, symbolic_loss_fn=None, physics_loss_fn=None,
                 use_feedback=True, lr=1e-3, weight_decay=1e-5, gradient_clip=1.0, seed=42):
        self.model = model.to(device)
        self.device = device
        self.symbolic_loss_fn = symbolic_loss_fn.to(device) if symbolic_loss_fn else None
        self.physics_loss_fn = physics_loss_fn.to(device) if physics_loss_fn else None
        self.use_feedback = use_feedback
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.gradient_clip = gradient_clip
        torch.manual_seed(seed)

    def _forward_with_feedback(self, modalities, prev_gors=None):
        """
        Forward pass with optional closed-loop feedback injection.

        If use_feedback and prev_gors is available:
          1. Forward → gors_pred
          2. Compute r_rule, r_phy
          3. r_t = r_rule + r_phy
          4. I_fb = W_f * r_t  →  back to soma (approximated via output correction)
          5. Corrected gors = gors_pred - η * r_t  (fast-loop correction)
        """
        output = self.model(modalities)
        gors = output["gors"]  # [B, 1]

        r_rule = torch.tensor(0.0, device=self.device)
        r_phy = torch.tensor(0.0, device=self.device)

        if self.use_feedback and prev_gors is not None:
            # Compute rule residual
            if self.symbolic_loss_fn is not None:
                _, r_rule, _ = self.symbolic_loss_fn(gors, modalities)

            # Compute physics residual
            if self.physics_loss_fn is not None:
                _, r_phy, _ = self.physics_loss_fn(gors, modalities, prev_gors.squeeze(-1))

            # Fast-loop: correct GORS toward consistency
            r_total = r_rule + r_phy
            gors = gors - 0.01 * r_total  # η_fast = 0.01

        output["gors"] = torch.clamp(gors, 0.0, 1.0)
        output["r_rule"] = r_rule
        output["r_phy"] = r_phy
        return output

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        metrics_acc = {"loss": 0.0, "mse": 0.0, "L_rule": 0.0, "L_phy": 0.0, "L_fb": 0.0}
        detail = {}
        n = 0
        t0 = time.time()

        for i, (modalities, targets) in enumerate(dataloader):
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            targets = targets.to(self.device).unsqueeze(-1)  # [B, 1]

            output = self._forward_with_feedback(modalities)
            gors = output["gors"]  # [B, 1]

            # Data loss: MSE between predicted GORS and pseudo-risk target
            L_data = self.mse_loss(gors, targets)
            loss = L_data
            metrics_acc["mse"] += L_data.item()

            # Symbolic loss
            if self.symbolic_loss_fn is not None:
                L_rule, r_rule, sym_m = self.symbolic_loss_fn(gors, modalities)
                loss = loss + L_rule
                metrics_acc["L_rule"] += L_rule.item()
                for k, v in sym_m.items():
                    detail[k] = detail.get(k, 0.0) + v

            # Physics loss
            if self.physics_loss_fn is not None:
                L_phy, r_phy, phys_m = self.physics_loss_fn(gors, modalities)
                loss = loss + L_phy
                metrics_acc["L_phy"] += L_phy.item()
                for k, v in phys_m.items():
                    detail[k] = detail.get(k, 0.0) + v

            # Feedback loss (penalize large residuals)
            if self.use_feedback:
                r_total = output.get("r_rule", torch.tensor(0.0)) + output.get("r_phy", torch.tensor(0.0))
                L_fb = gors_cfg.closed_loop.lambda_feedback * (r_total ** 2)
                loss = loss + L_fb
                metrics_acc["L_fb"] += L_fb.item()

            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            metrics_acc["loss"] += loss.item()
            n += 1

            if i == 0 or (i + 1) % 10 == 0:
                logger.info("  batch %3d/%d | loss=%.4f mse=%.4f sym=%.4f phy=%.4f | %.1fs",
                            i + 1, len(dataloader), loss.item(), L_data.item(),
                            metrics_acc["L_rule"] / max(n, 1), metrics_acc["L_phy"] / max(n, 1),
                            time.time() - t0)

        n = max(n, 1)
        return {k: v / n for k, v in metrics_acc.items()}, {k: v / n for k, v in detail.items()}

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        total_mse = 0.0
        total_mae = 0.0
        all_gors = []
        all_targets = []
        n = 0

        for modalities, targets in dataloader:
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            targets = targets.to(self.device).unsqueeze(-1)

            output = self.model(modalities)
            gors = output["gors"]

            total_mse += ((gors - targets) ** 2).sum().item()
            total_mae += (gors - targets).abs().sum().item()
            n += targets.numel()

            all_gors.append(gors.cpu())
            all_targets.append(targets.cpu())

        all_gors = torch.cat(all_gors).numpy()
        all_targets = torch.cat(all_targets).numpy()

        # Risk-level accuracy: correlation between predicted and target risk
        from scipy.stats import spearmanr
        try:
            rho, _ = spearmanr(all_gors.flatten(), all_targets.flatten())
        except Exception:
            rho = 0.0

        return {
            "mse": total_mse / max(n, 1),
            "rmse": np.sqrt(total_mse / max(n, 1)),
            "mae": total_mae / max(n, 1),
            "spearman_rho": float(rho),
            "gors_mean_pred": float(all_gors.mean()),
            "gors_mean_target": float(all_targets.mean()),
        }

    def fit(self, train_loader, val_loader, epochs, checkpoint_dir, output_dir, model_tag):
        checkpoint_dir = Path(checkpoint_dir)
        output_dir = Path(output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        history = {"train_loss": [], "train_mse": [], "train_L_rule": [], "train_L_phy": [],
                   "val_rmse": [], "val_mae": [], "val_rho": []}
        best_rmse = float("inf")
        best_state = None

        logger.info("Starting training: %d epochs, %d batches/epoch", epochs, len(train_loader))

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            metrics, detail = self.train_epoch(train_loader, epoch)
            val_m = self.evaluate(val_loader)
            self.scheduler.step()

            history["train_loss"].append(metrics["loss"])
            history["train_mse"].append(metrics["mse"])
            history["train_L_rule"].append(metrics["L_rule"])
            history["train_L_phy"].append(metrics["L_phy"])
            history["val_rmse"].append(val_m["rmse"])
            history["val_mae"].append(val_m["mae"])
            history["val_rho"].append(val_m["spearman_rho"])

            if val_m["rmse"] < best_rmse:
                best_rmse = val_m["rmse"]
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
                logger.info(
                    f"Epoch {epoch:3d}/{epochs} | "
                    f"Loss={metrics['loss']:.4f} MSE={metrics['mse']:.4f} "
                    f"Sym={metrics['L_rule']:.4f} Phy={metrics['L_phy']:.4f} | "
                    f"Val RMSE={val_m['rmse']:.4f} ρ={val_m['spearman_rho']:.3f} | "
                    f"{time.time()-t0:.0f}s"
                )

        # Save
        ckpt_path = checkpoint_dir / f"{model_tag}.pth"
        torch.save({"model_state_dict": self.model.state_dict(), "best_rmse": best_rmse}, ckpt_path)
        if best_state:
            best_path = checkpoint_dir / f"{model_tag}_best.pth"
            torch.save({"model_state_dict": best_state, "best_rmse": best_rmse}, best_path)
        with open(output_dir / f"history_{model_tag}.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info("Saved: %s (best RMSE=%.4f)", ckpt_path, best_rmse)
        return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train GORS framework")
    parser.add_argument("--data-root", default=gors_cfg.data.root)
    parser.add_argument("--zones-per-split", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default="gors_full")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    # Ablation flags
    parser.add_argument("--no-symbolic", action="store_true")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    parser.add_argument("--multi-comp", action="store_true", help="Multi-comp Izhikevich (legacy, unstable)")
    parser.add_argument("--multi-comp-lif", action="store_true", help="Multi-comp LIF (stable, dendrites leak-only)")
    parser.add_argument("--no-spike", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Device: %s", device)

    # ---- Data Loading ----
    zones = gors_cfg.data.train_zones
    if args.zones_per_split > 0:
        zones = zones[:args.zones_per_split]

    logger.info("Loading %d zones...", len(zones))
    zone_frames, _, _ = load_psml_zone_frames(
        args.data_root, zones, max_rows_per_zone=args.max_rows, normalize="zscore")

    all_windows = []
    for frames in zone_frames.values():
        n = len(frames)
        n_windows = max(0, (n - args.seq_len) // args.stride + 1)
        for i in range(n_windows):
            all_windows.append(frames[i * args.stride : i * args.stride + args.seq_len])
    all_data = np.stack(all_windows, axis=0)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_data))
    train_size = int(len(all_data) * (1 - args.val_ratio))
    train_data = all_data[indices[:train_size]]
    val_data = all_data[indices[train_size:]]

    # Generate GORS labels
    logger.info("Generating GORS labels...")
    train_gors = generate_gors_labels(train_data, args.seq_len)
    val_gors = generate_gors_labels(val_data, args.seq_len)
    logger.info("GORS distribution: train μ=%.3f σ=%.3f | val μ=%.3f σ=%.3f",
                train_gors.mean(), train_gors.std(), val_gors.mean(), val_gors.std())

    # Datasets
    train_ds = GORSDataset(train_data, train_gors, seq_len=args.seq_len)
    val_ds = GORSDataset(val_data, val_gors, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=multimodal_psml_collate, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=multimodal_psml_collate, num_workers=args.num_workers)

    # ---- Model ----
    model = GORSModel(model_cfg=gors_cfg.model, seq_len=args.seq_len)
    logger.info("Model: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    # ---- Losses ----
    sym_fn = None if args.no_symbolic else GORSSymbolicLoss(
        temperature=gors_cfg.symbolic.temperature,
        lambda_symbolic=gors_cfg.symbolic.lambda_symbolic,
    )
    phys_fn = None if args.no_physics else GORSPhysicsLoss(
        w_balance=gors_cfg.physics.w_balance,
        w_ramp=gors_cfg.physics.w_ramp,
        lambda_physics=gors_cfg.physics.lambda_physics,
    )

    logger.info("Symbolic: %s | Physics: %s | Feedback: %s | Arch: %s | Spike: %s",
                not args.no_symbolic, not args.no_physics, not args.no_feedback,
                "multi-LIF" if args.multi_comp_lif else ("multi-Izh" if args.multi_comp else "single-LIF"),
                not args.no_spike)

    # ---- Train ----
    # Override model config for architecture variants
    model.snn.use_multi_comp = args.multi_comp
    model.snn.use_multi_comp_lif = args.multi_comp_lif
    model.snn.use_spike = not args.no_spike
    if args.no_spike:
        model.snn.blocks = nn.ModuleList([
            m for m in model.snn.blocks if not hasattr(m, 'lif_weight')
        ])

    trainer = GORSTrainer(
        model, device, sym_fn, phys_fn,
        use_feedback=not args.no_feedback,
        lr=args.lr,
        weight_decay=gors_cfg.training.weight_decay,
        gradient_clip=gors_cfg.training.gradient_clip,
        seed=gors_cfg.training.seed,
    )

    history = trainer.fit(
        train_loader, val_loader, args.epochs,
        args.checkpoint_dir, args.output_dir, args.tag,
    )

    best_epoch = np.argmin(history["val_rmse"]) + 1
    logger.info("Done. Best Val RMSE=%.4f at epoch %d, ρ=%.4f",
                min(history["val_rmse"]), best_epoch,
                history["val_rho"][best_epoch - 1])


if __name__ == "__main__":
    main()
