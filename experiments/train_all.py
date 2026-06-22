#!/usr/bin/env python3
"""
Unified training script for all GORS paper experiments.
Supports baselines (LSTM, Transformer, TCN, SNN-LIF) and GORS variants.

Usage:
  # Baselines
  CUDA_VISIBLE_DEVICES=0 python experiments/train_all.py --model lstm       --epochs 50
  CUDA_VISIBLE_DEVICES=1 python experiments/train_all.py --model transformer --epochs 50
  CUDA_VISIBLE_DEVICES=2 python experiments/train_all.py --model tcn         --epochs 50
  CUDA_VISIBLE_DEVICES=3 python experiments/train_all.py --model snn_lif     --epochs 50

  # GORS full + ablations
  CUDA_VISIBLE_DEVICES=4 python experiments/train_all.py --model gors --tag gors_full
  CUDA_VISIBLE_DEVICES=5 python experiments/train_all.py --model gors --no-symbolic --tag gors_no_sym
  CUDA_VISIBLE_DEVICES=6 python experiments/train_all.py --model gors --no-physics --tag gors_no_phy
  CUDA_VISIBLE_DEVICES=7 python experiments/train_all.py --model gors --no-feedback --tag gors_no_fb
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import math
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
logger = logging.getLogger("train_all")


# ===========================================================================
# Baseline Models
# ===========================================================================

class LSTMBaseline(nn.Module):
    """2-layer LSTM for GORS regression."""
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, modalities_data):
        # modalities_data has keys: load, renewable, irradiance, weather
        # Each [B, seq, n_feat] → concat along last dim → [B, seq, 11]
        x = torch.cat([modalities_data[k] for k in ["load", "renewable", "irradiance", "weather"]], dim=-1)
        out, _ = self.lstm(x)
        gors = torch.sigmoid(self.head(out[:, -1, :]))
        return {"gors": gors, "final_representation": out[:, -1, :]}


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1), :])


class TransformerBaseline(nn.Module):
    """4-layer Transformer encoder for GORS regression."""
    def __init__(self, input_dim=11, d_model=128, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            batch_first=True, dim_feedforward=512,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, modalities_data):
        x = torch.cat([modalities_data[k] for k in ["load", "renewable", "irradiance", "weather"]], dim=-1)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
        out = self.encoder(x, mask=mask, is_causal=True)
        gors = torch.sigmoid(self.head(out[:, -1, :]))
        return {"gors": gors, "final_representation": out[:, -1, :]}


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.1):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.conv1(x))
        out = self.dropout(out)
        out = self.relu(self.conv2(out))
        out = self.dropout(out)
        out = out[:, :, :residual.size(2)]  # trim to match
        return self.relu(out + residual)


class TCNBaseline(nn.Module):
    """4-layer TCN for GORS regression."""
    def __init__(self, input_dim=11, hidden_dim=128, num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)
        channels = [hidden_dim] * num_layers
        dilations = [2 ** i for i in range(num_layers)]
        self.blocks = nn.ModuleList([
            TCNBlock(channels[i], channels[min(i+1, num_layers-1)], 3, dilations[i], dropout)
            for i in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, modalities_data):
        x = torch.cat([modalities_data[k] for k in ["load", "renewable", "irradiance", "weather"]], dim=-1)
        x = x.transpose(1, 2)  # [B, 11, seq]
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        out = x.mean(dim=-1)  # [B, hidden_dim]
        gors = torch.sigmoid(self.head(out))
        return {"gors": gors, "final_representation": out}


class SNNLIFBaseline(nn.Module):
    """Single-compartment LIF SNN baseline for GORS regression."""
    def __init__(self, input_dim=11, hidden_dim=128, num_blocks=3, seq_len=96,
                 beta=0.9, threshold=1.0):
        super().__init__()
        self.seq_len = seq_len
        self.threshold = threshold
        self.beta = beta
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            ) for _ in range(num_blocks)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, modalities_data):
        x = torch.cat([modalities_data[k] for k in ["load", "renewable", "irradiance", "weather"]], dim=-1)
        B, T, _ = x.shape
        h = self.input_proj(x)  # [B, T, hidden_dim]
        v = torch.zeros(B, h.size(-1), device=x.device)
        spikes_all = []

        for t in range(T):
            inp = h[:, t, :]  # [B, hidden_dim]
            v = self.beta * v + inp
            spike = (v >= self.threshold).float()
            v = v * (1 - spike)  # reset
            spikes_all.append(spike.unsqueeze(1))

        spikes = torch.cat(spikes_all, dim=1)  # [B, T, hidden_dim]
        # Pass through blocks
        out = spikes
        for block in self.blocks:
            residual = out
            out = block(out)
            out = out + residual[:, :out.size(1), :out.size(-1)]
        gors = torch.sigmoid(self.head(out.mean(dim=1)))
        return {
            "gors": gors,
            "final_representation": out.mean(dim=1),
            "spike_rate": spikes.mean(),
        }


# ===========================================================================
# GORS Model (from train_gors.py)
# ===========================================================================

class GORSModel(nn.Module):
    """Wraps NeuroSymbolicDecisionModel for GORS output."""

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
            num_decision_intents=1,
            seq_len=seq_len,
            dropout=model_cfg.dropout,
            use_lif=model_cfg.use_lif_in_blocks,
            use_multi_comp=model_cfg.use_multi_comp,
            soma_neuron_type=model_cfg.soma_neuron_type,
        )

    def forward(self, modalities_data):
        output = self.snn(modalities_data)
        gors = torch.sigmoid(output["decision_logits"])
        return {
            "gors": gors,
            "final_representation": output["final_representation"],
            "spike_rate": output.get("spike_rate", None),
            "modality_weights": output.get("modality_weights", None),
        }


# ===========================================================================
# Trainer
# ===========================================================================

class Trainer:
    def __init__(self, model, device, symbolic_loss_fn=None, physics_loss_fn=None,
                 use_feedback=True, lr=1e-3, weight_decay=1e-5, gradient_clip=1.0,
                 seed=42, model_type="gors"):
        self.model = model.to(device)
        self.device = device
        self.symbolic_loss_fn = symbolic_loss_fn.to(device) if symbolic_loss_fn else None
        self.physics_loss_fn = physics_loss_fn.to(device) if physics_loss_fn else None
        self.use_feedback = use_feedback
        self.model_type = model_type
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.gradient_clip = gradient_clip
        torch.manual_seed(seed)

    def _forward_with_feedback(self, modalities, prev_gors=None):
        output = self.model(modalities)
        gors = output["gors"]
        r_rule = torch.tensor(0.0, device=self.device)
        r_phy = torch.tensor(0.0, device=self.device)

        if self.use_feedback and prev_gors is not None and self.model_type == "gors":
            if self.symbolic_loss_fn is not None:
                _, r_rule, _ = self.symbolic_loss_fn(gors, modalities)
            if self.physics_loss_fn is not None:
                _, r_phy, _ = self.physics_loss_fn(gors, modalities, prev_gors.squeeze(-1))
            gors = gors - 0.01 * (r_rule + r_phy)

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
            targets = targets.to(self.device).unsqueeze(-1)

            output = self._forward_with_feedback(modalities)
            gors = output["gors"]

            L_data = self.mse_loss(gors, targets)
            loss = L_data
            metrics_acc["mse"] += L_data.item()

            if self.symbolic_loss_fn is not None:
                L_rule, r_rule, sym_m = self.symbolic_loss_fn(gors, modalities)
                loss = loss + L_rule
                metrics_acc["L_rule"] += L_rule.item()
                for k, v in sym_m.items():
                    detail[k] = detail.get(k, 0.0) + v

            if self.physics_loss_fn is not None:
                L_phy, r_phy, phys_m = self.physics_loss_fn(gors, modalities)
                loss = loss + L_phy
                metrics_acc["L_phy"] += L_phy.item()
                for k, v in phys_m.items():
                    detail[k] = detail.get(k, 0.0) + v

            if self.use_feedback and self.model_type == "gors":
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

            if i == 0 or (i + 1) % 50 == 0:
                logger.info("  batch %4d/%d | loss=%.4f mse=%.4f | %.1fs",
                            i + 1, len(dataloader), loss.item(), L_data.item(),
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

        logger.info("Training: %d epochs, %d batches/epoch", epochs, len(train_loader))

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
                    f"Epoch {epoch:3d}/{epochs} | Loss={metrics['loss']:.4f} MSE={metrics['mse']:.4f} "
                    f"Val RMSE={val_m['rmse']:.4f} rho={val_m['spearman_rho']:.3f} | {time.time()-t0:.0f}s"
                )

        ckpt_path = checkpoint_dir / f"{model_tag}.pth"
        torch.save({"model_state_dict": self.model.state_dict(), "best_rmse": best_rmse}, ckpt_path)
        if best_state:
            torch.save({"model_state_dict": best_state, "best_rmse": best_rmse},
                       checkpoint_dir / f"{model_tag}_best.pth")
        with open(output_dir / f"history_{model_tag}.json", "w") as f:
            json.dump(history, f, indent=2)

        best_epoch = np.argmin(history["val_rmse"]) + 1
        logger.info("Done. Best Val RMSE=%.4f (epoch %d), rho=%.4f",
                    min(history["val_rmse"]), best_epoch,
                    history["val_rho"][best_epoch - 1])
        return history


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Unified GORS Paper Experiment Runner")
    parser.add_argument("--model", required=True,
                        choices=["lstm", "transformer", "tcn", "snn_lif", "gors"])
    parser.add_argument("--data-root", default=gors_cfg.data.root)
    parser.add_argument("--zones-per-split", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    # GORS ablation flags
    parser.add_argument("--no-symbolic", action="store_true")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-feedback", action="store_true")
    args = parser.parse_args()

    # Auto-tag
    if args.tag is None:
        tag_parts = [args.model]
        if args.model == "gors":
            if args.no_symbolic: tag_parts.append("no_sym")
            elif args.no_physics: tag_parts.append("no_phy")
            elif args.no_feedback: tag_parts.append("no_fb")
            else: tag_parts.append("full")
        args.tag = "_".join(tag_parts)

    device = torch.device(args.device)
    logger.info("=" * 60)
    logger.info("Model: %s | Tag: %s | Device: %s | Epochs: %d",
                args.model, args.tag, device, args.epochs)
    logger.info("=" * 60)

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
    logger.info("Total windows: %d", len(all_data))

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_data))
    train_size = int(len(all_data) * (1 - args.val_ratio))
    train_data = all_data[indices[:train_size]]
    val_data = all_data[indices[train_size:]]

    logger.info("Generating GORS labels...")
    train_gors = generate_gors_labels(train_data, args.seq_len)
    val_gors = generate_gors_labels(val_data, args.seq_len)
    logger.info("GORS distribution: train mu=%.3f sigma=%.3f | val mu=%.3f sigma=%.3f",
                train_gors.mean(), train_gors.std(), val_gors.mean(), val_gors.std())

    train_ds = GORSDataset(train_data, train_gors, seq_len=args.seq_len)
    val_ds = GORSDataset(val_data, val_gors, seq_len=args.seq_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=multimodal_psml_collate, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=multimodal_psml_collate, num_workers=args.num_workers)

    # ---- Model ----
    if args.model == "lstm":
        model = LSTMBaseline(hidden_dim=128)
    elif args.model == "transformer":
        model = TransformerBaseline(d_model=128)
    elif args.model == "tcn":
        model = TCNBaseline(hidden_dim=128)
    elif args.model == "snn_lif":
        model = SNNLIFBaseline(hidden_dim=128, seq_len=args.seq_len)
    elif args.model == "gors":
        model = GORSModel(model_cfg=gors_cfg.model, seq_len=args.seq_len)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model: %.2fM params", n_params)

    # ---- Losses (GORS only) ----
    sym_fn = None
    phys_fn = None
    if args.model == "gors":
        if not args.no_symbolic:
            sym_fn = GORSSymbolicLoss(
                temperature=gors_cfg.symbolic.temperature,
                lambda_symbolic=gors_cfg.symbolic.lambda_symbolic,
            )
        if not args.no_physics:
            phys_fn = GORSPhysicsLoss(
                w_balance=gors_cfg.physics.w_balance,
                w_ramp=gors_cfg.physics.w_ramp,
                lambda_physics=gors_cfg.physics.lambda_physics,
            )
        logger.info("Symbolic: %s | Physics: %s | Feedback: %s",
                    sym_fn is not None, phys_fn is not None, not args.no_feedback)

    # ---- Train ----
    trainer = Trainer(
        model, device, sym_fn, phys_fn,
        use_feedback=(not args.no_feedback) and args.model == "gors",
        lr=args.lr,
        weight_decay=gors_cfg.training.weight_decay,
        gradient_clip=gors_cfg.training.gradient_clip,
        seed=gors_cfg.training.seed,
        model_type=args.model,
    )

    history = trainer.fit(
        train_loader, val_loader, args.epochs,
        args.checkpoint_dir, args.output_dir, args.tag,
    )

    best_epoch = np.argmin(history["val_rmse"]) + 1
    logger.info("=" * 60)
    logger.info("FINAL: %s | Best RMSE=%.4f (ep %d) | rho=%.4f | Params=%.2fM",
                args.tag, min(history["val_rmse"]), best_epoch,
                history["val_rho"][best_epoch - 1], n_params)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
