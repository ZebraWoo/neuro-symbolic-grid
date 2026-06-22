#!/usr/bin/env python3
"""
3-Layer SNN-MLP Anomaly Classification Demo — Real PSML Data
=============================================================
LIF spiking neurons + surrogate gradient, trained on real
PSML minute-level power-grid data for binary anomaly detection.

Model: input → fc1+LIF → fc2+LIF → fc3+LIF → rate coding → classifier(2)

Labels (rule-based): a window is "anomaly-like" if any feature
exceeds `anomaly_std_threshold` standard deviations.

Usage:
  python demo/snn_mlp_demo.py --epochs 12 --data-fraction 0.33
  python demo/plot_snn_mlp_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

_BUNDLE = Path(__file__).resolve().parent.parent
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from src.data.load_renewable_dataset import (
    LoadRenewableDataLoader, TimeSeriesDataset,
)


# ======================================================================
# Config
# ======================================================================

def _default_zones(data_root: str) -> List[str]:
    loader = LoadRenewableDataLoader(data_root)
    return sorted(loader.zones.keys())[:2]


@dataclass
class Config:
    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones: Tuple[str, ...] = ("ERCOT_zone_1_", "MISO_zone_1_")
    seq_len: int = 96
    stride: int = 96
    num_features: int = 11
    batch_size: int = 64
    hidden1: int = 128
    hidden2: int = 64
    hidden3: int = 32
    epochs: int = 12
    lr: float = 1e-3
    threshold: float = 1.0
    leak: float = 0.9
    seed: int = 42
    device: str = "auto"
    output_dir: str = "demo/snn_results"
    max_windows_per_zone: int = 500
    data_fraction: float = 1.0
    anomaly_std_threshold: float = 2.5

    def __post_init__(self):
        if isinstance(self.zones, list):
            self.zones = tuple(self.zones)
        if not self.zones:
            self.zones = tuple(_default_zones(self.data_root)[:2])


# ======================================================================
# Surrogate Gradient + LIF Neuron
# ======================================================================

class SurrogateSpike(torch.autograd.Function):
    """Heaviside forward + triangular surrogate gradient."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        th = ctx.threshold
        return grad_output * torch.clamp(1.0 - (x - th).abs(), min=0.0), None


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron (iterates over time)."""

    def __init__(self, threshold: float = 1.0, leak: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        mem = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(x.shape[1]):
            mem = self.leak * mem + x[:, t, :]
            spk = SurrogateSpike.apply(mem, self.threshold)
            mem = mem * (1.0 - spk)
            spikes.append(spk.unsqueeze(1))
        return torch.cat(spikes, dim=1)


# ======================================================================
# 3-Layer SNN-MLP Model
# ======================================================================

class ThreeLayerSNNMLP(nn.Module):
    """input → fc1+LIF → fc2+LIF → fc3+LIF → rate coding → classifier."""

    def __init__(self, input_dim: int, h1: int, h2: int, h3: int,
                 threshold: float, leak: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.lif1 = LIFNeuron(threshold, leak)
        self.fc2 = nn.Linear(h1, h2)
        self.lif2 = LIFNeuron(threshold, leak)
        self.fc3 = nn.Linear(h2, h3)
        self.lif3 = LIFNeuron(threshold, leak)
        self.classifier = nn.Linear(h3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.lif1(self.fc1(x))
        z2 = self.lif2(self.fc2(z1))
        z3 = self.lif3(self.fc3(z2))
        rate = z3.mean(dim=1)  # rate coding
        return self.classifier(rate)


# ======================================================================
# Rule-based labeling
# ======================================================================

def label_window(window: torch.Tensor, threshold: float) -> int:
    """Mark as anomaly if any feature contains large enough deviations."""
    score = window.abs().amax(dim=0)
    return int((score > threshold).any().item())


# ======================================================================
# Real PSML data loading
# ======================================================================

def load_real_data(cfg: Config) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = LoadRenewableDataLoader(cfg.data_root)
    x_list, y_list = [], []

    for zone in cfg.zones:
        df = loader.load_zone(zone)
        dataset = TimeSeriesDataset(df, seq_len=cfg.seq_len,
                                    stride=cfg.stride, normalize="zscore")
        n_use = min(len(dataset), cfg.max_windows_per_zone)
        if 0.0 < cfg.data_fraction < 1.0:
            n_use = min(n_use, max(1, int(len(dataset) * cfg.data_fraction)))

        for idx in range(n_use):
            w_np, _ = dataset[idx]
            w = torch.from_numpy(w_np)
            x_list.append(w)
            y_list.append(label_window(w, cfg.anomaly_std_threshold))

    if not x_list:
        raise RuntimeError("No windows created from real-data zones.")
    return torch.stack(x_list).float(), torch.tensor(y_list, dtype=torch.long)


# ======================================================================
# Evaluation
# ======================================================================

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.shape[0]
            total_correct += (logits.argmax(1) == yb).sum().item()
            total += xb.shape[0]
    return total_loss / total, total_correct / total


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="3-layer SNN-MLP Anomaly Classification (Real PSML Data)")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="demo/snn_results")
    parser.add_argument("--data-root", type=str,
                        default="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable")
    parser.add_argument("--zones", nargs="+",
                        default=["ERCOT_zone_1_", "MISO_zone_1_"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-windows", type=int, default=500,
                        dest="max_windows_per_zone")
    parser.add_argument("--data-fraction", type=float, default=1.0)
    parser.add_argument("--anomaly-std-threshold", type=float, default=2.5,
                        dest="anomaly_std_threshold")
    args = parser.parse_args()

    cfg = Config(
        data_root=args.data_root,
        zones=tuple(args.zones),
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, output_dir=args.output_dir,
        seq_len=args.seq_len, stride=args.stride,
        max_windows_per_zone=args.max_windows_per_zone,
        data_fraction=args.data_fraction,
        anomaly_std_threshold=args.anomaly_std_threshold,
    )
    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else cfg.device if cfg.device != "auto" else "cpu")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== 3-layer SNN-MLP Anomaly Classification (Real PSML) ===")
    print(f"device={device}, seq_len={cfg.seq_len}, zones={list(cfg.zones)}")
    print(f"max_windows_per_zone={cfg.max_windows_per_zone}, "
          f"data_fraction={cfg.data_fraction}")

    x, y = load_real_data(cfg)
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = ThreeLayerSNNMLP(
        input_dim=cfg.num_features, h1=cfg.hidden1, h2=cfg.hidden2,
        h3=cfg.hidden3, threshold=cfg.threshold, leak=cfg.leak,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"dataset_windows={len(dataset)}, "
          f"anomaly_ratio={(y.float().mean().item()):.4f}")
    print("Model: input → [fc1+lif] → [fc2+lif] → [fc3+lif] → classifier")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    history = {"train_loss": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss, total_correct, total = 0.0, 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.shape[0]
            total_correct += (logits.argmax(1) == yb).sum().item()
            total += xb.shape[0]

        train_loss = total_loss / total
        train_acc = total_correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    hist_path = out / "snn_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {hist_path}")
    print("Plot: python demo/plot_snn_mlp_demo.py")


if __name__ == "__main__":
    main()
