#!/usr/bin/env python3
"""
Small demo: 3-layer SNN-MLP for anomaly classification on real power data.

Model:
  input -> Linear1 + LIF -> Linear2 + LIF -> Linear3 + LIF -> classifier

Task:
  Binary classification on real power sequence windows:
  - class 0: normal window
  - class 1: anomaly-like window labeled by a lightweight rule
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset


@dataclass
class DemoConfig:
    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones: tuple[str, ...] = ("ERCOT_zone_1_", "MISO_zone_1_")
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
    output_dir: str = "demo/results"
    max_windows_per_zone: int = 300
    data_fraction: float = 1.0
    anomaly_std_threshold: float = 2.5


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        threshold = ctx.threshold
        # Triangular surrogate gradient around threshold.
        grad_x = grad_output * torch.clamp(1.0 - (x - threshold).abs(), min=0.0)
        return grad_x, None


class LIFNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0, leak: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, dim]
        mem = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(x.shape[1]):
            mem = self.leak * mem + x[:, t, :]
            spk = SurrogateSpike.apply(mem, self.threshold)
            mem = mem * (1.0 - spk)  # reset after spike
            spikes.append(spk.unsqueeze(1))
        return torch.cat(spikes, dim=1)


class ThreeLayerSNNMLP(nn.Module):
    def __init__(self, input_dim: int, h1: int, h2: int, h3: int, threshold: float, leak: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.lif1 = LIFNeuron(threshold, leak)
        self.fc2 = nn.Linear(h1, h2)
        self.lif2 = LIFNeuron(threshold, leak)
        self.fc3 = nn.Linear(h2, h3)
        self.lif3 = LIFNeuron(threshold, leak)
        self.classifier = nn.Linear(h3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, time, features]
        z1 = self.lif1(self.fc1(x))
        z2 = self.lif2(self.fc2(z1))
        z3 = self.lif3(self.fc3(z2))
        # Rate coding: average spikes across time
        rate = z3.mean(dim=1)
        return self.classifier(rate)


def label_window(window: torch.Tensor, threshold: float) -> int:
    # Mark a window as anomaly-like if any feature contains enough large deviations.
    score = window.abs().amax(dim=0)
    return int((score > threshold).any().item())


def load_real_power_data(cfg: DemoConfig) -> tuple[torch.Tensor, torch.Tensor]:
    loader = LoadRenewableDataLoader(cfg.data_root)
    frames: list[pd.DataFrame] = []

    for zone in cfg.zones:
        frames.append(loader.load_zone(zone))

    x_list = []
    y_list = []
    for zone, frame in zip(cfg.zones, frames):
        dataset = TimeSeriesDataset(frame, seq_len=cfg.seq_len, stride=cfg.stride, normalize="zscore")
        num_windows = min(len(dataset), cfg.max_windows_per_zone)
        if 0.0 < cfg.data_fraction < 1.0:
            num_windows = min(num_windows, max(1, int(len(dataset) * cfg.data_fraction)))
        for idx in range(num_windows):
            window_np, _ = dataset[idx]
            window = torch.from_numpy(window_np)
            label = label_window(window, cfg.anomaly_std_threshold)
            x_list.append(window)
            y_list.append(label)

    if not x_list:
        raise RuntimeError("No windows were created from the selected real-data zones.")

    x = torch.stack(x_list).float()
    y = torch.tensor(y_list, dtype=torch.long)
    return x, y


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            total_loss += loss.item() * xb.shape[0]
            pred = logits.argmax(dim=1)
            total_correct += (pred == yb).sum().item()
            total += xb.shape[0]
    return total_loss / total, total_correct / total


def main():
    parser = argparse.ArgumentParser(description="3-layer SNN MLP demo for anomaly classification")
    parser.add_argument("--epochs", type=int, default=DemoConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=DemoConfig.batch_size)
    parser.add_argument("--data-root", type=str, default=DemoConfig.data_root)
    parser.add_argument("--zones", nargs="+", default=list(DemoConfig.zones))
    parser.add_argument("--seq-len", type=int, default=DemoConfig.seq_len)
    parser.add_argument("--stride", type=int, default=DemoConfig.stride)
    parser.add_argument("--num-features", type=int, default=DemoConfig.num_features)
    parser.add_argument("--lr", type=float, default=DemoConfig.lr)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default=DemoConfig.output_dir)
    parser.add_argument("--max-windows-per-zone", type=int, default=DemoConfig.max_windows_per_zone)
    parser.add_argument("--data-fraction", type=float, default=DemoConfig.data_fraction,
        help="Use only this fraction of time windows per zone, e.g. 0.33 for one-third")
    parser.add_argument("--anomaly-std-threshold", type=float, default=DemoConfig.anomaly_std_threshold)
    args = parser.parse_args()

    cfg = DemoConfig(
        data_root=args.data_root,
        zones=tuple(args.zones),
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        stride=args.stride,
        num_features=args.num_features,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        max_windows_per_zone=args.max_windows_per_zone,
        data_fraction=args.data_fraction,
        anomaly_std_threshold=args.anomaly_std_threshold,
    )
    torch.manual_seed(cfg.seed)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    print("=== 3-layer SNN MLP Demo ===")
    print(f"device={device}, seq_len={cfg.seq_len}, stride={cfg.stride}, features={cfg.num_features}")
    print(f"data_root={cfg.data_root}")
    print(f"zones={list(cfg.zones)}")
    print(f"max_windows_per_zone={cfg.max_windows_per_zone}, anomaly_std_threshold={cfg.anomaly_std_threshold}")

    x, y = load_real_power_data(cfg)
    dataset = TensorDataset(x, y)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = ThreeLayerSNNMLP(
        input_dim=cfg.num_features,
        h1=cfg.hidden1,
        h2=cfg.hidden2,
        h3=cfg.hidden3,
        threshold=cfg.threshold,
        leak=cfg.leak,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.CrossEntropyLoss()

    print(f"dataset_windows={len(dataset)}, anomaly_ratio={(y.float().mean().item()):.4f}")
    print("Model: input -> [fc1+lif] -> [fc2+lif] -> [fc3+lif] -> classifier")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * xb.shape[0]
            total_correct += (logits.argmax(dim=1) == yb).sum().item()
            total += xb.shape[0]

        train_loss = total_loss / total
        train_acc = total_correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    history_path = output_dir / "history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved history: {history_path}")
    print("Done. This is a minimal SNN-MLP demo for the operator modeling storyline.")


if __name__ == "__main__":
    main()
