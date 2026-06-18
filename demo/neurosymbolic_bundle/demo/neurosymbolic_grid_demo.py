#!/usr/bin/env python3
"""
Neuro-Symbolic Grid Control — Unified Joint Training (Real PSML Data)
======================================================================
SNN-MLP anomaly classification + neuro-symbolic control, jointly trained.

Branch (--branch):
  snn       Early-fusion SNN-MLP (11-d) + CE + proximity
  control   Neuro-symbolic control (KCL/Voltage/Flow/Complement penalties)
  both      Joint training: SNN + control (default)

Joint loss (both):
  L = L_ce + w_ctrl * L_penalty + w_smooth * L_smooth

Usage:
  python demo/neurosymbolic_grid_demo.py --branch both --epochs 80 --data-fraction 0.33
  python demo/plot_neurosymbolic_demo.py
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
import torch.nn.functional as F

_BUNDLE = Path(__file__).resolve().parent.parent
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from src.data.load_renewable_dataset import (
    LoadRenewableDataLoader, TimeSeriesDataset, multi_zone_windows,
)


def _default_zones(data_root: str) -> List[str]:
    loader = LoadRenewableDataLoader(data_root)
    return sorted(loader.zones.keys())[:3]


# ======================================================================
# Surrogate Gradient + LIF + SNN-MLP
# ======================================================================

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x); ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors; th = ctx.threshold
        return grad_output * torch.clamp(1.0 - (x - th).abs(), min=0.0), None


class LIFNeuron(nn.Module):
    def __init__(self, threshold=1.0, leak=0.9):
        super().__init__()
        self.threshold, self.leak = threshold, leak

    def forward(self, x):
        mem = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(x.shape[1]):
            mem = self.leak * mem + x[:, t, :]
            spk = SurrogateSpike.apply(mem, self.threshold)
            mem = mem * (1.0 - spk)
            spikes.append(spk.unsqueeze(1))
        return torch.cat(spikes, dim=1)


class ThreeLayerSNNMLP(nn.Module):
    def __init__(self, input_dim, h1, h2, h3, threshold=1.0, leak=0.9):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1); self.lif1 = LIFNeuron(threshold, leak)
        self.fc2 = nn.Linear(h1, h2);       self.lif2 = LIFNeuron(threshold, leak)
        self.fc3 = nn.Linear(h2, h3);       self.lif3 = LIFNeuron(threshold, leak)
        self.classifier = nn.Linear(h3, 2)

    def forward(self, x):
        z1 = self.lif1(self.fc1(x))
        z2 = self.lif2(self.fc2(z1))
        z3 = self.lif3(self.fc3(z2))
        return self.classifier(z3.mean(dim=1)), z3


class RepresentationLearningHead(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=64):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim))
        self.dynamic_center = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, encoded):
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        emb = F.normalize(self.projection(pooled), p=2, dim=1)
        c = F.normalize(self.dynamic_center, p=2, dim=0)
        return emb, F.cosine_similarity(emb, c.unsqueeze(0))


# ======================================================================
# ControlNet + LogicNeuron
# ======================================================================

class ControlNet(nn.Module):
    """3-layer MLP: 7-d grid state → 9-d control action."""

    def __init__(self, in_dim=7, hidden=64, out_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class LogicNeuron(nn.Module):
    def __init__(self, temperature=20.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, margin):
        return torch.sigmoid(self.temperature * margin)


# ======================================================================
# Config
# ======================================================================

@dataclass
class Config:
    batch_size: int = 64
    seq_len: int = 96
    num_features: int = 11
    hidden1: int = 128; hidden2: int = 64; hidden3: int = 32
    embedding_dim: int = 64
    epochs: int = 80
    lr: float = 1e-3
    w_ctrl: float = 0.35; w_smooth: float = 0.01
    lif_threshold: float = 1.0; lif_leak: float = 0.9
    seed: int = 42
    device: str = "auto"
    output_dir: str = "demo/ns_results"
    branch: str = "both"

    # Symbolic constraint params
    volatility_threshold: float = 0.12
    v_min: float = 0.97
    v_max: float = 1.03
    tie_line_limit: float = 0.38
    logic_temperature: float = 20.0
    anomaly_std_threshold: float = 2.5

    # Data params
    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones_snn: List[str] = None   # 2 zones for SNN classification
    zones_ctrl: List[str] = None  # 3 zones for control (3 buses)
    stride: int = 96
    max_windows_per_zone: int = 500
    data_fraction: float = 0.33

    def __post_init__(self):
        defaults = _default_zones(self.data_root)
        if self.zones_snn is None:
            self.zones_snn = defaults[:2]
        if self.zones_ctrl is None:
            self.zones_ctrl = defaults[:3]
        if len(self.zones_ctrl) < 3:
            self.zones_ctrl = (self.zones_ctrl * 3)[:3]
        if len(self.zones_snn) < 2:
            self.zones_snn = (self.zones_snn * 2)[:2]


# ======================================================================
# Symbolic Penalty
# ======================================================================

def build_symbolic_penalty(action, load, ren, ren_vol, logic, cfg: Config):
    gen = action[:, 0:3]
    storage = action[:, 3:6]
    v_cmd = action[:, 6:9]

    gen = 1.2 * torch.sigmoid(gen) + 0.1
    storage = 0.5 * torch.tanh(storage)
    v = 1.0 + 0.12 * torch.tanh(v_cmd)

    inj = gen + ren + storage - load
    kcl_margin = 0.03 - torch.abs(inj)
    kcl_truth = logic(kcl_margin)
    kcl_pen = torch.mean(F.relu(-kcl_margin) ** 2)

    v_pen = torch.mean(F.relu(cfg.v_min - v) ** 2 + F.relu(v - cfg.v_max) ** 2)
    v_truth = logic(torch.minimum(v - cfg.v_min, cfg.v_max - v))

    f12 = 1.6 * torch.abs(inj[:, 0] - inj[:, 1])
    f23 = 1.6 * torch.abs(inj[:, 1] - inj[:, 2])
    f13 = 1.4 * torch.abs(inj[:, 0] - inj[:, 2])
    tie_flow = torch.stack([f12, f23, f13], dim=1)
    flow_margin = cfg.tie_line_limit - tie_flow
    flow_truth = logic(flow_margin)
    flow_pen = torch.mean(F.relu(tie_flow - cfg.tie_line_limit) ** 2)

    vol_gate = torch.sigmoid(25.0 * (ren_vol - cfg.volatility_threshold))
    ren_c = ren - torch.mean(ren, dim=1, keepdim=True)
    expert = -0.6 * torch.tanh(ren_c)
    comp_gap = storage - expert
    comp_margin = 0.08 - torch.abs(comp_gap)
    comp_truth = logic(comp_margin)
    comp_pen = torch.mean(vol_gate * (comp_gap ** 2))

    corrected_s = (1.0 - vol_gate) * storage + vol_gate * expert
    corrected = torch.cat([gen, corrected_s, v], dim=1)

    total = 2.0 * kcl_pen + 1.5 * v_pen + 1.5 * flow_pen + 2.5 * comp_pen

    return total, corrected, {
        "kcl_penalty": kcl_pen.item(), "v_penalty": v_pen.item(),
        "flow_penalty": flow_pen.item(), "comp_penalty": comp_pen.item(),
    }, {
        "kcl_truth": torch.mean(kcl_truth).item(),
        "voltage_truth": torch.mean(v_truth).item(),
        "flow_truth": torch.mean(flow_truth).item(),
        "complement_truth": torch.mean(comp_truth).item(),
    }


# ======================================================================
# Unified Model
# ======================================================================

class UnifiedModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.branch = cfg.branch
        self.use_snn = cfg.branch in ("snn", "both")
        self.use_ctrl = cfg.branch in ("control", "both")

        if self.use_snn:
            self.snn = ThreeLayerSNNMLP(cfg.num_features, cfg.hidden1,
                                        cfg.hidden2, cfg.hidden3,
                                        cfg.lif_threshold, cfg.lif_leak)
            self.risk_head = RepresentationLearningHead(cfg.hidden3, cfg.embedding_dim)

        if self.use_ctrl:
            self.policy = ControlNet(in_dim=7, hidden=64, out_dim=9)

    def forward_snn(self, x):
        logits, z3 = self.snn(x)
        _, prox = self.risk_head(z3)
        return logits, prox


# ======================================================================
# Data Samplers
# ======================================================================

class SNNDataSampler:
    """Load 2 zones for SNN classification."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.windows: List[np.ndarray] = []
        for zone in cfg.zones_snn:
            loader = LoadRenewableDataLoader(cfg.data_root)
            df = loader.load_zone(zone)
            ds = TimeSeriesDataset(df, seq_len=cfg.seq_len,
                                   stride=cfg.stride, normalize="zscore")
            n_total = len(ds)
            n_use = min(n_total, cfg.max_windows_per_zone)
            if 0.0 < cfg.data_fraction < 1.0:
                n_use = min(n_use, max(1, int(n_total * cfg.data_fraction)))
            w = np.stack([ds[i][0] for i in range(n_use)], axis=0)
            self.windows.append(w)
        # Merge windows from both zones
        self.data = np.concatenate(self.windows, axis=0)  # (N, T, 11)
        rng = np.random.default_rng(cfg.seed)
        self.data = self.data[rng.permutation(len(self.data))]
        print(f"SNNDataSampler: {len(self.data)} windows from {cfg.zones_snn}")

    def sample(self, batch, seed, device):
        g = torch.Generator(device="cpu").manual_seed(seed)
        idx = torch.randint(0, len(self.data), (batch,), generator=g).numpy()
        x = torch.from_numpy(self.data[idx]).float().to(device)
        score = x.abs().amax(dim=1).amax(dim=1)
        y = (score > self.cfg.anomaly_std_threshold).long().to(device)
        return x, y


class ControlDataSampler:
    """Load 3 zones as 3-bus system for neuro-symbolic control."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.zone_data = multi_zone_windows(
            cfg.data_root, cfg.zones_ctrl,
            seq_len=cfg.seq_len, stride=cfg.stride, normalize="zscore",
            max_windows_per_zone=cfg.max_windows_per_zone,
            data_fraction=cfg.data_fraction,
        )
        self.n_windows = min(d.shape[0] for d in self.zone_data)
        print(f"ControlDataSampler: {len(cfg.zones_ctrl)} zones, "
              f"{self.n_windows} windows/zone")

    def sample(self, batch, seed, device):
        g = torch.Generator(device="cpu").manual_seed(seed)
        loads, rens = [], []
        for zi in range(3):
            data = self.zone_data[zi]
            idx = torch.randint(0, data.shape[0], (batch,), generator=g).numpy()
            w = data[idx]
            loads.append(w[:, :, 0].mean(axis=1))
            rens.append(w[:, :, 1:3].sum(axis=2).mean(axis=1))
        load = torch.from_numpy(np.stack(loads, axis=1)).float().to(device)
        ren = torch.from_numpy(np.stack(rens, axis=1)).float().to(device)
        rv = torch.mean(torch.abs(ren - torch.roll(ren, 1, 0)), dim=1, keepdim=True)
        if rv.shape[0] > 1:
            rv[0] = rv[1]
        gs = torch.cat([load, ren, rv], dim=1)
        return gs, load, ren, rv


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Grid Control — Unified Joint Training (Real PSML)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-ctrl", type=float, default=0.35)
    parser.add_argument("--branch", choices=["snn", "control", "both"],
                        default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="demo/ns_results")
    parser.add_argument("--data-root", type=str,
                        default="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable")
    parser.add_argument("--zones-snn", nargs="+", default=None,
                        help="2 zones for SNN classification")
    parser.add_argument("--zones-ctrl", nargs="+", default=None,
                        help="3 zones for control (3 buses)")
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        w_ctrl=args.w_ctrl, branch=args.branch,
        device=args.device, output_dir=args.output_dir,
        data_root=args.data_root,
        zones_snn=args.zones_snn, zones_ctrl=args.zones_ctrl,
        seq_len=args.seq_len, stride=args.stride,
        max_windows_per_zone=args.max_windows,
        data_fraction=args.data_fraction,
    )
    torch.manual_seed(cfg.seed)

    device = torch.device(
        "cuda" if cfg.device == "auto" and torch.cuda.is_available()
        else cfg.device if cfg.device != "auto" else "cpu")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    snn_sampler = None
    ctrl_sampler = None
    if cfg.branch in ("snn", "both"):
        print("Loading SNN classification data...")
        snn_sampler = SNNDataSampler(cfg)
    if cfg.branch in ("control", "both"):
        print("Loading control data...")
        ctrl_sampler = ControlDataSampler(cfg)

    # ---- Model ----
    model = UnifiedModel(cfg).to(device)
    logic = LogicNeuron(temperature=cfg.logic_temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict = {
        "branch": cfg.branch,
        "loss_total": [], "loss_ce": [], "mean_proximity": [],
        "kcl_penalty": [], "v_penalty": [], "flow_penalty": [],
        "comp_penalty": [],
        "kcl_truth": [], "voltage_truth": [], "flow_truth": [],
        "complement_truth": [],
    }

    print(f"=== Neuro-Symbolic Grid Control — Unified Joint Training ===")
    print(f"SNN zones={cfg.zones_snn}, Control zones={cfg.zones_ctrl}")
    print(f"Branch={cfg.branch} | SNN={cfg.branch in ('snn','both')} | "
          f"Control={cfg.branch in ('control','both')}")
    print(f"device={device}, epochs={cfg.epochs}, batch={cfg.batch_size}")

    for epoch in range(1, cfg.epochs + 1):
        seed = cfg.seed + epoch
        loss_ce = torch.tensor(0.0, device=device)
        total_pen = torch.tensor(0.0, device=device)
        prox_mean, acc = 0.0, 0.0

        # SNN branch
        if cfg.branch in ("snn", "both"):
            x, y = snn_sampler.sample(cfg.batch_size, seed, device)
            logits, prox = model.forward_snn(x)
            loss_ce = F.cross_entropy(logits, y)
            prox_mean = float(prox.mean().item())
            acc = (logits.argmax(1) == y).float().mean().item()

        # Control branch
        diag = {"kcl_penalty": 0.0, "v_penalty": 0.0,
                "flow_penalty": 0.0, "comp_penalty": 0.0}
        truth = {"kcl_truth": 0.0, "voltage_truth": 0.0,
                 "flow_truth": 0.0, "complement_truth": 0.0}
        smooth = torch.tensor(0.0, device=device)

        if cfg.branch in ("control", "both"):
            gs, load, ren, rv = ctrl_sampler.sample(cfg.batch_size, seed, device)
            raw = model.policy(gs)
            total_pen, corrected, diag, truth = build_symbolic_penalty(
                raw, load, ren, rv, logic, cfg)
            smooth = torch.mean(corrected ** 2)

        # Joint loss
        loss = cfg.w_smooth * smooth
        if cfg.branch in ("control", "both"):
            loss = loss + cfg.w_ctrl * total_pen
        if cfg.branch in ("snn", "both"):
            loss = loss + loss_ce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in [
            ("loss_total", loss.item()), ("loss_ce", loss_ce.item()),
            ("mean_proximity", prox_mean),
            ("kcl_penalty", diag["kcl_penalty"]),
            ("v_penalty", diag["v_penalty"]),
            ("flow_penalty", diag["flow_penalty"]),
            ("comp_penalty", diag["comp_penalty"]),
            ("kcl_truth", truth["kcl_truth"]),
            ("voltage_truth", truth["voltage_truth"]),
            ("flow_truth", truth["flow_truth"]),
            ("complement_truth", truth["complement_truth"]),
        ]:
            history.setdefault(k, []).append(v)

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            msg = (f"Epoch {epoch:03d}/{cfg.epochs} | loss={loss.item():.4f}")
            if cfg.branch in ("snn", "both"):
                msg += f" | ce={loss_ce.item():.4f} acc={acc:.3f} prox={prox_mean:.3f}"
            if cfg.branch in ("control", "both"):
                msg += (f" | pen(kcl/v/flow/comp)={diag['kcl_penalty']:.4f}/"
                        f"{diag['v_penalty']:.6f}/{diag['flow_penalty']:.4f}/"
                        f"{diag['comp_penalty']:.4f}")
            print(msg)

    hist_path = out / "ns_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {hist_path}")
    print("Plot: python demo/plot_neurosymbolic_demo.py")


if __name__ == "__main__":
    main()
