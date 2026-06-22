#!/usr/bin/env python3
"""
Real-data neuro-symbolic grid control demo.

Provides reusable classes/functions for the unified techdoc demo:
  ControlNet, LogicNeuron, NSConfig, build_symbolic_penalty, RealGridDataSampler

Can also be run standalone:
  python demo/neurosymbolic_grid_demo.py --epochs 80 --data-fraction 0.33
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

# Ensure bundle root on path so src imports work
_BUNDLE = Path(__file__).resolve().parent.parent
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from src.data.load_renewable_dataset import (
    LoadRenewableDataLoader, TimeSeriesDataset,
)


def _default_zones(data_root: str) -> List[str]:
    loader = LoadRenewableDataLoader(data_root)
    return sorted(loader.zones.keys())[:3]


@dataclass
class NSConfig:
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    hidden: int = 64
    seed: int = 42
    device: str = "auto"
    output_dir: str = "demo/ns_results"

    volatility_threshold: float = 0.12
    v_min: float = 0.97
    v_max: float = 1.03
    tie_line_limit: float = 0.38
    logic_temperature: float = 20.0

    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones: List[str] = None
    seq_len: int = 96
    stride: int = 96
    max_windows_per_zone: int = 500
    data_fraction: float = 0.33
    normalize: str = "zscore"

    def __post_init__(self):
        if self.zones is None:
            self.zones = _default_zones(self.data_root)[:3]
        if len(self.zones) < 3:
            self.zones = (self.zones * 3)[:3]


# ======================================================================
# Neural Controller + Logic Neuron
# ======================================================================

class ControlNet(nn.Module):
    """3-layer MLP policy: 7-d grid state -> 9-d control action."""

    def __init__(self, in_dim: int = 7, hidden: int = 64, out_dim: int = 9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LogicNeuron(nn.Module):
    """Map constraint margin -> [0,1] soft truth."""

    def __init__(self, temperature: float = 20.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, margin: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.temperature * margin)


# ======================================================================
# Real PSML Data Sampler
# ======================================================================

class RealGridDataSampler:
    """Pre-load 3 PSML zones as 3 buses, provide per-epoch batches."""

    def __init__(self, cfg: NSConfig):
        self.cfg = cfg
        loader = LoadRenewableDataLoader(cfg.data_root)
        self.zones = cfg.zones[:3]
        self.zone_data: List[np.ndarray] = []

        for zone in self.zones:
            df = loader.load_zone(zone)
            ds = TimeSeriesDataset(df, seq_len=cfg.seq_len,
                                   stride=cfg.stride, normalize=cfg.normalize)
            n_total = len(ds)
            n_use = min(n_total, cfg.max_windows_per_zone)
            if 0.0 < cfg.data_fraction < 1.0:
                n_use = min(n_use, max(1, int(n_total * cfg.data_fraction)))
            windows = [ds[i][0] for i in range(n_use)]
            self.zone_data.append(np.stack(windows, axis=0))

        self.n_windows = min(d.shape[0] for d in self.zone_data)
        print(f"RealGridDataSampler: {len(self.zones)} zones, "
              f"{self.n_windows} windows/zone, data_fraction={cfg.data_fraction}")

    def sample(self, batch: int, seed: int, device: torch.device):
        g = torch.Generator(device="cpu").manual_seed(seed)
        loads, rens = [], []
        for zi in range(3):
            data = self.zone_data[zi]
            idx = torch.randint(0, data.shape[0], (batch,), generator=g).numpy()
            w = data[idx]  # (batch, T, 11)
            loads.append(w[:, :, 0].mean(axis=1))
            rens.append(w[:, :, 1:3].sum(axis=2).mean(axis=1))

        load = torch.from_numpy(np.stack(loads, axis=1)).float().to(device)
        ren = torch.from_numpy(np.stack(rens, axis=1)).float().to(device)
        ren_vol = torch.mean(torch.abs(ren - torch.roll(ren, 1, 0)), dim=1, keepdim=True)
        if ren_vol.shape[0] > 1:
            ren_vol[0] = ren_vol[1]
        state = torch.cat([load, ren, ren_vol], dim=1)
        return state, load, ren, ren_vol


# ======================================================================
# Differentiable Symbolic Penalties
# ======================================================================

def build_symbolic_penalty(
    action: torch.Tensor,
    load: torch.Tensor,
    ren: torch.Tensor,
    ren_vol: torch.Tensor,
    logic: LogicNeuron,
    cfg: NSConfig,
):
    """Compute 4 differentiable constraint penalties.

    action: [g1,g2,g3 | s1,s2,s3 | v1,v2,v3]  (raw)
    """
    gen = action[:, 0:3]
    storage = action[:, 3:6]
    v_cmd = action[:, 6:9]

    gen = 1.2 * torch.sigmoid(gen) + 0.1
    storage = 0.5 * torch.tanh(storage)
    v = 1.0 + 0.12 * torch.tanh(v_cmd)

    # 1) KCL
    inj = gen + ren + storage - load
    kcl_margin = 0.03 - torch.abs(inj)
    kcl_truth = logic(kcl_margin)
    kcl_pen = torch.mean(F.relu(-kcl_margin) ** 2)

    # 2) Voltage
    v_pen = torch.mean(F.relu(cfg.v_min - v) ** 2 + F.relu(v - cfg.v_max) ** 2)
    v_truth = logic(torch.minimum(v - cfg.v_min, cfg.v_max - v))

    # 3) Tie-line flow
    f12 = 1.6 * torch.abs(inj[:, 0] - inj[:, 1])
    f23 = 1.6 * torch.abs(inj[:, 1] - inj[:, 2])
    f13 = 1.4 * torch.abs(inj[:, 0] - inj[:, 2])
    tie_flow = torch.stack([f12, f23, f13], dim=1)
    flow_margin = cfg.tie_line_limit - tie_flow
    flow_truth = logic(flow_margin)
    flow_pen = torch.mean(F.relu(tie_flow - cfg.tie_line_limit) ** 2)

    # 4) Storage-renewable complement
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
        "mean_tie_flow": torch.mean(tie_flow).item(),
        "mean_voltage": torch.mean(v).item(),
        "mean_vol_gate": torch.mean(vol_gate).item(),
        "mean_load": torch.mean(load).item(), "mean_ren": torch.mean(ren).item(),
    }, {
        "kcl_truth": torch.mean(kcl_truth).item(),
        "voltage_truth": torch.mean(v_truth).item(),
        "flow_truth": torch.mean(flow_truth).item(),
        "complement_truth": torch.mean(comp_truth).item(),
    }


# ======================================================================
# Standalone main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Neuro-Symbolic Grid Control Demo (Real PSML Data)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="demo/ns_results")
    parser.add_argument("--data-root", type=str,
                        default="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable")
    parser.add_argument("--zones", nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    args = parser.parse_args()

    cfg = NSConfig(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        device=args.device, output_dir=args.output_dir,
        data_root=args.data_root,
        zones=args.zones if args.zones else _default_zones(args.data_root)[:3],
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

    print("Loading real PSML data...")
    sampler = RealGridDataSampler(cfg)

    model = ControlNet(in_dim=7, hidden=cfg.hidden, out_dim=9).to(device)
    logic = LogicNeuron(temperature=cfg.logic_temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "total_penalty": [], "kcl_penalty": [], "v_penalty": [],
        "flow_penalty": [], "comp_penalty": [],
        "kcl_truth": [], "voltage_truth": [], "flow_truth": [],
        "complement_truth": [], "mean_load": [], "mean_ren": [],
    }

    print(f"=== Neuro-Symbolic Grid Control (Real PSML, {cfg.zones}) ===")
    print(f"Windows/zone={sampler.n_windows}, data_fraction={cfg.data_fraction}")

    for epoch in range(1, cfg.epochs + 1):
        state, load, ren, ren_vol = sampler.sample(
            cfg.batch_size, cfg.seed + epoch, device)
        raw = model(state)
        total_pen, corrected, diag, truth = build_symbolic_penalty(
            raw, load, ren, ren_vol, logic, cfg)
        loss = total_pen + 0.01 * torch.mean(corrected ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["total_penalty"].append(loss.item())
        history["kcl_penalty"].append(diag["kcl_penalty"])
        history["v_penalty"].append(diag["v_penalty"])
        history["flow_penalty"].append(diag["flow_penalty"])
        history["comp_penalty"].append(diag["comp_penalty"])
        history["kcl_truth"].append(truth["kcl_truth"])
        history["voltage_truth"].append(truth["voltage_truth"])
        history["flow_truth"].append(truth["flow_truth"])
        history["complement_truth"].append(truth["complement_truth"])
        history["mean_load"].append(diag["mean_load"])
        history["mean_ren"].append(diag["mean_ren"])

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(f"Epoch {epoch:03d}/{cfg.epochs} | total={loss.item():.4f} | "
                  f"kcl={diag['kcl_penalty']:.4f} v={diag['v_penalty']:.6f} "
                  f"flow={diag['flow_penalty']:.4f} comp={diag['comp_penalty']:.4f} | "
                  f"truth(kcl/v/flow/comp)=({truth['kcl_truth']:.3f}/"
                  f"{truth['voltage_truth']:.3f}/{truth['flow_truth']:.3f}/"
                  f"{truth['complement_truth']:.3f})")

    hist_path = out / "ns_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {hist_path}")
    print("Done.")


if __name__ == "__main__":
    main()
