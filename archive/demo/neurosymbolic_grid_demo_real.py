#!/usr/bin/env python3
"""
Neuro-symbolic power-grid control demo — REAL PSML DATA VERSION.

Uses real PSML minute-level load/renewable data from 3 zones as 3 buses.
The neuro-symbolic penalty logic (KCL, voltage, tie-line, storage-renewable)
remains identical to the synthetic version.

Usage:
  python demo/neurosymbolic_grid_demo_real.py --epochs 80 --data-fraction 0.33
  python demo/plot_neurosymbolic_demo.py --history-file demo/ns_results/ns_history.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset


def _discover_default_zones(data_root: str) -> List[str]:
    """Pick the first 3 available zones as default."""
    loader = LoadRenewableDataLoader(data_root)
    return sorted(loader.zones.keys())[:3]


@dataclass
class NSConfig:
    steps: int = 300
    epochs: int = 80
    lr: float = 1e-3
    hidden: int = 64
    batch_size: int = 32
    seed: int = 42
    device: str = "auto"
    output_dir: str = "demo/ns_results"
    volatility_threshold: float = 0.12
    v_min: float = 0.97
    v_max: float = 1.03
    tie_line_limit: float = 0.38
    logic_temperature: float = 20.0

    # Real-data parameters
    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones: List[str] = field(default_factory=list)
    seq_len: int = 96
    stride: int = 96
    max_windows_per_zone: int = 500
    data_fraction: float = 0.33
    normalize: str = "zscore"

    def __post_init__(self):
        if not self.zones:
            self.zones = _discover_default_zones(self.data_root)[:3]
        if len(self.zones) < 3:
            # Pad with same zones if fewer than 3
            self.zones = (self.zones * 3)[:3]


# ---------------------------------------------------------------------------
# Neural controller + logic neuron (identical to synthetic version)
# ---------------------------------------------------------------------------

class ControlNet(nn.Module):
    """Neural policy that outputs control actions."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LogicNeuron(nn.Module):
    """
    Map continuous margins to soft boolean truth values.
    truth ~= 1 means "constraint satisfied", truth ~= 0 means violation.
    """

    def __init__(self, temperature: float = 20.0):
        super().__init__()
        self.temperature = temperature

    def forward(self, margin: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.temperature * margin)


# ---------------------------------------------------------------------------
# Real PSML data loading (replaces synthesize_grid_batch)
# ---------------------------------------------------------------------------

class RealGridDataSampler:
    """Pre-loads 3 PSML zones and provides per-epoch batches for the 3-bus model."""

    def __init__(self, cfg: NSConfig):
        self.cfg = cfg
        self.loader = LoadRenewableDataLoader(cfg.data_root)
        self.zones = cfg.zones[:3]

        # Load and normalize each zone
        self.zone_data: List[np.ndarray] = []  # list of (n_windows, seq_len, n_feats)
        self.n_windows_per_zone = 0

        for zone in self.zones:
            df = self.loader.load_zone(zone)
            dataset = TimeSeriesDataset(
                df, seq_len=cfg.seq_len, stride=cfg.stride, normalize=cfg.normalize,
            )
            n_total = len(dataset)
            n_use = min(n_total, cfg.max_windows_per_zone)
            if 0.0 < cfg.data_fraction < 1.0:
                n_use = min(n_use, max(1, int(n_total * cfg.data_fraction)))

            windows = []
            for i in range(n_use):
                w, _ = dataset[i]
                windows.append(w)
            data = np.stack(windows, axis=0)  # (n_use, seq_len, 11)
            self.zone_data.append(data)

        self.n_windows_per_zone = min(d.shape[0] for d in self.zone_data)
        print(f"RealGridDataSampler: {len(self.zones)} zones, "
              f"{self.n_windows_per_zone} windows/zone, "
              f"seq_len={cfg.seq_len}, data_fraction={cfg.data_fraction}")

    def sample_batch(
        self, batch: int, seed: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: state (B,7), load (B,3), ren (B,3), ren_vol (B,1)
        Maps:
          - zone i load  = mean(load_power column) across the sampled window
          - zone i ren   = mean(wind_power + solar_power) across the window
          - ren_vol      = mean absolute diff of ren across the batch
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = self.n_windows_per_zone

        loads = []
        rens = []
        for zi in range(3):
            data = self.zone_data[zi]  # (N, seq_len, 11)
            idx = torch.randint(0, data.shape[0], (batch,), generator=g).numpy()
            windows = data[idx]  # (batch, seq_len, 11)

            # load_power = column 0, wind_power=1, solar_power=2
            zone_load = windows[:, :, 0].mean(axis=1)   # (batch,)
            zone_ren = windows[:, :, 1:3].sum(axis=2).mean(axis=1)  # (batch,)
            loads.append(zone_load)
            rens.append(zone_ren)

        load = torch.from_numpy(np.stack(loads, axis=1)).float().to(device)  # (B, 3)
        ren = torch.from_numpy(np.stack(rens, axis=1)).float().to(device)    # (B, 3)

        # Volatility: |diff of ren| across buses
        ren_roll = torch.roll(ren, shifts=1, dims=0)
        ren_vol = torch.mean(torch.abs(ren - ren_roll), dim=1, keepdim=True)  # (B, 1)
        ren_vol[0] = ren_vol[1] if ren_vol.shape[0] > 1 else ren_vol[0]

        # State vector: concat load + ren + ren_vol
        state = torch.cat([load, ren, ren_vol], dim=1)  # (B, 7)
        return state, load, ren, ren_vol


# ---------------------------------------------------------------------------
# Symbolic penalty (identical to synthetic version)
# ---------------------------------------------------------------------------

def build_symbolic_penalty(
    action: torch.Tensor,
    load: torch.Tensor,
    ren: torch.Tensor,
    ren_vol: torch.Tensor,
    logic: LogicNeuron,
    cfg: NSConfig,
):
    """
    action layout:
    [g1,g2,g3,  storage1,storage2,storage3,  v_cmd1,v_cmd2,v_cmd3]
    """
    gen = action[:, 0:3]
    storage = action[:, 3:6]
    v_cmd = action[:, 6:9]

    # Smoothly bounded physical values
    gen = 1.2 * torch.sigmoid(gen) + 0.1
    storage = 0.5 * torch.tanh(storage)
    v = 1.0 + 0.12 * torch.tanh(v_cmd)

    # -------- 1) KCL-like nodal balance --------
    inj = gen + ren + storage - load
    kcl_violation = torch.abs(inj)
    kcl_margin = 0.03 - kcl_violation
    kcl_truth = logic(kcl_margin)
    kcl_penalty = torch.mean(F.relu(-kcl_margin) ** 2)

    # -------- 2) Voltage limit constraints --------
    v_low_margin = v - cfg.v_min
    v_high_margin = cfg.v_max - v
    v_truth = logic(torch.minimum(v_low_margin, v_high_margin))
    v_penalty = torch.mean(F.relu(cfg.v_min - v) ** 2 + F.relu(v - cfg.v_max) ** 2)

    # -------- 3) Tie-line thermal stability limit --------
    f12 = 1.6 * torch.abs(inj[:, 0] - inj[:, 1])
    f23 = 1.6 * torch.abs(inj[:, 1] - inj[:, 2])
    f13 = 1.4 * torch.abs(inj[:, 0] - inj[:, 2])
    tie_flow = torch.stack([f12, f23, f13], dim=1)
    flow_margin = cfg.tie_line_limit - tie_flow
    flow_truth = logic(flow_margin)
    flow_penalty = torch.mean(F.relu(tie_flow - cfg.tie_line_limit) ** 2)

    # -------- 4) Storage-renewable complementary rule --------
    vol_gate = torch.sigmoid(25.0 * (ren_vol - cfg.volatility_threshold))
    ren_centered = ren - torch.mean(ren, dim=1, keepdim=True)
    expert_target = -0.6 * torch.tanh(ren_centered)
    comp_gap = storage - expert_target
    comp_margin = 0.08 - torch.abs(comp_gap)
    comp_truth = logic(comp_margin)
    comp_penalty = torch.mean(vol_gate * (comp_gap ** 2))

    corrected_storage = (1.0 - vol_gate) * storage + vol_gate * expert_target
    corrected_action = torch.cat([gen, corrected_storage, v], dim=1)

    total_penalty = 2.0 * kcl_penalty + 1.5 * v_penalty + 1.5 * flow_penalty + 2.5 * comp_penalty

    truth_summary = {
        "kcl_truth": torch.mean(kcl_truth).item(),
        "voltage_truth": torch.mean(v_truth).item(),
        "flow_truth": torch.mean(flow_truth).item(),
        "complement_truth": torch.mean(comp_truth).item(),
    }

    diagnostics = {
        "kcl_penalty": kcl_penalty.item(),
        "v_penalty": v_penalty.item(),
        "flow_penalty": flow_penalty.item(),
        "comp_penalty": comp_penalty.item(),
        "mean_tie_flow": torch.mean(tie_flow).item(),
        "mean_voltage": torch.mean(v).item(),
        "mean_vol_gate": torch.mean(vol_gate).item(),
        "mean_load": torch.mean(load).item(),
        "mean_ren": torch.mean(ren).item(),
    }
    return total_penalty, corrected_action, diagnostics, truth_summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Neuro-symbolic grid control demo (real PSML data)")
    parser.add_argument("--epochs", type=int, default=NSConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=NSConfig.batch_size)
    parser.add_argument("--lr", type=float, default=NSConfig.lr)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default=NSConfig.output_dir)

    # Real data args
    parser.add_argument("--data-root", type=str, default=NSConfig.data_root)
    parser.add_argument("--zones", nargs="+", default=None,
                        help="3 PSML zone names (default: first 3 discovered)")
    parser.add_argument("--seq-len", type=int, default=NSConfig.seq_len)
    parser.add_argument("--stride", type=int, default=NSConfig.stride)
    parser.add_argument("--max-windows", type=int, default=NSConfig.max_windows_per_zone)
    parser.add_argument("--data-fraction", type=float, default=NSConfig.data_fraction,
                        help="Fraction of windows to use, e.g. 0.33 for 1/3")
    args = parser.parse_args()

    cfg = NSConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        output_dir=args.output_dir,
        data_root=args.data_root,
        zones=args.zones if args.zones else _discover_default_zones(args.data_root)[:3],
        seq_len=args.seq_len,
        stride=args.stride,
        max_windows_per_zone=args.max_windows,
        data_fraction=args.data_fraction,
    )
    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-load real data
    print("Loading real PSML data...")
    sampler = RealGridDataSampler(cfg)

    model = ControlNet(in_dim=7, hidden=cfg.hidden, out_dim=9).to(device)
    logic = LogicNeuron(temperature=cfg.logic_temperature).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "total_penalty": [],
        "kcl_penalty": [],
        "v_penalty": [],
        "flow_penalty": [],
        "comp_penalty": [],
        "kcl_truth": [],
        "voltage_truth": [],
        "flow_truth": [],
        "complement_truth": [],
        "mean_load": [],
        "mean_ren": [],
    }

    print("=== Neuro-Symbolic Grid Control Demo (REAL PSML DATA) ===")
    print(f"Zones used as 3 buses: {cfg.zones}")
    print(f"Windows/zone: {sampler.n_windows_per_zone}, data_fraction={cfg.data_fraction}")
    print("Constraints: KCL, voltage limits, tie-line thermal limits, storage-renewable complement")
    print(
        f"device={device}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"v_range=[{cfg.v_min:.2f},{cfg.v_max:.2f}], tie_line_limit={cfg.tie_line_limit:.2f}"
    )

    for epoch in range(1, cfg.epochs + 1):
        state, load, ren, ren_vol = sampler.sample_batch(
            cfg.batch_size, cfg.seed + epoch, device
        )
        raw_action = model(state)

        total_penalty, corrected_action, d, truth = build_symbolic_penalty(
            raw_action, load, ren, ren_vol, logic, cfg
        )

        smooth_loss = 0.01 * torch.mean(corrected_action ** 2)
        loss = total_penalty + smooth_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["total_penalty"].append(float(loss.item()))
        history["kcl_penalty"].append(d["kcl_penalty"])
        history["v_penalty"].append(d["v_penalty"])
        history["flow_penalty"].append(d["flow_penalty"])
        history["comp_penalty"].append(d["comp_penalty"])
        history["kcl_truth"].append(truth["kcl_truth"])
        history["voltage_truth"].append(truth["voltage_truth"])
        history["flow_truth"].append(truth["flow_truth"])
        history["complement_truth"].append(truth["complement_truth"])
        history["mean_load"].append(d.get("mean_load", 0.0))
        history["mean_ren"].append(d.get("mean_ren", 0.0))

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | total={loss.item():.5f} | "
                f"kcl={d['kcl_penalty']:.5f} v={d['v_penalty']:.5f} "
                f"flow={d['flow_penalty']:.5f} comp={d['comp_penalty']:.5f} | "
                f"truth(kcl/v/flow/comp)=({truth['kcl_truth']:.3f}/"
                f"{truth['voltage_truth']:.3f}/{truth['flow_truth']:.3f}/"
                f"{truth['complement_truth']:.3f}) | "
                f"L={d.get('mean_load',0):.3f} R={d.get('mean_ren',0):.3f}"
            )

    history_path = output_dir / "ns_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved history: {history_path}")
    print("Plot: python demo/plot_neurosymbolic_demo.py")
    print("Done. Real PSML data → neuro-symbolic penalty gradients.")


if __name__ == "__main__":
    main()
