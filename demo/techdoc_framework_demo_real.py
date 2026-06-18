#!/usr/bin/env python3
"""
Unified tech-doc demo — REAL PSML DATA VERSION.

Branches (--branch):
  - snn:        early-fusion SNN-MLP (11-d real PSML features) + CE + proximity + symbolic penalties
  - multimodal: 4-modality fusion (real PSML columns) + load pretrain (MSE) + symbolic penalties
  - both:       all of the above (default)

Real PSML data replaces the synthetic generators from the original demo.
The 4 modalities map directly to the 11 PSML columns:
  load(1) + renewable(2) + irradiance(4) + weather(4) = 11

Joint loss (both branch):
  L = L_ce + w_pred * L_pred + w_ctrl * L_penalty + w_smooth * L_smooth

Usage:
  python demo/techdoc_framework_demo_real.py --branch both --epochs 40 --data-fraction 0.33
  python demo/plot_techdoc_framework_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
for p in (_DEMO_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import neurosymbolic_grid_demo_real as nsgrid_real  # noqa: E402
from neurosymbolic_grid_demo_real import (  # noqa: E402
    ControlNet, LogicNeuron, NSConfig, build_symbolic_penalty,
)
from src.control.multimodal_control_network import MultimodalEmbedding  # noqa: E402
from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset  # noqa: E402

# PSML column → modality mapping (matches MultimodalPSMLDataset)
MODALITY_COLUMNS: Dict[str, List[str]] = {
    "load": ["load_power"],
    "renewable": ["wind_power", "solar_power"],
    "irradiance": ["DHI", "DNI", "GHI", "Solar Zenith Angle"],
    "weather": ["Dew Point", "Wind Speed", "Relative Humidity", "Temperature"],
}
MODALITY_DIMS: Dict[str, int] = {name: len(cols) for name, cols in MODALITY_COLUMNS.items()}
ALL_FEATURE_COLS: List[str] = [c for cols in MODALITY_COLUMNS.values() for c in cols]

# Column index mapping in the 11-d numpy array
_COL_IDX = {c: i for i, c in enumerate(ALL_FEATURE_COLS)}


def _discover_default_zones(data_root: str) -> List[str]:
    loader = LoadRenewableDataLoader(data_root)
    return sorted(loader.zones.keys())[:3]


# ---------------------------------------------------------------------------
# Surrogate gradient + LIF + SNN-MLP (unchanged from original)
# ---------------------------------------------------------------------------

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.threshold = threshold
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        th = ctx.threshold
        grad_x = grad_output * torch.clamp(1.0 - (x - th).abs(), min=0.0)
        return grad_x, None


class LIFNeuron(nn.Module):
    def __init__(self, threshold: float = 1.0, leak: float = 0.9):
        super().__init__()
        self.threshold = threshold
        self.leak = leak

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros(x.shape[0], x.shape[2], device=x.device, dtype=x.dtype)
        spikes = []
        for t in range(x.shape[1]):
            mem = self.leak * mem + x[:, t, :]
            spk = SurrogateSpike.apply(mem, self.threshold)
            mem = mem * (1.0 - spk)
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

    def forward(self, x: torch.Tensor) -> tuple:
        z1 = self.lif1(self.fc1(x))
        z2 = self.lif2(self.fc2(z1))
        z3 = self.lif3(self.fc3(z2))
        rate = z3.mean(dim=1)
        logits = self.classifier(rate)
        return logits, z3


class RepresentationLearningHead(nn.Module):
    def __init__(self, hidden_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.dynamic_center = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, encoded: torch.Tensor) -> tuple:
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        embedding = self.projection(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        c = F.normalize(self.dynamic_center, p=2, dim=0)
        proximity = F.cosine_similarity(embedding, c.unsqueeze(0))
        return embedding, proximity


class MultimodalPretrainHead(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, fused_seq: torch.Tensor) -> tuple:
        h = fused_seq.mean(dim=1)
        return self.head(h).squeeze(-1), h


# ---------------------------------------------------------------------------
# Real PSML data sampler for the unified framework
# ---------------------------------------------------------------------------

class RealUnifiedDataSampler:
    """Pre-loads 3 PSML zones and provides batches for SNN + multimodal + control."""

    def __init__(self, cfg: "TechDocDemoConfig"):
        self.cfg = cfg
        self.loader = LoadRenewableDataLoader(cfg.data_root)
        self.zones = cfg.zones[:3]

        # Per-zone: list of (window_11d, next_load)
        self.all_windows_11d: List[np.ndarray] = []  # each: (N, seq_len, 11)
        self.all_next_loads: List[np.ndarray] = []   # each: (N,)
        self.n_windows = 0

        # Also build grid-control data (same as neurosymbolic sampler)
        self.zone_loads: List[np.ndarray] = []  # each: (N,)
        self.zone_rens: List[np.ndarray] = []   # each: (N,)

        for zone in self.zones:
            df = self.loader.load_zone(zone)
            # Ensure columns exist
            missing = [c for c in ALL_FEATURE_COLS if c not in df.columns]
            if missing:
                print(f"WARNING: {zone} missing columns: {missing}")

            # Use available numeric columns that match ALL_FEATURE_COLS
            avail = [c for c in ALL_FEATURE_COLS if c in df.columns]
            df_sub = df[avail].apply(pd.to_numeric, errors="coerce").dropna()

            # Z-score normalize
            mean = df_sub.mean()
            std = df_sub.std() + 1e-8
            df_norm = ((df_sub - mean) / std).astype(np.float32)

            n_total_rows = len(df_norm)
            n_windows = (n_total_rows - cfg.seq_len - 1) // cfg.stride
            if n_windows < 1:
                raise RuntimeError(f"Not enough data for zone {zone}: need seq_len={cfg.seq_len}")

            n_use = min(n_windows, cfg.max_windows_per_zone)
            if 0.0 < cfg.data_fraction < 1.0:
                n_use = min(n_use, max(1, int(n_windows * cfg.data_fraction)))

            windows_11d = []
            next_loads = []
            zone_load_vals = []
            zone_ren_vals = []

            for i in range(n_use):
                start = i * cfg.stride
                end = start + cfg.seq_len
                block = df_norm.iloc[start:end]
                # Align to ALL_FEATURE_COLS order, fill missing with 0
                arr = np.zeros((cfg.seq_len, 11), dtype=np.float32)
                for j, col in enumerate(ALL_FEATURE_COLS):
                    if col in block.columns:
                        arr[:, j] = block[col].values.astype(np.float32)

                windows_11d.append(arr)

                # Next-step load
                target_row = end
                if target_row < len(df_norm) and "load_power" in df_norm.columns:
                    next_loads.append(float(df_norm.iloc[target_row]["load_power"]))
                else:
                    next_loads.append(0.0)

                # Grid state: avg load and avg ren over window
                load_val = arr[:, 0].mean()  # load_power
                ren_val = arr[:, 1:3].sum(axis=1).mean()  # wind + solar
                zone_load_vals.append(load_val)
                zone_ren_vals.append(ren_val)

            data_11d = np.stack(windows_11d, axis=0)
            self.all_windows_11d.append(data_11d)
            self.all_next_loads.append(np.array(next_loads, dtype=np.float32))
            self.zone_loads.append(np.array(zone_load_vals, dtype=np.float32))
            self.zone_rens.append(np.array(zone_ren_vals, dtype=np.float32))

        self.n_windows = min(d.shape[0] for d in self.all_windows_11d)
        print(f"RealUnifiedDataSampler: {len(self.zones)} zones, "
              f"{self.n_windows} windows/zone, "
              f"seq_len={cfg.seq_len}, data_fraction={cfg.data_fraction}")

    def sample_batch(
        self, batch: int, seed: int, device: torch.device
    ) -> Dict:
        """
        Returns dict with:
          x11:       (B, T, 11)  — SNN early-fusion input
          y_anom:    (B,)        — anomaly labels (rule-based)
          modalities: {name: (B, T, d_m)} — multimodal inputs
          y_load:    (B,)        — next-step load target
          grid_state:(B, 7)      — control policy input (load×3, ren×3, vol×1)
          load:      (B, 3)      — bus loads
          ren:       (B, 3)      — bus renewables
          ren_vol:   (B, 1)      — volatility
        """
        g = torch.Generator(device="cpu").manual_seed(seed)
        n = self.n_windows

        loads_bus = []
        rens_bus = []
        x11_batches = []
        y_load_batches = []
        mod_batches: Dict[str, list] = {k: [] for k in MODALITY_COLUMNS}

        for zi in range(3):
            idx = torch.randint(0, self.all_windows_11d[zi].shape[0], (batch,), generator=g).numpy()

            # 11-d windows
            w11 = self.all_windows_11d[zi][idx]  # (batch, seq_len, 11)
            x11_batches.append(torch.from_numpy(w11))

            # Next-step load
            yl = self.all_next_loads[zi][idx]
            y_load_batches.append(torch.from_numpy(yl))

            # Grid state
            l = self.zone_loads[zi][idx]
            r = self.zone_rens[zi][idx]
            loads_bus.append(torch.from_numpy(l))
            rens_bus.append(torch.from_numpy(r))

            # Split into modalities
            for mod_name, cols in MODALITY_COLUMNS.items():
                col_indices = [_COL_IDX[c] for c in cols if c in _COL_IDX]
                mod_data = w11[:, :, col_indices]  # (batch, seq_len, d_m)
                mod_batches[mod_name].append(torch.from_numpy(mod_data))

        # Concatenate across 3 zones → (3*batch, ...) then we'll use batch*3
        # Actually, we want to mix zones in each batch. Let's interleave.
        # Simpler: just stack and reshape to give variety.
        # For SNN: stack all zones → (3, B, T, 11) → reshape to (3B, T, 11) but we want B samples
        # Better: just use zone 0 for SNN/multimodal, zones 0-2 for grid control
        # Actually, let's just use zone 0 for SNN/multimodal to keep batch size consistent

        x11 = x11_batches[0].float().to(device)           # (B, T, 11)
        y_load = y_load_batches[0].float().to(device)      # (B,)

        # Anomaly labels (rule-based, same as original)
        score = x11.abs().amax(dim=1).amax(dim=1)  # (B,)
        y_anom = (score > 0.95).long().to(device)

        modalities: Dict[str, torch.Tensor] = {}
        for mod_name in MODALITY_COLUMNS:
            modalities[mod_name] = mod_batches[mod_name][0].float().to(device)  # (B, T, d_m)

        # Grid control: 3 buses from 3 zones
        load = torch.stack(loads_bus, dim=1).float().to(device)  # (B, 3)
        ren = torch.stack(rens_bus, dim=1).float().to(device)    # (B, 3)
        ren_vol = torch.mean(torch.abs(ren - torch.roll(ren, 1, 0)), dim=1, keepdim=True)
        ren_vol[0] = ren_vol[1] if ren_vol.shape[0] > 1 else ren_vol[0]
        grid_state = torch.cat([load, ren, ren_vol], dim=1)  # (B, 7)

        return {
            "x11": x11,
            "y_anom": y_anom,
            "modalities": modalities,
            "y_load": y_load,
            "grid_state": grid_state,
            "load": load,
            "ren": ren,
            "ren_vol": ren_vol,
        }


# ---------------------------------------------------------------------------
# Config + Unified Model
# ---------------------------------------------------------------------------

@dataclass
class TechDocDemoConfig:
    batch_size: int = 64
    seq_len: int = 96
    num_features: int = 11
    hidden1: int = 128
    hidden2: int = 64
    hidden3: int = 32
    embedding_dim: int = 64
    fusion_embedding_dim: int = 32
    epochs: int = 40
    lr: float = 1e-3
    w_ctrl: float = 0.35
    w_pred: float = 0.5
    w_smooth: float = 0.01
    lif_threshold: float = 1.0
    lif_leak: float = 0.9
    seed: int = 42
    device: str = "auto"
    output_dir: str = "demo/techdoc_results"
    branch: str = "both"

    # Real data params
    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones: List[str] = field(default_factory=list)
    max_windows_per_zone: int = 500
    data_fraction: float = 0.33
    stride: int = 96

    def __post_init__(self):
        if not self.zones:
            self.zones = _discover_default_zones(self.data_root)[:3]
        if len(self.zones) < 3:
            self.zones = (self.zones * 3)[:3]


class TechDocUnifiedModel(nn.Module):
    def __init__(self, cfg: TechDocDemoConfig):
        super().__init__()
        self.branch = cfg.branch
        self.use_snn = cfg.branch in ("snn", "both")
        self.use_multimodal = cfg.branch in ("multimodal", "both")

        if self.use_snn:
            self.snn = ThreeLayerSNNMLP(
                cfg.num_features, cfg.hidden1, cfg.hidden2, cfg.hidden3,
                cfg.lif_threshold, cfg.lif_leak,
            )
            self.risk_head = RepresentationLearningHead(cfg.hidden3, cfg.embedding_dim)

        if self.use_multimodal:
            self.multimodal_embed = MultimodalEmbedding(
                MODALITY_DIMS, hidden_dim=64, embedding_dim=cfg.fusion_embedding_dim,
            )
            self.pretrain_head = MultimodalPretrainHead(cfg.fusion_embedding_dim)

        policy_in = 7 + (cfg.fusion_embedding_dim if self.use_multimodal else 0)
        self.policy = ControlNet(in_dim=policy_in, hidden=64, out_dim=9)

    def forward_snn(self, x: torch.Tensor) -> tuple:
        logits, z3 = self.snn(x)
        _, proximity = self.risk_head(z3)
        return logits, z3, proximity

    def forward_multimodal(self, modalities: Dict[str, torch.Tensor]) -> tuple:
        fused, _ = self.multimodal_embed.forward_temporal(modalities)
        y_hat, h = self.pretrain_head(fused)
        return y_hat, h, fused

    def policy_state(
        self, grid_state: torch.Tensor, multimodal_h: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if multimodal_h is not None:
            return torch.cat([grid_state, multimodal_h], dim=1)
        return grid_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Tech-doc unified demo (real PSML data)")
    parser.add_argument("--epochs", type=int, default=TechDocDemoConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TechDocDemoConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TechDocDemoConfig.lr)
    parser.add_argument("--w-ctrl", type=float, default=TechDocDemoConfig.w_ctrl)
    parser.add_argument("--w-pred", type=float, default=TechDocDemoConfig.w_pred)
    parser.add_argument(
        "--branch", choices=["snn", "multimodal", "both"], default="both",
        help="snn=early fusion only; multimodal=fusion+load MSE; both=full framework",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default=TechDocDemoConfig.output_dir)

    # Real data args
    parser.add_argument("--data-root", type=str, default=TechDocDemoConfig.data_root)
    parser.add_argument("--zones", nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=TechDocDemoConfig.seq_len)
    parser.add_argument("--stride", type=int, default=TechDocDemoConfig.stride)
    parser.add_argument("--max-windows", type=int, default=TechDocDemoConfig.max_windows_per_zone)
    parser.add_argument("--data-fraction", type=float, default=TechDocDemoConfig.data_fraction,
                        help="Fraction of data to use, e.g. 0.33 for 1/3")
    args = parser.parse_args()

    cfg = TechDocDemoConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        w_ctrl=args.w_ctrl,
        w_pred=args.w_pred,
        output_dir=args.output_dir,
        device=args.device,
        branch=args.branch,
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

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Pre-load real data
    print("Loading real PSML data...")
    sampler = RealUnifiedDataSampler(cfg)

    ns_cfg = NSConfig(
        epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
        device=args.device, output_dir=str(out),
    )
    logic = LogicNeuron(temperature=ns_cfg.logic_temperature).to(device)

    model = TechDocUnifiedModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict = {
        "branch": cfg.branch,
        "loss_total": [],
        "loss_ce": [],
        "loss_pred": [],
        "rmse_pred": [],
        "mean_proximity": [],
        "kcl_penalty": [],
        "v_penalty": [],
        "flow_penalty": [],
        "comp_penalty": [],
        "kcl_truth": [],
        "voltage_truth": [],
        "flow_truth": [],
        "complement_truth": [],
    }

    print("=== Tech-doc unified framework demo (REAL PSML DATA) ===")
    print(f"Zones: {cfg.zones}")
    print(f"Windows/zone: {sampler.n_windows}, data_fraction={cfg.data_fraction}")
    print(
        f"device={device} | branch={cfg.branch} | "
        f"SNN={cfg.branch in ('snn', 'both')} | "
        f"multimodal={cfg.branch in ('multimodal', 'both')} | "
        f"neuro-symbolic penalties (KCL/V/flow/complement)"
    )

    for epoch in range(1, cfg.epochs + 1):
        seed = cfg.seed + epoch
        loss_ce = torch.tensor(0.0, device=device)
        loss_pred = torch.tensor(0.0, device=device)
        proximity_mean = 0.0
        acc = 0.0
        rmse_pred = 0.0
        multimodal_h = None

        batch_data = sampler.sample_batch(cfg.batch_size, seed, device)

        if cfg.branch in ("snn", "both"):
            x, y = batch_data["x11"], batch_data["y_anom"]
            logits, _, proximity = model.forward_snn(x)
            loss_ce = F.cross_entropy(logits, y)
            proximity_mean = float(proximity.mean().item())
            acc = (logits.argmax(dim=1) == y).float().mean().item()

        if cfg.branch in ("multimodal", "both"):
            y_hat, multimodal_h, _ = model.forward_multimodal(batch_data["modalities"])
            loss_pred = F.mse_loss(y_hat, batch_data["y_load"])
            rmse_pred = float(torch.sqrt(loss_pred).item())

        grid_state = batch_data["grid_state"]
        load = batch_data["load"]
        ren = batch_data["ren"]
        ren_vol = batch_data["ren_vol"]

        state = model.policy_state(grid_state, multimodal_h)
        raw_action = model.policy(state)
        total_penalty, corrected_action, diag, truth = build_symbolic_penalty(
            raw_action, load, ren, ren_vol, logic, ns_cfg,
        )
        smooth = torch.mean(corrected_action ** 2)

        loss = cfg.w_ctrl * total_penalty + cfg.w_smooth * smooth
        if cfg.branch in ("snn", "both"):
            loss = loss + loss_ce
        if cfg.branch in ("multimodal", "both"):
            loss = loss + cfg.w_pred * loss_pred

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        history["loss_total"].append(float(loss.item()))
        history["loss_ce"].append(float(loss_ce.item()))
        history["loss_pred"].append(float(loss_pred.item()))
        history["rmse_pred"].append(rmse_pred)
        history["mean_proximity"].append(proximity_mean)
        history["kcl_penalty"].append(diag["kcl_penalty"])
        history["v_penalty"].append(diag["v_penalty"])
        history["flow_penalty"].append(diag["flow_penalty"])
        history["comp_penalty"].append(diag["comp_penalty"])
        history["kcl_truth"].append(truth["kcl_truth"])
        history["voltage_truth"].append(truth["voltage_truth"])
        history["flow_truth"].append(truth["flow_truth"])
        history["complement_truth"].append(truth["complement_truth"])

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            msg = (
                f"Epoch {epoch:03d}/{cfg.epochs} | loss={loss.item():.4f} "
                f"pen(kcl/v/flow/comp)={diag['kcl_penalty']:.4f}/"
                f"{diag['v_penalty']:.4f}/{diag['flow_penalty']:.4f}/{diag['comp_penalty']:.4f}"
            )
            if cfg.branch in ("snn", "both"):
                msg += f" | ce={loss_ce.item():.4f} acc={acc:.3f} prox={proximity_mean:.3f}"
            if cfg.branch in ("multimodal", "both"):
                msg += f" | pred_mse={loss_pred.item():.4f} rmse={rmse_pred:.4f}"
            print(msg)

    hist_path = out / "techdoc_history.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {hist_path}")
    print("Plot: python demo/plot_techdoc_framework_demo.py")
    print("Done. Real PSML data → unified SNN + multimodal + neuro-symbolic framework.")


if __name__ == "__main__":
    main()
