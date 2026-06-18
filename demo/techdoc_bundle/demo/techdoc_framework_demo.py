#!/usr/bin/env python3
"""
TechDoc Unified Framework Demo — Real PSML Data
==================================================
SNN-MLP anomaly detection + multimodal load forecast + neuro-symbolic control.

Branch (--branch):
  snn          Early-fusion SNN-MLP (11-d) + CE + proximity + symbolic penalties
  multimodal   4-modality fusion + next-step load MSE + symbolic penalties
  both         All of the above (default)

Joint loss (both):
  L = L_ce + w_pred * L_pred + w_ctrl * L_penalty + w_smooth * L_smooth

Usage:
  python demo/techdoc_framework_demo.py --branch both --epochs 40 --data-fraction 0.33
  python demo/plot_techdoc_framework_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Bundle root on path
_BUNDLE = Path(__file__).resolve().parent.parent
if str(_BUNDLE) not in sys.path:
    sys.path.insert(0, str(_BUNDLE))

from demo.neurosymbolic_grid_demo import (  # noqa: E402
    ControlNet, LogicNeuron, NSConfig, build_symbolic_penalty,
)
from src.control.multimodal_control_network import MultimodalEmbedding  # noqa: E402
from src.data.load_renewable_dataset import (  # noqa: E402
    MODALITY_COLUMNS, MODALITY_DIMS, ALL_FEATURE_COLS, _COL_IDX,
    LoadRenewableDataLoader, load_multi_zone_multimodal,
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


class MultimodalPretrainHead(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, fused_seq):
        h = fused_seq.mean(dim=1)
        return self.head(h).squeeze(-1), h


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
    fusion_embedding_dim: int = 32
    epochs: int = 40
    lr: float = 1e-3
    w_ctrl: float = 0.35; w_pred: float = 0.5; w_smooth: float = 0.01
    lif_threshold: float = 1.0; lif_leak: float = 0.9
    seed: int = 42
    device: str = "auto"
    output_dir: str = "demo/techdoc_results"
    branch: str = "both"

    data_root: str = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    zones: List[str] = None
    stride: int = 96
    max_windows_per_zone: int = 500
    data_fraction: float = 0.33
    train_ratio: float = 0.8

    def __post_init__(self):
        if self.zones is None:
            self.zones = _default_zones(self.data_root)[:3]
        if len(self.zones) < 3:
            self.zones = (self.zones * 3)[:3]


# ======================================================================
# Unified Model
# ======================================================================

class UnifiedModel(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.branch = cfg.branch
        self.use_snn = cfg.branch in ("snn", "both")
        self.use_mm = cfg.branch in ("multimodal", "both")

        if self.use_snn:
            self.snn = ThreeLayerSNNMLP(cfg.num_features, cfg.hidden1,
                                        cfg.hidden2, cfg.hidden3,
                                        cfg.lif_threshold, cfg.lif_leak)
            self.risk_head = RepresentationLearningHead(cfg.hidden3, cfg.embedding_dim)
        if self.use_mm:
            self.mm_embed = MultimodalEmbedding(MODALITY_DIMS, hidden_dim=64,
                                                embedding_dim=cfg.fusion_embedding_dim)
            self.pretrain_head = MultimodalPretrainHead(cfg.fusion_embedding_dim)

        policy_in = 7 + (cfg.fusion_embedding_dim if self.use_mm else 0)
        self.policy = ControlNet(in_dim=policy_in, hidden=64, out_dim=9)

    def forward_snn(self, x):
        logits, z3 = self.snn(x)
        _, prox = self.risk_head(z3)
        return logits, prox

    def forward_mm(self, mods):
        fused, _ = self.mm_embed.forward_temporal(mods)
        y_hat, h = self.pretrain_head(fused)
        return y_hat, h

    def policy_state(self, gs, mm_h=None):
        return torch.cat([gs, mm_h], dim=1) if mm_h is not None else gs


# ======================================================================
# Batch Sampler
# ======================================================================

def sample_batch(data, indices, batch, seed, device, w11_list, gl_list, gr_list,
                 mods_list, nl_list):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = indices[torch.randint(0, len(indices), (batch,), generator=g).numpy()]

    w11 = torch.from_numpy(w11_list[0][idx]).float().to(device)
    y_anom = (w11.abs().amax(dim=1).amax(dim=1) > 0.95).long().to(device)

    mod_batch = {}
    for mod_name in MODALITY_COLUMNS:
        ci = [_COL_IDX[c] for c in MODALITY_COLUMNS[mod_name] if c in _COL_IDX]
        mod_batch[mod_name] = w11[:, :, ci]
    y_load = torch.from_numpy(nl_list[0][idx]).float().to(device)

    lbs, rbs = [], []
    for zi in range(3):
        lbs.append(torch.from_numpy(gl_list[zi][idx]).float())
        rbs.append(torch.from_numpy(gr_list[zi][idx]).float())
    load = torch.stack(lbs, dim=1).to(device)
    ren = torch.stack(rbs, dim=1).to(device)
    rv = torch.mean(torch.abs(ren - torch.roll(ren, 1, 0)), dim=1, keepdim=True)
    if rv.shape[0] > 1:
        rv[0] = rv[1]
    gs = torch.cat([load, ren, rv], dim=1)

    return {"x11": w11, "y_anom": y_anom, "modalities": mod_batch,
            "y_load": y_load, "grid_state": gs, "load": load,
            "ren": ren, "ren_vol": rv}


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TechDoc Unified Framework Demo (Real PSML Data)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--w-ctrl", type=float, default=0.35)
    parser.add_argument("--w-pred", type=float, default=0.5)
    parser.add_argument("--branch", choices=["snn", "multimodal", "both"],
                        default="both")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default="demo/techdoc_results")
    parser.add_argument("--data-root", type=str,
                        default="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable")
    parser.add_argument("--zones", nargs="+", default=None)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-windows", type=int, default=500)
    parser.add_argument("--data-fraction", type=float, default=1.0)
    args = parser.parse_args()

    cfg = Config(
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        w_ctrl=args.w_ctrl, w_pred=args.w_pred,
        branch=args.branch, device=args.device, output_dir=args.output_dir,
        data_root=args.data_root,
        zones=args.zones if args.zones else Config.zones,
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

    print("Loading real PSML multimodal data...")
    data = load_multi_zone_multimodal(
        cfg.data_root, cfg.zones, seq_len=cfg.seq_len, stride=cfg.stride,
        normalize="zscore", max_windows_per_zone=cfg.max_windows_per_zone,
        data_fraction=cfg.data_fraction,
        train_ratio=cfg.train_ratio, seed=cfg.seed)

    ns_cfg = NSConfig(epochs=cfg.epochs, batch_size=cfg.batch_size, lr=cfg.lr,
                      device=args.device, output_dir=str(out))
    logic = LogicNeuron(temperature=ns_cfg.logic_temperature).to(device)

    model = UnifiedModel(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history: dict = {
        "branch": cfg.branch, "loss_total": [], "loss_ce": [],
        "loss_pred": [], "rmse_pred": [], "mean_proximity": [],
        "kcl_penalty": [], "v_penalty": [], "flow_penalty": [],
        "comp_penalty": [], "kcl_truth": [], "voltage_truth": [],
        "flow_truth": [], "complement_truth": [],
    }

    print(f"=== TechDoc Unified Framework (Real PSML) ===")
    print(f"Zones={cfg.zones}, windows/zone={data['n_windows']}, "
          f"train={len(data['train_idx'])}, branch={cfg.branch}")
    print(f"SNN={cfg.branch in ('snn','both')}  "
          f"Multimodal={cfg.branch in ('multimodal','both')}  "
          f"Neuro-Symbolic: KCL/V/Flow/Complement")

    for epoch in range(1, cfg.epochs + 1):
        seed = cfg.seed + epoch
        loss_ce = torch.tensor(0.0, device=device)
        loss_pred = torch.tensor(0.0, device=device)
        prox_mean, acc, rmse_pred, mm_h = 0.0, 0.0, 0.0, None

        bd = sample_batch(data, data["train_idx"], cfg.batch_size, seed, device,
                          data["windows_11d"], data["grid_loads"],
                          data["grid_rens"], data["modalities"],
                          data["next_loads"])

        if cfg.branch in ("snn", "both"):
            logits, prox = model.forward_snn(bd["x11"])
            loss_ce = F.cross_entropy(logits, bd["y_anom"])
            prox_mean = float(prox.mean().item())
            acc = (logits.argmax(1) == bd["y_anom"]).float().mean().item()

        if cfg.branch in ("multimodal", "both"):
            y_hat, mm_h = model.forward_mm(bd["modalities"])
            loss_pred = F.mse_loss(y_hat, bd["y_load"])
            rmse_pred = float(torch.sqrt(loss_pred).item())

        state = model.policy_state(bd["grid_state"], mm_h)
        raw = model.policy(state)
        total_pen, corrected, diag, truth = build_symbolic_penalty(
            raw, bd["load"], bd["ren"], bd["ren_vol"], logic, ns_cfg)
        smooth = torch.mean(corrected ** 2)

        loss = cfg.w_ctrl * total_pen + cfg.w_smooth * smooth
        if cfg.branch in ("snn", "both"):
            loss = loss + loss_ce
        if cfg.branch in ("multimodal", "both"):
            loss = loss + cfg.w_pred * loss_pred

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for k, v in [
            ("loss_total", loss.item()), ("loss_ce", loss_ce.item()),
            ("loss_pred", loss_pred.item()), ("rmse_pred", rmse_pred),
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
            msg = (f"Epoch {epoch:03d}/{cfg.epochs} | loss={loss.item():.4f} "
                   f"pen(kcl/v/flow/comp)={diag['kcl_penalty']:.4f}/"
                   f"{diag['v_penalty']:.6f}/{diag['flow_penalty']:.4f}/"
                   f"{diag['comp_penalty']:.4f}")
            if cfg.branch in ("snn", "both"):
                msg += f" | ce={loss_ce.item():.4f} acc={acc:.3f} prox={prox_mean:.3f}"
            if cfg.branch in ("multimodal", "both"):
                msg += f" | mse={loss_pred.item():.4f} rmse={rmse_pred:.4f}"
            print(msg)

    hist_path = out / "techdoc_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Saved: {hist_path}")
    print("Plot: python demo/plot_techdoc_framework_demo.py")


if __name__ == "__main__":
    main()
