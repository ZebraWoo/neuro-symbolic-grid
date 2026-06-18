#!/usr/bin/env python3
"""
Unified tech-doc demo: SNN-MLP + risk operator + multimodal fusion + neuro-symbolic control.

Branches (--branch):
  - snn:        early-fusion SNN-MLP (11-d) + CE + proximity + symbolic penalties
  - multimodal: 4-modality fusion + load pretrain (MSE) + symbolic penalties (no CE)
  - both:       all of the above (default)

Neuro-symbolic operators are unchanged (imported from neurosymbolic_grid_demo):
  KCL, voltage, tie-line flow, storage-renewable complement + LogicNeuron truth.

Joint loss (both branch):
  L = L_ce + w_pred * L_pred + w_ctrl * L_penalty + w_smooth * L_smooth

Run from project root:
  python demo/techdoc_framework_demo.py
  python demo/techdoc_framework_demo.py --branch multimodal --epochs 40
  python demo/plot_techdoc_framework_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEMO_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DEMO_DIR.parent
for p in (_DEMO_DIR, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import neurosymbolic_grid_demo as nsgrid  # noqa: E402
from src.control.multimodal_control_network import MultimodalEmbedding  # noqa: E402

MODALITY_DIMS: Dict[str, int] = {
    "load": 1,
    "renewable": 2,
    "irradiance": 4,
    "weather": 4,
}


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
    """Document table: (B,T,11)->...->(B,T,32) spikes, then rate (B,32) -> (B,2)."""

    def __init__(self, input_dim: int, h1: int, h2: int, h3: int, threshold: float, leak: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h1)
        self.lif1 = LIFNeuron(threshold, leak)
        self.fc2 = nn.Linear(h1, h2)
        self.lif2 = LIFNeuron(threshold, leak)
        self.fc3 = nn.Linear(h2, h3)
        self.lif3 = LIFNeuron(threshold, leak)
        self.classifier = nn.Linear(h3, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z1 = self.lif1(self.fc1(x))
        z2 = self.lif2(self.fc2(z1))
        z3 = self.lif3(self.fc3(z2))
        rate = z3.mean(dim=1)
        logits = self.classifier(rate)
        return logits, z3


class RepresentationLearningHead(nn.Module):
    """Global pool -> embedding -> L2 -> cosine proximity to dynamic_center."""

    def __init__(self, hidden_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.dynamic_center = nn.Parameter(torch.randn(embedding_dim))

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.global_pool(encoded.transpose(1, 2)).squeeze(-1)
        embedding = self.projection(pooled)
        embedding = F.normalize(embedding, p=2, dim=1)
        c = F.normalize(self.dynamic_center, p=2, dim=0)
        proximity = F.cosine_similarity(embedding, c.unsqueeze(0))
        return embedding, proximity


class MultimodalPretrainHead(nn.Module):
    """MACPTM-demo: pooled fusion embedding -> next-step load (scalar)."""

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, fused_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = fused_seq.mean(dim=1)
        return self.head(h).squeeze(-1), h


def synthesize_psml_like_batch(
    batch: int, seq_len: int, feat_dim: int, seed: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Synthetic (B,T,11) early-fusion sequence + anomaly labels."""
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.linspace(0, 6.28, seq_len, device=device).unsqueeze(0).expand(batch, -1)
    phase = torch.randn(batch, 1, device=device, generator=g) * 0.3
    load = 0.5 + 0.25 * torch.sin(t + phase) + 0.04 * torch.randn(batch, seq_len, device=device, generator=g)
    wind = 0.35 + 0.2 * torch.sin(1.7 * t + phase * 2) + 0.05 * torch.randn(batch, seq_len, device=device, generator=g)
    solar = torch.clamp(0.1 + 0.55 * torch.sin(t + 1.2) ** 2, 0.0, 1.0) + 0.03 * torch.randn(
        batch, seq_len, device=device, generator=g
    )
    extra = 0.1 * torch.randn(batch, seq_len, max(0, feat_dim - 3), device=device, generator=g)
    x = torch.cat([load.unsqueeze(-1), wind.unsqueeze(-1), solar.unsqueeze(-1), extra], dim=-1)
    x = x[:, :, :feat_dim]
    score = x.abs().amax(dim=1).amax(dim=1)
    y = (score > 0.95).long()
    return x, y


def synthesize_multimodal_batch(
    batch: int, seq_len: int, seed: int, device: torch.device
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Four PSML-like modalities (B,T,dm) + next-step load target + 11-d early fusion for SNN branch.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    t = torch.linspace(0, 6.28, seq_len, device=device).unsqueeze(0).expand(batch, -1)
    phase = torch.randn(batch, 1, device=device, generator=g) * 0.3

    load = 0.5 + 0.25 * torch.sin(t + phase) + 0.04 * torch.randn(batch, seq_len, device=device, generator=g)
    wind = 0.35 + 0.2 * torch.sin(1.7 * t + phase * 2) + 0.05 * torch.randn(batch, seq_len, device=device, generator=g)
    solar = torch.clamp(0.1 + 0.55 * torch.sin(t + 1.2) ** 2, 0.0, 1.0) + 0.03 * torch.randn(
        batch, seq_len, device=device, generator=g
    )
    dhi = 0.3 + 0.15 * torch.sin(0.9 * t) + 0.03 * torch.randn(batch, seq_len, device=device, generator=g)
    dni = 0.25 + 0.12 * torch.sin(1.1 * t + 0.5) + 0.03 * torch.randn(batch, seq_len, device=device, generator=g)
    ghi = 0.4 + 0.2 * torch.sin(t + 0.8) + 0.03 * torch.randn(batch, seq_len, device=device, generator=g)
    zenith = 0.5 + 0.1 * torch.cos(t) + 0.02 * torch.randn(batch, seq_len, device=device, generator=g)
    dew = 0.4 + 0.08 * torch.sin(2.0 * t) + 0.02 * torch.randn(batch, seq_len, device=device, generator=g)
    wspd = 0.35 + 0.1 * torch.sin(1.5 * t + 1.0) + 0.03 * torch.randn(batch, seq_len, device=device, generator=g)
    rh = 0.55 + 0.1 * torch.cos(1.3 * t) + 0.02 * torch.randn(batch, seq_len, device=device, generator=g)
    temp = 0.45 + 0.12 * torch.sin(0.7 * t + phase) + 0.02 * torch.randn(batch, seq_len, device=device, generator=g)

    modalities = {
        "load": load.unsqueeze(-1),
        "renewable": torch.stack([wind, solar], dim=-1),
        "irradiance": torch.stack([dhi, dni, ghi, zenith], dim=-1),
        "weather": torch.stack([dew, wspd, rh, temp], dim=-1),
    }
    # Next-step load proxy (demo): one step beyond window end
    y_load = load[:, -1] + 0.02 * torch.randn(batch, device=device, generator=g)

    n_fill = max(0, 11 - 3)
    fillers = 0.1 * torch.randn(batch, seq_len, n_fill, device=device, generator=g)
    x11 = torch.cat(
        [load.unsqueeze(-1), wind.unsqueeze(-1), solar.unsqueeze(-1), fillers], dim=-1
    )[:, :, :11]
    return modalities, y_load, x11


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
    branch: str = "both"  # snn | multimodal | both


class TechDocUnifiedModel(nn.Module):
    def __init__(self, cfg: TechDocDemoConfig):
        super().__init__()
        self.branch = cfg.branch
        self.use_snn = cfg.branch in ("snn", "both")
        self.use_multimodal = cfg.branch in ("multimodal", "both")

        if self.use_snn:
            self.snn = ThreeLayerSNNMLP(
                cfg.num_features,
                cfg.hidden1,
                cfg.hidden2,
                cfg.hidden3,
                cfg.lif_threshold,
                cfg.lif_leak,
            )
            self.risk_head = RepresentationLearningHead(cfg.hidden3, cfg.embedding_dim)

        if self.use_multimodal:
            self.multimodal_embed = MultimodalEmbedding(
                MODALITY_DIMS, hidden_dim=64, embedding_dim=cfg.fusion_embedding_dim
            )
            self.pretrain_head = MultimodalPretrainHead(cfg.fusion_embedding_dim)

        policy_in = 7 + (cfg.fusion_embedding_dim if self.use_multimodal else 0)
        self.policy = nsgrid.ControlNet(in_dim=policy_in, hidden=64, out_dim=9)

    def forward_snn(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, z3 = self.snn(x)
        _, proximity = self.risk_head(z3)
        return logits, z3, proximity

    def forward_multimodal(
        self, modalities: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, _ = self.multimodal_embed.forward_temporal(modalities)
        y_hat, h = self.pretrain_head(fused)
        return y_hat, h, fused

    def policy_state(
        self, grid_state: torch.Tensor, multimodal_h: torch.Tensor | None = None
    ) -> torch.Tensor:
        if multimodal_h is not None:
            return torch.cat([grid_state, multimodal_h], dim=1)
        return grid_state


def main():
    parser = argparse.ArgumentParser(description="Tech-doc unified neuro-symbolic + multimodal demo")
    parser.add_argument("--epochs", type=int, default=TechDocDemoConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TechDocDemoConfig.batch_size)
    parser.add_argument("--lr", type=float, default=TechDocDemoConfig.lr)
    parser.add_argument("--w-ctrl", type=float, default=TechDocDemoConfig.w_ctrl)
    parser.add_argument("--w-pred", type=float, default=TechDocDemoConfig.w_pred)
    parser.add_argument(
        "--branch",
        choices=["snn", "multimodal", "both"],
        default=TechDocDemoConfig.branch,
        help="snn=early fusion only; multimodal=fusion+load MSE; both=full framework",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default=TechDocDemoConfig.output_dir)
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
    )
    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    ns_cfg = nsgrid.NSConfig(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=args.device,
        output_dir=str(out),
    )
    logic = nsgrid.LogicNeuron(temperature=ns_cfg.logic_temperature).to(device)

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

    print("=== Tech-doc unified framework demo ===")
    print(
        f"device={device} | branch={cfg.branch} | "
        f"SNN={cfg.branch in ('snn', 'both')} | multimodal={cfg.branch in ('multimodal', 'both')} | "
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

        modalities, y_load, x11 = synthesize_multimodal_batch(
            cfg.batch_size, cfg.seq_len, seed, device
        )

        if cfg.branch in ("snn", "both"):
            if cfg.branch == "snn":
                x, y = synthesize_psml_like_batch(
                    cfg.batch_size, cfg.seq_len, cfg.num_features, seed, device
                )
            else:
                x, y = x11, (x11.abs().amax(dim=1).amax(dim=1) > 0.95).long()
            logits, _, proximity = model.forward_snn(x)
            loss_ce = F.cross_entropy(logits, y)
            proximity_mean = float(proximity.mean().item())
            acc = (logits.argmax(dim=1) == y).float().mean().item()

        if cfg.branch in ("multimodal", "both"):
            y_hat, multimodal_h, _ = model.forward_multimodal(modalities)
            loss_pred = F.mse_loss(y_hat, y_load)
            rmse_pred = float(torch.sqrt(loss_pred).item())

        grid_state, load, ren, ren_vol = nsgrid.synthesize_grid_batch(
            cfg.batch_size, cfg.seed, epoch, device
        )
        state = model.policy_state(grid_state, multimodal_h)
        raw_action = model.policy(state)
        total_penalty, corrected_action, diag, truth = nsgrid.build_symbolic_penalty(
            raw_action, load, ren, ren_vol, logic, ns_cfg
        )
        smooth = torch.mean(corrected_action**2)

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


if __name__ == "__main__":
    main()
