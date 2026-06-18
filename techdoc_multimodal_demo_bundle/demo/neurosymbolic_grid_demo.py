#!/usr/bin/env python3
"""
Neuro-symbolic power-grid control demo.

This demo combines:
1) A neural controller (SNN-inspired small MLP) for continuous control actions.
2) Differentiable symbolic constraints converted by logic neurons:
   - KCL-like nodal power balance
   - Voltage safety bounds
   - Tie-line thermal stability limit
   - Storage/renewable complementary expert rule under high volatility

All violations generate differentiable penalty gradients to correct control actions.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def synthesize_grid_batch(batch: int, seed: int, step_offset: int, device: torch.device):
    g = torch.Generator(device=device).manual_seed(seed + step_offset)
    t = torch.linspace(0, 8.0, batch, device=device)

    # 3-bus load profile
    load = torch.stack(
        [
            1.0 + 0.2 * torch.sin(1.8 * t) + 0.05 * torch.randn(batch, generator=g, device=device),
            0.9 + 0.15 * torch.sin(1.2 * t + 0.6) + 0.05 * torch.randn(batch, generator=g, device=device),
            0.8 + 0.1 * torch.sin(2.1 * t + 1.1) + 0.04 * torch.randn(batch, generator=g, device=device),
        ],
        dim=1,
    )

    # Renewable injections (wind/solar-like aggregate to each bus)
    ren = torch.stack(
        [
            0.45 + 0.22 * torch.sin(2.4 * t + 0.2),
            0.50 + 0.20 * torch.sin(2.7 * t + 1.5),
            0.40 + 0.18 * torch.sin(2.2 * t + 2.2),
        ],
        dim=1,
    )
    ren = torch.clamp(ren, min=0.05)

    # Volatility proxy: |dP_ren/dt|
    ren_roll = torch.roll(ren, shifts=1, dims=0)
    ren_vol = torch.mean(torch.abs(ren - ren_roll), dim=1, keepdim=True)
    ren_vol[0] = ren_vol[1]

    # Feature vector to controller
    state = torch.cat([load, ren, ren_vol], dim=1)  # [batch, 7]
    return state, load, ren, ren_vol


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
    storage = action[:, 3:6]  # + discharge, - charge
    v_cmd = action[:, 6:9]

    # Smoothly bounded physical values
    gen = 1.2 * torch.sigmoid(gen) + 0.1            # [0.1, 1.3]
    storage = 0.5 * torch.tanh(storage)             # [-0.5, 0.5]
    v = 1.0 + 0.12 * torch.tanh(v_cmd)              # ~[0.88, 1.12]

    # -------- 1) KCL-like nodal balance --------
    # injection = generation + renewable + storage - load
    inj = gen + ren + storage - load
    kcl_violation = torch.abs(inj)                  # should be near 0
    kcl_margin = 0.03 - kcl_violation              # margin > 0 means good
    kcl_truth = logic(kcl_margin)
    kcl_penalty = torch.mean(F.relu(-kcl_margin) ** 2)

    # -------- 2) Voltage limit constraints --------
    v_low_margin = v - cfg.v_min
    v_high_margin = cfg.v_max - v
    v_truth = logic(torch.minimum(v_low_margin, v_high_margin))
    v_penalty = torch.mean(F.relu(cfg.v_min - v) ** 2 + F.relu(v - cfg.v_max) ** 2)

    # -------- 3) Tie-line thermal stability limit --------
    # Simple proxy flow based on nodal injection differences
    f12 = 1.6 * torch.abs(inj[:, 0] - inj[:, 1])
    f23 = 1.6 * torch.abs(inj[:, 1] - inj[:, 2])
    f13 = 1.4 * torch.abs(inj[:, 0] - inj[:, 2])
    tie_flow = torch.stack([f12, f23, f13], dim=1)
    flow_margin = cfg.tie_line_limit - tie_flow
    flow_truth = logic(flow_margin)
    flow_penalty = torch.mean(F.relu(tie_flow - cfg.tie_line_limit) ** 2)

    # -------- 4) Storage-renewable complementary rule --------
    # If volatility high -> enforce smoothing:
    # storage should counter net renewable ramps
    vol_gate = torch.sigmoid(25.0 * (ren_vol - cfg.volatility_threshold))  # [batch,1]
    ren_centered = ren - torch.mean(ren, dim=1, keepdim=True)
    expert_target = -0.6 * torch.tanh(ren_centered)  # counter fluctuation
    comp_gap = storage - expert_target
    comp_margin = 0.08 - torch.abs(comp_gap)
    comp_truth = logic(comp_margin)
    comp_penalty = torch.mean(vol_gate * (comp_gap ** 2))

    # Force correction to neural command under high volatility
    corrected_storage = (1.0 - vol_gate) * storage + vol_gate * expert_target
    corrected_action = torch.cat([gen, corrected_storage, v], dim=1)

    total_penalty = 2.0 * kcl_penalty + 1.5 * v_penalty + 1.5 * flow_penalty + 2.5 * comp_penalty

    # "Safety truth" summary
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
    }
    return total_penalty, corrected_action, diagnostics, truth_summary


def main():
    parser = argparse.ArgumentParser(description="Neuro-symbolic grid control demo")
    parser.add_argument("--epochs", type=int, default=NSConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=NSConfig.batch_size)
    parser.add_argument("--lr", type=float, default=NSConfig.lr)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output-dir", type=str, default=NSConfig.output_dir)
    args = parser.parse_args()

    cfg = NSConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=args.device, output_dir=args.output_dir)
    torch.manual_seed(cfg.seed)

    if cfg.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.device)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    }

    print("=== Neuro-Symbolic Grid Control Demo ===")
    print("Constraints: KCL, voltage limits, tie-line thermal limits, storage-renewable complement")
    print(
        f"device={device}, epochs={cfg.epochs}, batch_size={cfg.batch_size}, "
        f"v_range=[{cfg.v_min:.2f},{cfg.v_max:.2f}], tie_line_limit={cfg.tie_line_limit:.2f}, "
        f"volatility_threshold={cfg.volatility_threshold:.2f}, logic_temp={cfg.logic_temperature:.1f}"
    )

    for epoch in range(1, cfg.epochs + 1):
        state, load, ren, ren_vol = synthesize_grid_batch(cfg.batch_size, cfg.seed, epoch, device)
        raw_action = model(state)

        total_penalty, corrected_action, d, truth = build_symbolic_penalty(
            raw_action, load, ren, ren_vol, logic, cfg
        )

        # Add a small smoothness objective to avoid extreme commands
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

        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"Epoch {epoch:03d}/{cfg.epochs} | total={loss.item():.5f} | "
                f"kcl={d['kcl_penalty']:.5f} v={d['v_penalty']:.5f} flow={d['flow_penalty']:.5f} comp={d['comp_penalty']:.5f} | "
                f"truth(kcl/v/flow/comp)=({truth['kcl_truth']:.3f}/{truth['voltage_truth']:.3f}/{truth['flow_truth']:.3f}/{truth['complement_truth']:.3f})"
            )

    history_path = output_dir / "ns_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(f"Saved history: {history_path}")
    print("Done. Neuro-symbolic penalties provide direct safety-boundary correction gradients.")


if __name__ == "__main__":
    main()
