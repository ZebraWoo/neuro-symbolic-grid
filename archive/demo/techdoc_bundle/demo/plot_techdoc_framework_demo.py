#!/usr/bin/env python3
"""Plot unified tech-doc + multimodal demo results (real PSML data)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _has_signal(values, eps=1e-12):
    if not values:
        return False
    return bool(np.max(np.abs(np.asarray(values, dtype=float))) > eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-file", type=str,
                        default="demo/techdoc_results/techdoc_history.json")
    parser.add_argument("--output-dir", type=str,
                        default="demo/techdoc_results")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(args.history_file, "r") as f:
        h = json.load(f)

    branch = h.get("branch", "both")
    ep = np.arange(1, len(h["loss_total"]) + 1)
    show_ce = _has_signal(h.get("loss_ce", []))
    show_pred = _has_signal(h.get("loss_pred", []))

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    fig.suptitle(f"TechDoc Unified Framework — branch={branch} (Real PSML Data)",
                 fontsize=13)

    # (0,0) Loss
    ax = axes[0, 0]
    ax.plot(ep, h["loss_total"], "r-", lw=2, label="Total Loss")
    if show_ce:
        ax.plot(ep, h["loss_ce"], lw=1.5, label="CE (SNN-MLP)")
    if show_pred:
        ax.plot(ep, h["loss_pred"], lw=1.5, label="MSE (Multimodal)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (0,1) RMSE / Proximity
    ax = axes[0, 1]
    if show_pred and "rmse_pred" in h:
        ax.plot(ep, h["rmse_pred"], "g-", lw=2, label="Load Pred RMSE")
        ax.set_ylabel("RMSE"); ax.set_title("Multimodal: Next-Step Load RMSE")
    elif show_ce and _has_signal(h.get("mean_proximity", [])):
        ax.plot(ep, h["mean_proximity"], "purple", lw=2, label="Proximity")
        ax.set_ylabel("Proximity"); ax.set_title("Risk Operator: Cosine Proximity")
    ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (1,0) Proximity / MSE
    ax = axes[1, 0]
    if show_ce and _has_signal(h.get("mean_proximity", [])):
        ax.plot(ep, h["mean_proximity"], "purple", lw=2)
        ax.set_title("Risk Operator: Mean Proximity")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Value"); ax.grid(alpha=0.3)

    # (1,1) Penalties
    ax = axes[1, 1]
    ax.plot(ep, h["kcl_penalty"], label="KCL")
    ax.plot(ep, h["v_penalty"], label="Voltage")
    ax.plot(ep, h["flow_penalty"], label="Flow")
    ax.plot(ep, h["comp_penalty"], label="Complement")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Penalty")
    ax.set_title("Neuro-Symbolic Penalties"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2,0) Truth values
    ax = axes[2, 0]
    ax.plot(ep, h["kcl_truth"], label="KCL Truth")
    ax.plot(ep, h["voltage_truth"], label="Voltage Truth")
    ax.plot(ep, h["flow_truth"], label="Flow Truth")
    ax.plot(ep, h["complement_truth"], label="Complement Truth")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Soft Truth"); ax.set_ylim(0, 1)
    ax.set_title("Logic Neuron Outputs"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (2,1) Summary text
    ax = axes[2, 1]; ax.axis("off")
    lines = [f"Branch: {branch}  |  Neuro-symbolic: KCL/V/Flow/Complement"]
    if show_ce:
        lines.append(f"SNN-MLP: 11-d, CE+Proximity")
    if show_pred:
        lines.append(f"Multimodal: load(1)+ren(2)+irr(4)+weather(4), next-step load MSE")
    lines += ["", f"Total Loss: {h['loss_total'][0]:.4f} -> {h['loss_total'][-1]:.4f}"]
    if show_ce:
        lines.append(f"CE Loss:    {h['loss_ce'][0]:.4f} -> {h['loss_ce'][-1]:.4f}")
    if show_pred and "rmse_pred" in h:
        lines.append(f"Pred RMSE:  {h['rmse_pred'][0]:.4f} -> {h['rmse_pred'][-1]:.4f}")
    if _has_signal(h.get("mean_proximity", [])):
        lines.append(f"Proximity:  {h['mean_proximity'][0]:.3f} -> {h['mean_proximity'][-1]:.3f}")
    lines += ["", f"Flow Pen: {h['flow_penalty'][0]:.3f} -> {h['flow_penalty'][-1]:.3f}"]
    lines += [f"Flow Truth: {h['flow_truth'][0]:.3f} -> {h['flow_truth'][-1]:.3f}"]
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / "techdoc_framework_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path} (branch={branch})")


if __name__ == "__main__":
    main()
