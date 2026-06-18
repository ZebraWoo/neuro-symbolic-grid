#!/usr/bin/env python3
"""Plot demo/techdoc_results/techdoc_history.json (unified tech-doc + multimodal)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _has_signal(values: list, eps: float = 1e-12) -> bool:
    if not values:
        return False
    arr = np.asarray(values, dtype=float)
    return bool(np.max(np.abs(arr)) > eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-file",
        type=str,
        default="demo/techdoc_results/techdoc_history.json",
    )
    parser.add_argument("--output-dir", type=str, default="demo/techdoc_results")
    args = parser.parse_args()
    outp = Path(args.output_dir)
    outp.mkdir(parents=True, exist_ok=True)

    with open(args.history_file, "r", encoding="utf-8") as f:
        h = json.load(f)

    branch = h.get("branch", "both")
    ep = np.arange(1, len(h["loss_total"]) + 1)
    show_ce = _has_signal(h.get("loss_ce", []))
    show_pred = _has_signal(h.get("loss_pred", []))

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle(f"Tech-doc unified demo (branch={branch})", fontsize=13)

    ax = axes[0, 0]
    ax.plot(ep, h["loss_total"], label="Total loss", linewidth=2, color="tab:red")
    if show_ce:
        ax.plot(ep, h["loss_ce"], label="CE (SNN-MLP)", linewidth=1.5)
    if show_pred:
        ax.plot(ep, h["loss_pred"], label="MSE pred (multimodal)", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    if show_pred and "rmse_pred" in h:
        ax.plot(ep, h["rmse_pred"], color="tab:green", linewidth=2, label="Load pred RMSE")
        ax.set_ylabel("RMSE")
        ax.set_title("Multimodal pretrain: load forecasting RMSE")
    elif show_ce:
        ax.plot(ep, h["mean_proximity"], color="tab:purple", linewidth=2)
        ax.set_ylabel("Mean proximity")
        ax.set_title("Risk operator: cosine proximity")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if show_ce and _has_signal(h.get("mean_proximity", [])):
        ax.plot(ep, h["mean_proximity"], color="tab:purple", linewidth=2)
        ax.set_title("Risk operator: mean proximity")
    elif show_pred:
        ax.plot(ep, h["loss_pred"], color="tab:olive", linewidth=2)
        ax.set_title("Multimodal load MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, h["kcl_penalty"], label="KCL")
    ax.plot(ep, h["v_penalty"], label="Voltage")
    ax.plot(ep, h["flow_penalty"], label="Flow")
    ax.plot(ep, h["comp_penalty"], label="Complement")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Penalty")
    ax.set_title("Neuro-symbolic penalties (unchanged)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.plot(ep, h["kcl_truth"], label="KCL truth")
    ax.plot(ep, h["voltage_truth"], label="Voltage truth")
    ax.plot(ep, h["flow_truth"], label="Flow truth")
    ax.plot(ep, h["complement_truth"], label="Complement truth")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Soft truth")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Logic neuron outputs")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.axis("off")
    lines = [
        f"Branch: {branch}",
        "Neuro-symbolic: KCL, voltage, flow, complement",
    ]
    if show_ce:
        lines.append("SNN-MLP: 11-d early fusion + CE + proximity")
    if show_pred:
        lines.append("Multimodal: load/renewable/irradiance/weather fusion")
        lines.append("Pretrain: next-step load MSE")
    if show_ce and show_pred:
        lines.append("Policy state: grid(7) + fusion embed(32)")
    ax.text(0.05, 0.95, "\n".join(lines), va="top", fontsize=10, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = outp / "techdoc_framework_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path} (branch={branch})")


if __name__ == "__main__":
    main()
