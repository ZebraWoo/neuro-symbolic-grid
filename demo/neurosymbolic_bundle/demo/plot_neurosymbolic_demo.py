#!/usr/bin/env python3
"""Plot unified joint-training results for the Neuro-Symbolic Grid Control Demo."""

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
                        default="demo/ns_results/ns_history.json")
    parser.add_argument("--output-dir", type=str,
                        default="demo/ns_results")
    args = parser.parse_args()

    with open(args.history_file, "r") as f:
        h = json.load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    branch = h.get("branch", "both")
    ep = np.arange(1, len(h["loss_total"]) + 1)
    show_ce = _has_signal(h.get("loss_ce", []))
    show_ctrl = _has_signal(h.get("kcl_penalty", []))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Neuro-Symbolic Unified Training — branch={branch} (Real PSML Data)",
                 fontsize=13)

    # (0,0) Training Loss
    ax = axes[0, 0]
    ax.plot(ep, h["loss_total"], "r-", lw=2, label="Total Loss")
    if show_ce:
        ax.plot(ep, h["loss_ce"], lw=1.5, label="CE (SNN-MLP)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (0,1) Proximity
    ax = axes[0, 1]
    if show_ce and _has_signal(h.get("mean_proximity", [])):
        ax.plot(ep, h["mean_proximity"], "purple", lw=2)
        ax.set_title("Risk Operator: Mean Proximity")
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Proximity"); ax.grid(alpha=0.3)

    # (1,0) Penalties
    ax = axes[1, 0]
    if show_ctrl:
        ax.plot(ep, h["kcl_penalty"], label="KCL")
        ax.plot(ep, h["v_penalty"], label="Voltage")
        ax.plot(ep, h["flow_penalty"], label="Flow")
        ax.plot(ep, h["comp_penalty"], label="Complement")
        ax.set_title("Neuro-Symbolic Penalties")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Penalty"); ax.grid(alpha=0.3)

    # (1,1) Truth values
    ax = axes[1, 1]
    if show_ctrl:
        ax.plot(ep, h["kcl_truth"], label="KCL Truth")
        ax.plot(ep, h["voltage_truth"], label="Voltage Truth")
        ax.plot(ep, h["flow_truth"], label="Flow Truth")
        ax.plot(ep, h["complement_truth"], label="Complement Truth")
        ax.set_ylim(0, 1); ax.set_title("Logic Neuron Outputs")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Soft Truth"); ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out / "ns_framework_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path} (branch={branch})")

    # ---- Separate penalty + truth plots (backward compatible) ----
    if show_ctrl:
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(ep, h["loss_total"], "k-", lw=2.5, label="Total Penalty")
        ax.plot(ep, h["kcl_penalty"], lw=1.8, label="KCL")
        ax.plot(ep, h["v_penalty"], lw=1.8, label="Voltage")
        ax.plot(ep, h["flow_penalty"], lw=1.8, label="Flow")
        ax.plot(ep, h["comp_penalty"], lw=1.8, label="Complement")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Penalty")
        ax.set_title("Penalty Decomposition"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(out / "ns_penalty_curve.png", dpi=150)
        plt.close()
        print(f"Saved: {out / 'ns_penalty_curve.png'}")

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(ep, h["kcl_truth"], lw=2, label="KCL Truth")
        ax.plot(ep, h["voltage_truth"], lw=2, label="Voltage Truth")
        ax.plot(ep, h["flow_truth"], lw=2, label="Flow Truth")
        ax.plot(ep, h["complement_truth"], lw=2, label="Complement Truth")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Soft Truth"); ax.set_ylim(0, 1)
        ax.set_title("Logic Neuron Truth Values"); ax.grid(alpha=0.3); ax.legend()
        plt.tight_layout(); plt.savefig(out / "ns_truth_curve.png", dpi=150)
        plt.close()
        print(f"Saved: {out / 'ns_truth_curve.png'}")

    # ---- Summary Report ----
    best = int(np.argmin(np.array(h["loss_total"])) + 1)
    drop = 0.0
    if h["loss_total"][0] != 0:
        drop = (h["loss_total"][0] - h["loss_total"][-1]) / h["loss_total"][0] * 100

    lines = [
        "Neuro-Symbolic Unified Training — Summary",
        "=" * 56, "",
        f"Branch: {branch}  |  Epochs: {len(ep)}", "",
        f"Total Loss: {h['loss_total'][0]:.4f} → {h['loss_total'][-1]:.4f}  ({drop:.1f}%)",
    ]
    if show_ce:
        lines.append(f"CE Loss:    {h['loss_ce'][0]:.4f} → {h['loss_ce'][-1]:.4f}")
    if _has_signal(h.get("mean_proximity", [])):
        lines.append(f"Proximity:  {h['mean_proximity'][0]:.3f} → {h['mean_proximity'][-1]:.3f}")
    if show_ctrl:
        lines += [
            "", "Final Penalties",
            f"  KCL:         {h['kcl_penalty'][-1]:.6f}",
            f"  Voltage:     {h['v_penalty'][-1]:.6f}",
            f"  Flow:        {h['flow_penalty'][-1]:.6f}",
            f"  Complement:  {h['comp_penalty'][-1]:.6f}",
        ]
    lines += ["", "Model: 3-layer SNN-MLP + ControlNet + 4 symbolic constraints",
              "Data:  Real PSML minute-level load & renewable"]

    fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=11.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9))
    plt.tight_layout(); plt.savefig(out / "summary_report.png", dpi=150)
    plt.close()
    print(f"Saved: {out / 'summary_report.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
