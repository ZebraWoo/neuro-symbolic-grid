#!/usr/bin/env python3
"""
Plot results for neuro-symbolic grid control demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_ns_history(history_file: Path, output_dir: Path) -> None:
    with open(history_file, "r", encoding="utf-8") as f:
        h = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(h["total_penalty"]) + 1)

    # 1) Total + component penalties
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(epochs, h["total_penalty"], label="Total penalty", linewidth=2.5)
    ax.plot(epochs, h["kcl_penalty"], label="KCL penalty", linewidth=1.8)
    ax.plot(epochs, h["v_penalty"], label="Voltage penalty", linewidth=1.8)
    ax.plot(epochs, h["flow_penalty"], label="Flow penalty", linewidth=1.8)
    ax.plot(epochs, h["comp_penalty"], label="Complement penalty", linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Penalty")
    ax.set_title("Neuro-Symbolic Penalty Decomposition")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ns_penalty_curve.png", dpi=150)
    plt.close()

    # 2) Logic truth curves
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(epochs, h["kcl_truth"], label="KCL truth", linewidth=2.0)
    ax.plot(epochs, h["voltage_truth"], label="Voltage truth", linewidth=2.0)
    ax.plot(epochs, h["flow_truth"], label="Flow truth", linewidth=2.0)
    ax.plot(epochs, h["complement_truth"], label="Complement truth", linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Soft truth value")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Logic Neuron Truth Values")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "ns_truth_curve.png", dpi=150)
    plt.close()

    # 3) Summary report
    best_epoch = int(np.argmin(np.array(h["total_penalty"])) + 1)
    summary = [
        "Neuro-Symbolic Grid Demo Summary",
        "=" * 58,
        "",
        f"Epochs: {len(epochs)}",
        "",
        "Penalty Metrics",
        f"  - Initial total penalty: {h['total_penalty'][0]:.6f}",
        f"  - Final total penalty:   {h['total_penalty'][-1]:.6f}",
        f"  - Best total penalty:    {np.min(h['total_penalty']):.6f} (Epoch {best_epoch})",
        "",
        "Final Constraint Penalties",
        f"  - KCL penalty:           {h['kcl_penalty'][-1]:.6f}",
        f"  - Voltage penalty:       {h['v_penalty'][-1]:.6f}",
        f"  - Flow penalty:          {h['flow_penalty'][-1]:.6f}",
        f"  - Complement penalty:    {h['comp_penalty'][-1]:.6f}",
        "",
        "Final Logic Truth Values",
        f"  - KCL truth:             {h['kcl_truth'][-1]:.4f}",
        f"  - Voltage truth:         {h['voltage_truth'][-1]:.4f}",
        f"  - Flow truth:            {h['flow_truth'][-1]:.4f}",
        f"  - Complement truth:      {h['complement_truth'][-1]:.4f}",
        "",
        "Interpretation",
        "  - Lower penalties indicate safer control outputs.",
        "  - Higher truth values indicate better symbolic rule satisfaction.",
        "  - Gradients from rule violations are fed back into neural control.",
    ]

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.text(
        0.05,
        0.95,
        "\n".join(summary),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11.5,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "summary_report.png", dpi=150)
    plt.close()

    print(f"Saved plots to: {output_dir}")
    print("  - ns_penalty_curve.png")
    print("  - ns_truth_curve.png")
    print("  - summary_report.png")


def main():
    parser = argparse.ArgumentParser(description="Plot neuro-symbolic demo results")
    parser.add_argument("--history-file", type=str, default="demo/ns_results/ns_history.json")
    parser.add_argument("--output-dir", type=str, default="demo/ns_results")
    args = parser.parse_args()

    plot_ns_history(Path(args.history_file), Path(args.output_dir))


if __name__ == "__main__":
    main()
