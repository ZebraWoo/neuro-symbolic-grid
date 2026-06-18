#!/usr/bin/env python3
"""Plot SNN-MLP anomaly classification demo results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-file", type=str,
                        default="demo/snn_results/snn_history.json")
    parser.add_argument("--output-dir", type=str,
                        default="demo/snn_results")
    args = parser.parse_args()

    with open(args.history_file, "r") as f:
        h = json.load(f)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(h["train_loss"]) + 1))

    # ---- Loss Curve ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, h["train_loss"], "o-", lw=2, label="Train Loss")
    ax.plot(epochs, h["val_loss"], "s-", lw=2, label="Validation Loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("SNN-MLP Demo — Loss Curves (Real PSML Data)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(out / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {out / 'loss_curve.png'}")

    # ---- Accuracy Curve ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, h["train_acc"], "o-", lw=2, label="Train Accuracy")
    ax.plot(epochs, h["val_acc"], "s-", lw=2, label="Validation Accuracy")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
    ax.set_title("SNN-MLP Demo — Accuracy Curves (Real PSML Data)")
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout(); plt.savefig(out / "accuracy_curve.png", dpi=150)
    plt.close()
    print(f"Saved: {out / 'accuracy_curve.png'}")

    # ---- Summary Report ----
    best_val_acc = max(h["val_acc"])
    best_epoch = h["val_acc"].index(best_val_acc) + 1
    best_val_loss = min(h["val_loss"])
    best_loss_epoch = h["val_loss"].index(best_val_loss) + 1

    train_drop = 0.0
    if h["train_loss"][0] != 0:
        train_drop = ((h["train_loss"][0] - h["train_loss"][-1])
                      / h["train_loss"][0] * 100)

    fig = plt.figure(figsize=(11, 7)); ax = fig.add_subplot(111)
    ax.axis("off")
    lines = [
        "SNN-MLP Anomaly Classification Demo — Summary",
        "=" * 56, "",
        f"Epochs: {len(epochs)}", "",
        "Loss Metrics",
        f"  Initial train loss: {h['train_loss'][0]:.6f}",
        f"  Final train loss:   {h['train_loss'][-1]:.6f}",
        f"  Train loss drop:    {train_drop:.2f}%",
        f"  Best val loss:      {best_val_loss:.6f} (Epoch {best_loss_epoch})",
        f"  Final val loss:     {h['val_loss'][-1]:.6f}", "",
        "Accuracy Metrics",
        f"  Initial train acc:  {h['train_acc'][0]:.4f}",
        f"  Final train acc:    {h['train_acc'][-1]:.4f}",
        f"  Initial val acc:    {h['val_acc'][0]:.4f}",
        f"  Final val acc:      {h['val_acc'][-1]:.4f}",
        f"  Best val acc:       {best_val_acc:.4f} (Epoch {best_epoch})", "",
        "Model: 3-layer SNN-MLP with LIF neurons + surrogate gradient",
        "Data:  Real PSML minute-level load & renewable",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            va="top", ha="left", fontsize=12, family="monospace",
            bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9))
    plt.tight_layout(); plt.savefig(out / "summary_report.png", dpi=150)
    plt.close()
    print(f"Saved: {out / 'summary_report.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
