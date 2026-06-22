#!/usr/bin/env python3
"""
Plot training results for the 3-layer SNN-MLP demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def plot_history(history_path: Path, output_dir: Path) -> None:
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(history["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="s", linewidth=2, label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("SNN-MLP Demo Loss Curves")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Train Accuracy")
    ax.plot(epochs, history["val_acc"], marker="s", linewidth=2, label="Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("SNN-MLP Demo Accuracy Curves")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=150)
    plt.close()

    best_val_acc = max(history["val_acc"])
    best_val_acc_epoch = history["val_acc"].index(best_val_acc) + 1
    best_val_loss = min(history["val_loss"])
    best_val_loss_epoch = history["val_loss"].index(best_val_loss) + 1

    train_loss_drop = 0.0
    val_loss_drop = 0.0
    if history["train_loss"][0] != 0:
        train_loss_drop = ((history["train_loss"][0] - history["train_loss"][-1]) / history["train_loss"][0]) * 100.0
    if history["val_loss"][0] != 0:
        val_loss_drop = ((history["val_loss"][0] - history["val_loss"][-1]) / history["val_loss"][0]) * 100.0

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111)
    ax.axis("off")

    summary_text = "\n".join(
        [
            "SNN-MLP Demo Summary Report",
            "=" * 56,
            "",
            f"Epochs: {len(epochs)}",
            "",
            "Loss Metrics",
            f"  - Initial train loss: {history['train_loss'][0]:.6f}",
            f"  - Final train loss:   {history['train_loss'][-1]:.6f}",
            f"  - Train loss drop:    {train_loss_drop:.2f}%",
            f"  - Best val loss:      {best_val_loss:.6f} (Epoch {best_val_loss_epoch})",
            f"  - Final val loss:     {history['val_loss'][-1]:.6f}",
            f"  - Val loss drop:      {val_loss_drop:.2f}%",
            "",
            "Accuracy Metrics",
            f"  - Initial train acc:  {history['train_acc'][0]:.4f}",
            f"  - Final train acc:    {history['train_acc'][-1]:.4f}",
            f"  - Initial val acc:    {history['val_acc'][0]:.4f}",
            f"  - Final val acc:      {history['val_acc'][-1]:.4f}",
            f"  - Best val acc:       {best_val_acc:.4f} (Epoch {best_val_acc_epoch})",
            "",
            "Takeaway",
            "  - This demo shows that the 3-layer SNN-MLP can learn",
            "    anomaly-like patterns from real power sequence windows.",
        ]
    )

    ax.text(
        0.05,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="whitesmoke", alpha=0.9),
    )
    plt.tight_layout()
    plt.savefig(output_dir / "summary_report.png", dpi=150)
    plt.close()

    print(f"Saved plots to: {output_dir}")
    print("  - loss_curve.png")
    print("  - accuracy_curve.png")
    print("  - summary_report.png")


def main():
    parser = argparse.ArgumentParser(description="Plot SNN-MLP demo training history")
    parser.add_argument("--history-file", type=str, default="demo/results/history.json")
    parser.add_argument("--output-dir", type=str, default="demo/results")
    args = parser.parse_args()

    plot_history(Path(args.history_file), Path(args.output_dir))


if __name__ == "__main__":
    main()
