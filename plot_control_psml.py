#!/usr/bin/env python3
"""Plot training curves from outputs/training_history_psml.json (or --history-file)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_history(history_path: Path, output_dir: Path) -> None:
    with open(history_path, "r", encoding="utf-8") as f:
        h = json.load(f)

    epochs = np.arange(1, len(h["train_pred_mse"]) + 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(epochs, h["train_pred_mse"], "b-o", markersize=4, label="Train MSE (load pred)")
    if h.get("val_mse"):
        ax.plot(epochs, h["val_mse"], "r-s", markersize=4, label="Val MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE (normalized load)")
    ax.set_title("PSML multimodal load forecasting")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1]
    if h.get("val_rmse"):
        ax.plot(epochs, h["val_rmse"], "g-^", markersize=4, label="Val RMSE")
    ax.plot(epochs, np.sqrt(h["train_pred_mse"]), "b--", alpha=0.7, label="Train RMSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE (normalized load)")
    ax.set_title("RMSE curves")
    ax.grid(True, alpha=0.3)
    ax.legend()

    meta = h.get("run_config", {})
    if meta:
        fig.suptitle(
            f"zones={meta.get('zones')} | seq_len={meta.get('seq_len')} | "
            f"epochs={len(epochs)} | best_val_rmse={meta.get('best_val_rmse', 'n/a')}",
            fontsize=10,
            y=1.02,
        )

    plt.tight_layout()
    out_png = output_dir / "psml_load_pred_curves.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-file",
        type=str,
        default="outputs/training_history_psml.json",
    )
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    plot_history(Path(args.history_file), Path(args.output_dir))


if __name__ == "__main__":
    main()
