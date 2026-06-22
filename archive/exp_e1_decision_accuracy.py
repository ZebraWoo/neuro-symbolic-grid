#!/usr/bin/env python3
"""
Experiment E1: Decision Intent Accuracy — all baselines vs Ours.

Usage:
  # Single baseline
  python experiments/exp_e1_decision_accuracy.py --model lstm --epochs 50

  # All baselines (sequential)
  python experiments/exp_e1_decision_accuracy.py --model all --epochs 50

  # Quick test
  python experiments/exp_e1_decision_accuracy.py --model lstm --epochs 5 --zones-per-split 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import default_cfg
from experiments.models.baseline_models import (
    LSTMBaseline, TransformerBaseline, TCNBaseline,
    SNNLIFBaseline, SNNIzhikevichBaseline, flatten_modalities,
)
from experiments.trainers.base_trainer import DecisionTrainer
from experiments.eval_utils import compute_decision_metrics
from src.data.multimodal_psml_dataset import (
    MODALITY_DIMS,
    multimodal_psml_collate,
    load_psml_zone_frames,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


BASELINE_BUILDERS = {
    "lstm": lambda: LSTMBaseline(),
    "transformer": lambda: TransformerBaseline(),
    "tcn": lambda: TCNBaseline(),
    "snn_lif": lambda: SNNLIFBaseline(),
    "snn_izh": lambda: SNNIzhikevichBaseline(),
}


class BaselineWrapper(torch.nn.Module):
    """Wraps a baseline to accept multimodal dict input."""

    def __init__(self, baseline):
        super().__init__()
        self.baseline = baseline

    def forward(self, modalities_data):
        x = flatten_modalities(modalities_data, MODALITY_DIMS)
        return self.baseline(x)


def build_dataloaders(data_root, zones, seq_len, stride, batch_size, val_ratio, num_workers, max_rows=None):
    """Build train/val dataloaders with labels."""
    zone_frames, _, _ = load_psml_zone_frames(
        data_root, zones, max_rows_per_zone=max_rows, normalize="zscore",
    )

    all_windows = []
    for frames in zone_frames.values():
        n = len(frames)
        n_windows = max(0, (n - seq_len) // stride + 1)
        for i in range(n_windows):
            start = i * stride
            all_windows.append(frames[start:start + seq_len])

    all_data = np.stack(all_windows, axis=0)
    rng = np.random.RandomState(42)
    indices = rng.permutation(len(all_data))
    train_size = int(len(all_data) * (1 - val_ratio))

    train_data = all_data[indices[:train_size]]
    val_data = all_data[indices[train_size:]]

    # Generate labels
    from experiments.label_decision_intents import (
        DecisionLabelConfig, LabeledDataset, _generate_labels_for_data,
    )

    label_cfg = DecisionLabelConfig()
    train_labels = _generate_labels_for_data(train_data, label_cfg, seq_len)
    val_labels = _generate_labels_for_data(val_data, label_cfg, seq_len)

    train_ds = LabeledDataset(train_data, train_labels, seq_len=seq_len)
    val_ds = LabeledDataset(val_data, val_labels, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=multimodal_psml_collate, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=multimodal_psml_collate, num_workers=num_workers)

    return train_loader, val_loader


def run_baseline(model_name, args, device):
    logger.info(f"\n{'='*60}\nTraining {model_name}\n{'='*60}")

    zones = args.zones or default_cfg.data.train_zones[:args.zones_per_split] if args.zones_per_split > 0 else default_cfg.data.train_zones[:3]
    val_zones = default_cfg.data.val_zones[:2]

    train_loader, val_loader = build_dataloaders(
        args.data_root, zones + val_zones, args.seq_len, args.stride,
        args.batch_size, args.val_ratio, args.num_workers, args.max_rows,
    )

    baseline = BASELINE_BUILDERS[model_name]()
    model = BaselineWrapper(baseline)

    trainer = DecisionTrainer(
        model=model, device=device, loss_mode="bce_only",
        lr=args.lr,
        weight_decay=default_cfg.training.weight_decay,
        gradient_clip=default_cfg.training.gradient_clip,
        seed=default_cfg.training.seed,
    )

    history = trainer.fit(
        train_loader, val_loader,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        model_tag=f"e1_{model_name}",
    )

    best_f1 = max(history["val_macro_f1"]) if history["val_macro_f1"] else 0.0
    logger.info(f"{model_name}: Best val Macro F1 = {best_f1:.4f}")
    return {"model": model_name, "best_macro_f1": best_f1, "history": history}


def main():
    parser = argparse.ArgumentParser(description="E1: Decision Intent Accuracy")
    parser.add_argument("--model", default="all", choices=["all", "lstm", "transformer", "tcn", "snn_lif", "snn_izh"])
    parser.add_argument("--data-root", default=default_cfg.data.root)
    parser.add_argument("--zones", nargs="+", default=None)
    parser.add_argument("--zones-per-split", type=int, default=3)
    parser.add_argument("--seq-len", type=int, default=default_cfg.data.seq_len)
    parser.add_argument("--stride", type=int, default=default_cfg.data.stride)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device(args.device)

    models_to_run = list(BASELINE_BUILDERS.keys()) if args.model == "all" else [args.model]

    results = {}
    for model_name in models_to_run:
        result = run_baseline(model_name, args, device)
        results[model_name] = result

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("E1 Results Summary")
    logger.info("=" * 60)
    for model_name, r in results.items():
        logger.info(f"  {model_name:20s}: Macro F1 = {r['best_macro_f1']:.4f}")

    summary_path = Path(args.output_dir) / "e1_summary.json"
    with open(summary_path, "w") as f:
        json.dump({k: {"best_macro_f1": v["best_macro_f1"]} for k, v in results.items()}, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
