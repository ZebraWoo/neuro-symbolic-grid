#!/usr/bin/env python3
"""
Train the full Neuro-Symbolic Decision Model (Ours).

Usage:
  # Quick test
  python experiments/exp_ours_full.py --epochs 5 --zones-per-split 2

  # Full training
  python experiments/exp_ours_full.py --epochs 50 --batch-size 32

  # Ablation: no symbolic
  python experiments/exp_ours_full.py --no-symbolic --tag ours_no_sym

  # Ablation: no physics
  python experiments/exp_ours_full.py --no-physics --tag ours_no_phys

  # Ablation: no closed-loop
  python experiments/exp_ours_full.py --no-closed-loop --tag ours_no_cl

  # Ablation: single-compartment SNN
  python experiments/exp_ours_full.py --no-multi-comp --tag ours_single_comp

  # Ablation: no spike (MLP instead of SNN)
  python experiments/exp_ours_full.py --no-spike --tag ours_no_spike
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import default_cfg
from experiments.models.neuro_symbolic_model import NeuroSymbolicDecisionModel
from experiments.losses.symbolic_loss import SymbolicRuleLoss
from experiments.losses.physics_loss import PhysicsConstraintLoss
from experiments.trainers.base_trainer import DecisionTrainer
from src.data.multimodal_psml_dataset import (
    MODALITY_DIMS,
    multimodal_psml_collate,
    load_psml_zone_frames,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train Ours full model")
    parser.add_argument("--data-root", default=default_cfg.data.root)
    parser.add_argument("--zones", nargs="+", default=None)
    parser.add_argument("--zones-per-split", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=default_cfg.data.seq_len)
    parser.add_argument("--stride", type=int, default=default_cfg.data.stride)
    parser.add_argument("--epochs", type=int, default=default_cfg.training.epochs)
    parser.add_argument("--batch-size", type=int, default=default_cfg.training.batch_size)
    parser.add_argument("--lr", type=float, default=default_cfg.training.lr)
    parser.add_argument("--device", default=default_cfg.training.device)
    parser.add_argument("--tag", default="ours_full")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Ablation flags
    parser.add_argument("--no-symbolic", action="store_true")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-closed-loop", action="store_true")
    parser.add_argument("--no-multi-comp", action="store_true")
    parser.add_argument("--no-spike", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Zones
    zones = args.zones or default_cfg.data.train_zones
    if args.zones_per_split > 0:
        zones = zones[:args.zones_per_split]

    logger.info("Loading data from %d zones...", len(zones))

    # Build dataloaders
    # Load frames
    zone_frames, zone_stats, _ = load_psml_zone_frames(
        args.data_root, zones, max_rows_per_zone=args.max_rows, normalize="zscore",
    )

    all_windows = []
    for zn, frames in zone_frames.items():
        n = len(frames)
        n_windows = max(0, (n - args.seq_len) // args.stride + 1)
        logger.info("  %s: %d rows → %d windows", zn, n, n_windows)
        for i in range(n_windows):
            start = i * args.stride
            all_windows.append(frames[start:start + args.seq_len])

    all_data = np.stack(all_windows, axis=0)
    logger.info("Total windows: %d", len(all_data))

    # Shuffle & split
    rng = np.random.RandomState(default_cfg.training.seed)
    indices = rng.permutation(len(all_data))
    train_size = int(len(all_data) * (1 - args.val_ratio))
    train_data = all_data[indices[:train_size]]
    val_data = all_data[indices[train_size:]]
    logger.info("Train windows: %d, Val windows: %d", len(train_data), len(val_data))

    # Generate labels
    from experiments.label_decision_intents import (
        DecisionLabelConfig, LabeledDataset, _generate_labels_for_data,
    )
    label_cfg = DecisionLabelConfig()

    train_labels = _generate_labels_for_data(train_data, label_cfg, args.seq_len)
    val_labels = _generate_labels_for_data(val_data, label_cfg, args.seq_len)

    logger.info("Label distributions (train):")
    for i, name in enumerate(default_cfg.model.intent_names):
        pct = 100.0 * train_labels[:, i].mean()
        logger.info("  %s: %.1f%%", name, pct)

    # Create datasets
    train_dataset = LabeledDataset(train_data, train_labels, seq_len=args.seq_len)
    val_dataset = LabeledDataset(val_data, val_labels, seq_len=args.seq_len)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=multimodal_psml_collate, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=multimodal_psml_collate, num_workers=args.num_workers,
    )

    # Build model
    model = NeuroSymbolicDecisionModel(
        modality_dims=MODALITY_DIMS,
        hidden_dim=default_cfg.model.hidden_dim,
        embedding_dim=default_cfg.model.embedding_dim,
        num_blocks=default_cfg.model.num_blocks,
        num_heads=default_cfg.model.num_heads,
        num_decision_intents=default_cfg.model.num_decision_intents,
        seq_len=args.seq_len,
        dropout=default_cfg.model.dropout,
        use_lif=not args.no_spike,
        use_multi_comp=not args.no_multi_comp,
        soma_neuron_type=default_cfg.model.soma_neuron_type,
    )
    logger.info("Model: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    # Build loss modules
    loss_mode = "bce_only"
    symbolic_loss_fn = None
    physics_loss_fn = None

    if not args.no_symbolic:
        symbolic_loss_fn = SymbolicRuleLoss(
            temperature=default_cfg.symbolic.temperature,
            rule_weights=default_cfg.symbolic.rule_weights,
            lambda_symbolic=default_cfg.symbolic.lambda_symbolic,
        )
        loss_mode = "neuro_symbolic"

    if not args.no_physics:
        physics_loss_fn = PhysicsConstraintLoss(
            ramp_limit=default_cfg.physics.ramp_limit,
            soc_min=default_cfg.physics.soc_min,
            soc_max=default_cfg.physics.soc_max,
            balance_tolerance=default_cfg.physics.balance_tolerance,
            curtail_max=default_cfg.physics.curtail_max,
            w_ramp=default_cfg.physics.w_ramp,
            w_soc=default_cfg.physics.w_soc,
            w_balance=default_cfg.physics.w_balance,
            w_curtail=default_cfg.physics.w_curtail,
            lambda_physics=default_cfg.physics.lambda_physics,
        )
        loss_mode = "neuro_symbolic"

    logger.info("Loss mode: %s | Symbolic: %s | Physics: %s | Multi-comp: %s | Spike: %s",
                loss_mode, not args.no_symbolic, not args.no_physics,
                not args.no_multi_comp, not args.no_spike)

    # Train
    trainer = DecisionTrainer(
        model=model,
        device=device,
        loss_mode=loss_mode,
        symbolic_loss_fn=symbolic_loss_fn,
        physics_loss_fn=physics_loss_fn,
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
        model_tag=args.tag,
    )

    logger.info("Done. Best macro F1: %.4f", max(history["val_macro_f1"]) if history["val_macro_f1"] else 0)


if __name__ == "__main__":
    main()
