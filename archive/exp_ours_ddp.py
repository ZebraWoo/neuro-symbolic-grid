#!/usr/bin/env python3
"""
DDP (multi-GPU) version of exp_ours_full.py. Launch with torchrun.

Usage:
  torchrun --standalone --nproc_per_node=4 experiments/exp_ours_ddp.py --epochs 50

  # Quick test
  torchrun --standalone --nproc_per_node=2 experiments/exp_ours_ddp.py --epochs 5 --zones-per-split 2
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from experiments.config import default_cfg
from experiments.models.neuro_symbolic_model import NeuroSymbolicDecisionModel
from experiments.losses.symbolic_loss import SymbolicRuleLoss
from experiments.losses.physics_loss import PhysicsConstraintLoss
from src.data.multimodal_psml_dataset import MODALITY_DIMS, multimodal_psml_collate, load_psml_zone_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


class DDPTrainer:
    """Simplified trainer for DDP — each GPU runs its own train/val loop."""

    def __init__(self, model, rank, local_rank, lr, weight_decay, gradient_clip, seed):
        self.rank = rank
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.model = model.to(self.device)
        self.model = DDP(self.model, device_ids=[local_rank], find_unused_parameters=True)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=50)
        self.gradient_clip = gradient_clip
        torch.manual_seed(seed + rank)

    def train_epoch(self, dataloader, epoch, symbolic_loss_fn, physics_loss_fn, loss_mode):
        self.model.train()
        total_loss = 0.0
        total_bce = 0.0
        total_sym = 0.0
        total_phys = 0.0
        n = 0
        t0 = time.time()
        dataloader.sampler.set_epoch(epoch)

        for i, (modalities, labels) in enumerate(dataloader):
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            labels = labels.to(self.device)

            output = self.model(modalities)
            logits = output["decision_logits"]
            loss_bce = self.bce_loss(logits, labels)
            loss = loss_bce

            if loss_mode == "neuro_symbolic" and symbolic_loss_fn is not None:
                features, soc = self._extract_features(modalities, labels)
                sym_loss, _ = symbolic_loss_fn(logits, features, soc)
                loss = loss + sym_loss
                total_sym += sym_loss.item()
                if physics_loss_fn is not None:
                    phys_loss, _ = physics_loss_fn(logits, features, None, soc)
                    loss = loss + phys_loss
                    total_phys += phys_loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            if self.gradient_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
            self.optimizer.step()

            total_loss += loss.item()
            total_bce += loss_bce.item()
            n += 1

            if self.rank == 0 and (i == 0 or (i + 1) % 20 == 0):
                logger.info("[R0] batch %3d/%d | loss=%.4f bce=%.4f | %.1fs",
                            i + 1, len(dataloader), loss.item(), loss_bce.item(), time.time() - t0)

        n = max(n, 1)
        return {"loss": total_loss / n, "loss_bce": total_bce / n,
                "loss_sym": total_sym / n, "loss_phys": total_phys / n}

    @torch.no_grad()
    def evaluate(self, dataloader):
        self.model.eval()
        # Gather predictions from all GPUs
        all_logits = []
        all_labels = []
        for modalities, labels in dataloader:
            modalities = {k: v.to(self.device) for k, v in modalities.items()}
            output = self.model(modalities)
            all_logits.append(output["decision_logits"])
            all_labels.append(labels.to(self.device))

        local_logits = torch.cat(all_logits, dim=0)
        local_labels = torch.cat(all_labels, dim=0)

        # Gather across GPUs
        gathered_logits = [torch.zeros_like(local_logits) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(local_labels) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_logits, local_logits)
        dist.all_gather(gathered_labels, local_labels)

        all_l = torch.cat(gathered_logits, dim=0)
        all_t = torch.cat(gathered_labels, dim=0)

        probs = torch.sigmoid(all_l)
        preds = (probs > 0.5).float()
        per_class_f1 = []
        for i in range(all_t.shape[1]):
            tp = ((preds[:, i] == 1) & (all_t[:, i] == 1)).sum().item()
            fp = ((preds[:, i] == 1) & (all_t[:, i] == 0)).sum().item()
            fn = ((preds[:, i] == 0) & (all_t[:, i] == 1)).sum().item()
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            per_class_f1.append(f1)

        macro_f1 = float(np.mean(per_class_f1))
        hamming = (preds != all_t).float().mean().item()
        return {"macro_f1": macro_f1, "hamming": hamming}

    @staticmethod
    def _extract_features(modalities, labels):
        B = labels.shape[0]
        device = labels.device
        soc = 0.5 + 0.3 * (labels[:, 1] - labels[:, 2])
        soc = torch.clamp(soc, 0.05, 0.95)
        features = {
            "delta_renewable": torch.zeros(B, device=device),
            "delta_load": torch.zeros(B, device=device),
            "delta_net_load": torch.zeros(B, device=device),
            "delta_power": torch.zeros(B, device=device),
            "wind_speed": torch.zeros(B, device=device),
            "wind_threshold": torch.tensor(10.0, device=device),
            "soc_min": torch.tensor(0.10, device=device),
            "soc_max": torch.tensor(0.90, device=device),
            "ramp_limit": torch.tensor(0.15, device=device),
        }
        return features, soc


def main():
    rank, world_size, local_rank = setup_distributed()
    logger.info("[R%d] DDP initialized: rank=%d/%d, GPU=%d", rank, rank, world_size, local_rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=default_cfg.data.root)
    parser.add_argument("--zones-per-split", type=int, default=0)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tag", default="ours_ddp")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-symbolic", action="store_true")
    parser.add_argument("--no-physics", action="store_true")
    parser.add_argument("--no-multi-comp", action="store_true")
    parser.add_argument("--no-spike", action="store_true")
    args = parser.parse_args()

    # --- Data loading (rank 0 loads, then broadcast to all) ---
    zones = default_cfg.data.train_zones
    if args.zones_per_split > 0:
        zones = zones[:args.zones_per_split]

    from experiments.label_decision_intents import (
        DecisionLabelConfig, LabeledDataset, _generate_labels_for_data,
    )

    if rank == 0:
        logger.info("[R0] Loading %d zones...", len(zones))
        zone_frames, _, _ = load_psml_zone_frames(
            args.data_root, zones, max_rows_per_zone=args.max_rows, normalize="zscore")

        all_windows = []
        for frames in zone_frames.values():
            n = len(frames)
            n_windows = max(0, (n - args.seq_len) // args.stride + 1)
            for i in range(n_windows):
                all_windows.append(frames[i * args.stride : i * args.stride + args.seq_len])
        all_data = np.stack(all_windows, axis=0)

        rng = np.random.RandomState(42)
        indices = rng.permutation(len(all_data))
        train_size = int(len(all_data) * (1 - args.val_ratio))
        train_data = all_data[indices[:train_size]]
        val_data = all_data[indices[train_size:]]

        label_cfg = DecisionLabelConfig()
        train_labels = _generate_labels_for_data(train_data, label_cfg, args.seq_len)
        val_labels = _generate_labels_for_data(val_data, label_cfg, args.seq_len)

        logger.info("[R0] Train: %d, Val: %d", len(train_data), len(val_data))
        for i, name in enumerate(default_cfg.model.intent_names):
            logger.info("[R0]   %s: %.1f%%", name, 100.0 * train_labels[:, i].mean())

        # Package for broadcast
        data_package = [train_data, val_data, train_labels, val_labels]
    else:
        data_package = [None, None, None, None]

    # Broadcast to all ranks
    dist.broadcast_object_list(data_package, src=0)
    train_data, val_data, train_labels, val_labels = data_package

    if rank != 0:
        logger.info("[R%d] Received data: train=%d val=%d", rank, len(train_data), len(val_data))

    train_ds = LabeledDataset(train_data, train_labels, seq_len=args.seq_len)
    val_ds = LabeledDataset(val_data, val_labels, seq_len=args.seq_len)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler,
                              collate_fn=multimodal_psml_collate, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, sampler=val_sampler,
                            collate_fn=multimodal_psml_collate, num_workers=args.num_workers)
    if rank == 0:
        logger.info("Train batches/GPU: %d", len(train_loader))

    # --- Model ---
    model = NeuroSymbolicDecisionModel(
        modality_dims=MODALITY_DIMS,
        hidden_dim=64, embedding_dim=32, num_blocks=4, num_heads=4,
        num_decision_intents=5, seq_len=args.seq_len, dropout=0.1,
        use_lif=not args.no_spike, use_multi_comp=not args.no_multi_comp,
    )
    if rank == 0:
        logger.info("Model: %.1fM params", sum(p.numel() for p in model.parameters()) / 1e6)

    loss_mode = "bce_only"
    sym_fn, phys_fn = None, None
    if not args.no_symbolic:
        sym_fn = SymbolicRuleLoss(temperature=20.0, lambda_symbolic=0.15).to(local_rank)
        loss_mode = "neuro_symbolic"
    if not args.no_physics:
        phys_fn = PhysicsConstraintLoss(lambda_physics=0.25).to(local_rank)
        loss_mode = "neuro_symbolic"

    trainer = DDPTrainer(model, rank, local_rank, args.lr, 1e-5, 1.0, 42)

    # --- Training ---
    history = {"train_loss": [], "val_macro_f1": [], "val_hamming": []}
    best_f1 = 0.0
    best_state = None

    if rank == 0:
        logger.info("Starting DDP training: %d epochs × %d batches × %d GPUs",
                    args.epochs, len(train_loader), world_size)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = trainer.train_epoch(train_loader, epoch, sym_fn, phys_fn, loss_mode)
        val_metrics = trainer.evaluate(val_loader)
        trainer.scheduler.step()

        if rank == 0:
            history["train_loss"].append(train_metrics["loss"])
            history["val_macro_f1"].append(val_metrics["macro_f1"])
            history["val_hamming"].append(val_metrics["hamming"])

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                best_state = {k: v.cpu().clone() for k, v in trainer.model.module.state_dict().items()}

            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                logger.info(
                    f"[R0] Epoch {epoch:3d}/{args.epochs} | "
                    f"Loss={train_metrics['loss']:.4f} "
                    f"BCE={train_metrics['loss_bce']:.4f} | "
                    f"Val F1_m={val_metrics['macro_f1']:.3f} "
                    f"Hamm={val_metrics['hamming']:.3f} | "
                    f"{time.time()-t0:.0f}s"
                )

    # Save (rank 0 only)
    if rank == 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        ckpt_path = Path(args.checkpoint_dir) / f"{args.tag}.pth"
        torch.save({"model_state_dict": trainer.model.module.state_dict(), "best_f1": best_f1}, ckpt_path)
        if best_state:
            best_path = Path(args.checkpoint_dir) / f"{args.tag}_best.pth"
            torch.save({"model_state_dict": best_state, "best_f1": best_f1}, best_path)
        with open(Path(args.output_dir) / f"history_{args.tag}.json", "w") as f:
            json.dump(history, f, indent=2)
        logger.info("Saved: %s (best F1=%.4f)", ckpt_path, best_f1)

    cleanup()


if __name__ == "__main__":
    main()
