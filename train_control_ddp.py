#!/usr/bin/env python3
"""
Distributed (DDP) training for PSML multimodal load forecasting.

Launch (8 GPUs on one node):
  torchrun --standalone --nproc_per_node=8 train_control_ddp.py \\
    --zones ERCOT_zone_1_ CAISO_zone_1_ MISO_zone_1_ \\
    --epochs 50 --batch-size 64

Effective global batch size = batch_size * world_size (per-GPU batch in args).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.multimodal_psml_dataset import (  # noqa: E402
    MODALITY_DIMS,
    load_psml_zone_frames,
    MultimodalPSMLDataset,
    multimodal_psml_collate,
)
from train_control import build_model, eval_epoch, train_epoch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return local_rank, dist.get_rank(), dist.get_world_size()
    raise RuntimeError(
        "DDP requires torchrun. Example:\n"
        "  torchrun --standalone --nproc_per_node=8 train_control_ddp.py ..."
    )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DDP PSML multimodal training")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable",
    )
    parser.add_argument("--zones", type=str, nargs="+", default=["ERCOT_zone_1_"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-rows-per-zone", type=int, default=None)
    parser.add_argument("--model-type", default="multimodal")
    parser.add_argument("--use-lif", action="store_true", help="Temporal LIF in spike blocks")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(
            "DDP world_size=%d | per-GPU batch=%d | global batch≈%d",
            world_size,
            args.batch_size,
            args.batch_size * world_size,
        )

    # Load data once per process (simple; for huge data consider rank-0 broadcast)
    zone_frames, zone_stats, load_col_idx = load_psml_zone_frames(
        args.data_root, args.zones, "zscore", args.max_rows_per_zone
    )
    shared = dict(
        data_root=args.data_root,
        zones=args.zones,
        seq_len=args.seq_len,
        stride=args.stride,
        zone_frames=zone_frames,
        zone_stats=zone_stats,
        load_col_idx=load_col_idx,
    )
    full = MultimodalPSMLDataset(**shared)
    n = len(full)
    import numpy as np

    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val_ratio))
    train_indices = [full.indices[i] for i in perm[n_val:]]
    val_indices = [full.indices[i] for i in perm[:n_val]]

    train_ds = MultimodalPSMLDataset(**shared, indices=train_indices)
    val_ds = MultimodalPSMLDataset(**shared, indices=val_indices)

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=multimodal_psml_collate,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, sampler=train_sampler, drop_last=True, **loader_kw
    )
    val_loader = torch.utils.data.DataLoader(val_ds, sampler=val_sampler, **loader_kw)

    model = build_model(args.model_type, args.seq_len, device, use_lif=args.use_lif)
    # LIF layers / confidence head may be unused when loss is MSE(pred) only
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )
    optimizer = optim.Adam(model.parameters(), lr=args.lr * world_size)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history = {
        "train_pred_mse": [],
        "val_rmse": [],
        "run_config": {
            "ddp": True,
            "world_size": world_size,
            "per_gpu_batch": args.batch_size,
            "global_batch": args.batch_size * world_size,
            "zones": args.zones,
            "epochs": args.epochs,
            "lr": args.lr,
            "use_lif": args.use_lif or args.model_type == "lif",
        },
    }
    best_val_rmse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        avg_loss, _, avg_pred = train_epoch(
            model, None, optimizer, train_loader, device, aux_weight=0.0
        )

        val_mse, val_rmse = eval_epoch(model, val_loader, device)
        # aggregate val mse across ranks
        t = torch.tensor([val_mse, val_rmse], device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        val_mse, val_rmse = t[0].item(), t[1].item()

        scheduler.step()

        if rank == 0:
            history["train_pred_mse"].append(avg_pred)
            history["val_rmse"].append(val_rmse)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                logger.info(
                    "Epoch %d/%d | pred_mse=%.5f | val_rmse=%.5f",
                    epoch,
                    args.epochs,
                    avg_pred,
                    val_rmse,
                )
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = {k: v.cpu() for k, v in model.module.state_dict().items()}

    if rank == 0:
        payload = {
            "model_state_dict": model.module.state_dict(),
            "modalities": MODALITY_DIMS,
            "seq_len": args.seq_len,
            "zones": args.zones,
            "ddp_world_size": world_size,
        }
        tag = "_lif" if (args.use_lif or args.model_type == "lif") else ""
        torch.save(payload, Path(args.checkpoint_dir) / f"control_model_psml_ddp{tag}.pth")
        if best_state is not None:
            torch.save(
                {**payload, "model_state_dict": best_state},
                Path(args.checkpoint_dir) / f"control_model_psml_ddp{tag}_best.pth",
            )
        history["run_config"]["best_val_rmse"] = best_val_rmse
        hist_path = Path(args.output_dir) / f"training_history_psml_ddp{tag}.json"
        with open(hist_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        logger.info("Saved DDP checkpoints and %s", hist_path)
        try:
            from plot_control_psml import plot_history

            plot_history(hist_path, Path(args.output_dir))
        except Exception as e:
            logger.warning("Plot failed: %s", e)

    cleanup_distributed()


if __name__ == "__main__":
    main()
