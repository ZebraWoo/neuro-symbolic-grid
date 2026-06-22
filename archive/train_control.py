#!/usr/bin/env python3
"""
Multimodal grid-control pretraining on PSML minute-level data.

Target: predict next-step load_power (normalized) from multimodal window.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.control.multimodal_control_network import ControlPretrainingLoss, MultimodalControlNetwork
from src.data.multimodal_psml_dataset import MODALITY_DIMS, create_multimodal_dataloaders

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_model(
    model_type: str,
    seq_len: int,
    device: torch.device,
    use_lif: bool = False,
) -> MultimodalControlNetwork:
    modalities = MODALITY_DIMS
    num_blocks = 4 if model_type in ("multimodal", "hybrid", "lif") else 2
    use_lif_blocks = use_lif or model_type in ("lif", "hybrid")

    model = MultimodalControlNetwork(
        modalities=modalities,
        hidden_dim=64,
        embedding_dim=32,
        num_blocks=num_blocks,
        num_control_outputs=1,
        seq_len=seq_len,
        use_lif_in_blocks=use_lif_blocks,
    )
    if use_lif_blocks:
        logger.info("MultimodalControlNetwork: Temporal LIF enabled in spike blocks")
    return model.to(device)


def train_epoch(
    model: MultimodalControlNetwork,
    loss_fn: ControlPretrainingLoss,
    optimizer: optim.Optimizer,
    dataloader,
    device: torch.device,
    aux_weight: float,
) -> tuple[float, dict[str, float], float]:
    model.train()
    total_loss = 0.0
    total_pred = 0.0
    n_batches = 0
    detail_acc: dict[str, float] = {}

    for modalities, target in dataloader:
        modalities = {k: v.to(device) for k, v in modalities.items()}
        target = target.to(device)

        output = model(modalities)
        pred = output["control_actions"]
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)

        loss_pred = F.mse_loss(pred, target)
        loss_dict: dict[str, float] = {}

        if aux_weight > 0:
            original = modalities["load"][:, -1, :]
            embeddings = output["final_representation"].unsqueeze(1)
            aux_total, loss_dict = loss_fn(
                original_data=original,
                reconstructed_data=pred,
                embeddings=embeddings,
                control_actions=pred.unsqueeze(1),
                confidence=output["confidence"],
            )
            loss = loss_pred + aux_weight * aux_total
        else:
            loss = loss_pred

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_pred += loss_pred.item()
        n_batches += 1
        for k, v in loss_dict.items():
            detail_acc[k] = detail_acc.get(k, 0.0) + v

    avg = total_loss / max(n_batches, 1)
    avg_pred = total_pred / max(n_batches, 1)
    for k in detail_acc:
        detail_acc[k] /= max(n_batches, 1)
    return avg, detail_acc, avg_pred


@torch.no_grad()
def eval_epoch(model, dataloader, device) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    n = 0
    for modalities, target in dataloader:
        modalities = {k: v.to(device) for k, v in modalities.items()}
        target = target.to(device)
        pred = model(modalities)["control_actions"]
        if pred.dim() == 1:
            pred = pred.unsqueeze(-1)
        total_mse += F.mse_loss(pred, target, reduction="sum").item()
        n += target.numel()
    mse = total_mse / max(n, 1)
    return mse, mse**0.5


def main():
    parser = argparse.ArgumentParser(description="PSML multimodal control pretraining")
    parser.add_argument(
        "--data-root",
        type=str,
        default="/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable",
    )
    parser.add_argument("--zones", type=str, nargs="+", default=["ERCOT_zone_1_"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--stride", type=int, default=96)
    parser.add_argument("--max-rows-per-zone", type=int, default=None, help="Debug: cap CSV rows")
    parser.add_argument("--model-type", default="multimodal", choices=["multimodal", "lif", "izh", "hybrid"])
    parser.add_argument(
        "--use-lif",
        action="store_true",
        help="Enable Temporal LIF in control blocks (multimodal + SNN)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--aux-weight",
        type=float,
        default=0.0,
        help="Weight for ControlPretrainingLoss (0 = prediction MSE only)",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    logger.info("Loading PSML multimodal data from %s", args.data_root)
    logger.info("Zones: %s | seq_len=%d stride=%d", args.zones, args.seq_len, args.stride)

    train_loader, val_loader = create_multimodal_dataloaders(
        data_root=args.data_root,
        zones=args.zones,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
        max_rows_per_zone=args.max_rows_per_zone,
        val_ratio=args.val_ratio,
        num_workers=args.num_workers,
    )

    model = build_model(args.model_type, args.seq_len, device, use_lif=args.use_lif)
    loss_fn = ControlPretrainingLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    history: dict = {
        "train_loss": [],
        "train_pred_mse": [],
        "val_mse": [],
        "val_rmse": [],
        "lr": [],
        "modalities": MODALITY_DIMS,
        "target": "next_step_load_power",
        "run_config": {
            "data_root": args.data_root,
            "zones": args.zones,
            "seq_len": args.seq_len,
            "stride": args.stride,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
            "val_ratio": args.val_ratio,
            "use_lif": args.use_lif or args.model_type in ("lif", "hybrid"),
            "model_type": args.model_type,
        },
    }

    best_val_rmse = float("inf")
    best_state = None

    logger.info("Train batches: %d", len(train_loader))
    if val_loader:
        logger.info("Val batches: %d", len(val_loader))

    for epoch in range(1, args.epochs + 1):
        avg_loss, detail, avg_pred = train_epoch(
            model, loss_fn, optimizer, train_loader, device, args.aux_weight
        )
        history["train_loss"].append(avg_loss)
        history["train_pred_mse"].append(avg_pred)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        val_mse, val_rmse = float("nan"), float("nan")
        if val_loader is not None:
            val_mse, val_rmse = eval_epoch(model, val_loader, device)
            history["val_mse"].append(val_mse)
            history["val_rmse"].append(val_rmse)

        scheduler.step()

        if val_loader is not None and val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            msg = (
                f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.5f} pred_mse={avg_pred:.5f}"
            )
            if val_loader is not None:
                msg += f" | val_mse={val_mse:.5f} val_rmse={val_rmse:.5f}"
            logger.info(msg)

    history["run_config"]["best_val_rmse"] = (
        best_val_rmse if best_val_rmse < float("inf") else None
    )

    ckpt_payload = {
        "modalities": MODALITY_DIMS,
        "seq_len": args.seq_len,
        "zones": args.zones,
        "target": "next_step_load_power",
    }

    tag = "_lif" if (args.use_lif or args.model_type == "lif") else ""
    ckpt_path = Path(args.checkpoint_dir) / f"control_model_psml{tag}.pth"
    torch.save({**ckpt_payload, "model_state_dict": model.state_dict()}, ckpt_path)

    best_path = Path(args.checkpoint_dir) / f"control_model_psml{tag}_best.pth"
    if best_state is not None:
        torch.save({**ckpt_payload, "model_state_dict": best_state}, best_path)
        logger.info("Saved best checkpoint (val_rmse=%.5f): %s", best_val_rmse, best_path)

    hist_path = Path(args.output_dir) / f"training_history_psml{tag}.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    logger.info("Saved checkpoint: %s", ckpt_path)
    logger.info("Saved history: %s", hist_path)

    try:
        from plot_control_psml import plot_history

        plot_history(hist_path, Path(args.output_dir))
    except Exception as e:
        logger.warning("Could not auto-plot curves: %s (run: python plot_control_psml.py)", e)


if __name__ == "__main__":
    main()
