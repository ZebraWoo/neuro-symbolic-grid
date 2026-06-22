#!/usr/bin/env python3
"""
Generate decision intent labels from PSML data.

Produces: For each sliding window at time t, 5 binary labels
representing the recommended decision intents for time t+1.

Labels:
  0: Increase Generation  (Δnet_load > +δ)
  1: Charge ESS           (Δrenewable > +δ AND SOC < SOC_max)
  2: Discharge ESS        (Δrenewable < -δ AND SOC > SOC_min)
  3: Voltage Support      (Δload > +2δ OR Δrenewable < -2δ)
  4: Risk Warning         (wind_speed > τ OR Δload > 3δ)

Usage:
  python experiments/label_decision_intents.py
  python experiments/label_decision_intents.py --data-root ... --output labels/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.load_renewable_dataset import LoadRenewableDataLoader, TimeSeriesDataset
from experiments.config import default_cfg, DecisionLabelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Column indices in PSML data
COL_LOAD = 0
COL_WIND = 1
COL_SOLAR = 2
COL_WIND_SPEED = 8  # "Wind Speed" column


def compute_decision_labels(
    windows: np.ndarray,      # (N, seq_len, 11)
    next_windows: np.ndarray,  # (N, seq_len, 11) — the window at t+1
    cfg: DecisionLabelConfig,
) -> np.ndarray:               # (N, 5) binary labels
    """
    For each pair of consecutive windows (t, t+1), derive 5 decision intents.
    """
    N = windows.shape[0]

    # Compute window-level averages for key quantities
    load_t = windows[:, :, COL_LOAD].mean(axis=1)           # (N,)
    load_t1 = next_windows[:, :, COL_LOAD].mean(axis=1)     # (N,)
    wind_t = windows[:, :, COL_WIND].mean(axis=1)
    wind_t1 = next_windows[:, :, COL_WIND].mean(axis=1)
    solar_t = windows[:, :, COL_SOLAR].mean(axis=1)
    solar_t1 = next_windows[:, :, COL_SOLAR].mean(axis=1)
    wind_speed_t = windows[:, :, COL_WIND_SPEED].mean(axis=1)
    wind_speed_t1 = next_windows[:, :, COL_WIND_SPEED].mean(axis=1)

    renewable_t = wind_t + solar_t
    renewable_t1 = wind_t1 + solar_t1
    net_load_t = load_t - renewable_t
    net_load_t1 = load_t1 - renewable_t1

    # Delta computations
    delta_net_load = net_load_t1 - net_load_t
    delta_renewable = renewable_t1 - renewable_t
    delta_load = load_t1 - load_t

    # Threshold
    delta = cfg.delta_ratio * np.std(net_load_t)

    # SOC estimation (simplified: track cumulative renewable surplus/deficit)
    # We use a proxy SOC based on cumulative delta_renewable normalized to [0,1]
    # This is NOT a real SOC but serves as a consistent label-generation proxy
    soc_proxy = estimate_soc_proxy(renewable_t, load_t)

    labels = np.zeros((N, 5), dtype=np.float32)

    # Label 0: Increase Generation
    labels[:, 0] = (delta_net_load > delta).astype(np.float32)

    # Label 1: Charge ESS (renewable rising AND SOC has room)
    labels[:, 1] = ((delta_renewable > delta) & (soc_proxy < cfg.soc_max)).astype(np.float32)

    # Label 2: Discharge ESS (renewable dropping AND SOC has charge)
    labels[:, 2] = ((delta_renewable < -delta) & (soc_proxy > cfg.soc_min)).astype(np.float32)

    # Label 3: Voltage Support (large net load swing)
    labels[:, 3] = ((delta_load > 2 * delta) | (delta_renewable < -2 * delta)).astype(np.float32)

    # Label 4: Risk Warning (high wind or extreme load change)
    # Use std for z-score normalized data; use mean for raw data
    load_ref = np.std(load_t) if np.std(load_t) > 0.1 else np.mean(np.abs(load_t)) + 1e-6
    wind_ref = np.std(wind_speed_t) if np.std(wind_speed_t) > 0.1 else np.mean(np.abs(wind_speed_t)) + 1e-6
    wind_thresh = cfg.wind_speed_threshold * max(wind_ref, 1.0)  # adaptive to data scale
    labels[:, 4] = ((wind_speed_t1 > wind_thresh) |
                    (np.abs(delta_load) > cfg.load_surge_threshold * load_ref)).astype(np.float32)

    return labels


def estimate_soc_proxy(renewable: np.ndarray, load: np.ndarray) -> np.ndarray:
    """
    Simplified SOC proxy for label generation.
    Uses renewable-to-load ratio as a heuristic for storage state.
    SOC ≈ 0.5 + 0.4 * tanh((renewable - load) / mean(load))
    Maps to roughly [0.1, 0.9].
    """
    mean_load = np.mean(load)
    if mean_load < 1e-6:
        return np.full_like(renewable, 0.5)
    ratio = (renewable - load) / mean_load
    return 0.5 + 0.4 * np.tanh(ratio)


def generate_labels_for_zones(
    data_root: str,
    zones: list[str],
    seq_len: int,
    stride: int,
    cfg: DecisionLabelConfig,
    output_dir: Path,
    split_name: str,
) -> dict:
    """
    Generate decision labels for a set of zones. Returns summary stats.
    """
    loader = LoadRenewableDataLoader(data_root)
    all_labels = []
    all_zone_ids = []
    total_windows = 0

    for zone in zones:
        try:
            df = loader.load_zone(zone)
        except FileNotFoundError:
            logger.warning("Zone %s not found, skipping", zone)
            continue

        dataset = TimeSeriesDataset(
            df, seq_len=seq_len, stride=stride, normalize="zscore",
        )
        n = len(dataset)
        if n < 2:
            logger.warning("Zone %s has only %d windows, skipping", zone, n)
            continue

        # Get windows at t and t+1
        windows_t = []
        windows_t1 = []
        for i in range(n - 1):
            w, _ = dataset[i]
            w1, _ = dataset[i + 1]
            windows_t.append(w)
            windows_t1.append(w1)

        windows_t = np.stack(windows_t, axis=0)     # (n-1, seq_len, 11)
        windows_t1 = np.stack(windows_t1, axis=0)    # (n-1, seq_len, 11)

        labels = compute_decision_labels(windows_t, windows_t1, cfg)
        all_labels.append(labels)
        all_zone_ids.extend([zone] * labels.shape[0])
        total_windows += labels.shape[0]
        logger.info("  %s: %d windows → %d label pairs", zone, n, labels.shape[0])

    if not all_labels:
        raise ValueError(f"No valid zones found in {split_name} split!")

    all_labels = np.concatenate(all_labels, axis=0)

    # Save
    output_path = output_dir / f"decision_labels_{split_name}.npy"
    np.save(output_path, all_labels)

    zone_path = output_dir / f"decision_labels_{split_name}_zones.json"
    with open(zone_path, "w") as f:
        json.dump(all_zone_ids, f)

    # Statistics
    label_names = ["Increase Gen", "Charge ESS", "Discharge ESS", "Voltage Support", "Risk Warning"]
    stats = {
        "split": split_name,
        "total_windows": total_windows,
        "num_zones": len(set(all_zone_ids)),
        "label_distribution": {},
    }
    for i, name in enumerate(label_names):
        count = int(all_labels[:, i].sum())
        pct = 100.0 * count / total_windows
        stats["label_distribution"][name] = {"count": count, "percentage": round(pct, 2)}

    logger.info("  === %s Summary ===", split_name)
    logger.info("  Total windows: %d", total_windows)
    for name, d in stats["label_distribution"].items():
        logger.info("  %s: %d (%.2f%%)", name, d["count"], d["percentage"])

    return stats


def _generate_labels_for_data(data: np.ndarray, label_cfg, seq_len: int) -> np.ndarray:
    """Generate decision labels for pre-loaded window data."""
    N = data.shape[0]
    if N < 2:
        return np.zeros((N, 5), dtype=np.float32)

    windows_t = data[:-1]
    windows_t1 = data[1:]
    labels = compute_decision_labels(windows_t, windows_t1, label_cfg)
    labels_padded = np.zeros((N, 5), dtype=np.float32)
    labels_padded[:-1] = labels
    labels_padded[-1] = labels[-1] if len(labels) > 0 else np.zeros(5, dtype=np.float32)
    return labels_padded


# Column indices in the 11-dim PSML feature vector
FEATURE_COLS = {
    "load_power": 0,
    "wind_power": 1,
    "solar_power": 2,
    "DHI": 3, "DNI": 4, "GHI": 5, "Solar Zenith Angle": 6,
    "Dew Point": 7, "Wind Speed": 8, "Relative Humidity": 9, "Temperature": 10,
}

MODALITY_SPLITS = {
    "load": [0],                      # load_power
    "renewable": [1, 2],              # wind_power, solar_power
    "irradiance": [3, 4, 5, 6],       # DHI, DNI, GHI, Solar Zenith Angle
    "weather": [7, 8, 9, 10],         # Dew Point, Wind Speed, Relative Humidity, Temperature
}


class PreloadedDataset(torch.utils.data.Dataset):
    """
    Dataset that takes pre-loaded numpy arrays and returns modality dicts.

    Compatible with MultimodalPSMLDataset's output format:
      - modalities: Dict[str, Tensor[seq_len, dim]]
      - target: scalar (next-step load, not used for decision support)
    """

    def __init__(self, data: np.ndarray, seq_len: int = 96):
        """
        Args:
            data: [N, seq_len, 11] pre-loaded and normalized PSML windows
            seq_len: sequence length (should match data.shape[1])
        """
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]  # [seq_len, 11]
        modalities = {}
        for name, col_indices in MODALITY_SPLITS.items():
            modalities[name] = window[:, col_indices]
        # target is a dummy (not used for decision support)
        target = torch.tensor(0.0)
        return modalities, target


class LabeledDataset(torch.utils.data.Dataset):
    """Wraps PreloadedDataset to return (modalities, decision_labels)."""

    def __init__(self, data: np.ndarray, labels: np.ndarray, seq_len: int = 96):
        self.base = PreloadedDataset(data, seq_len=seq_len)
        self.labels = labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        modalities, _ = self.base[idx]
        label = torch.from_numpy(self.labels[idx]).float()
        return modalities, label


def main():
    parser = argparse.ArgumentParser(description="Generate decision intent labels")
    parser.add_argument("--data-root", default=default_cfg.data.root)
    parser.add_argument("--seq-len", type=int, default=default_cfg.data.seq_len)
    parser.add_argument("--stride", type=int, default=default_cfg.data.stride)
    parser.add_argument("--output-dir", default="experiments/labels")
    parser.add_argument("--delta-ratio", type=float, default=default_cfg.labels.delta_ratio)
    parser.add_argument("--zones-per-split", type=int, default=3,
                        help="Limit zones per split for quick test (0 = all)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label_cfg = DecisionLabelConfig(delta_ratio=args.delta_ratio)

    logger.info("Generating decision intent labels...")
    logger.info("Data root: %s", args.data_root)
    logger.info("seq_len=%d, stride=%d, delta_ratio=%.2f",
                args.seq_len, args.stride, args.delta_ratio)

    splits = {
        "train": default_cfg.data.train_zones,
        "val": default_cfg.data.val_zones,
        "test": default_cfg.data.test_zones,
    }

    all_stats = {}
    for split_name, zones in splits.items():
        if args.zones_per_split > 0:
            zones = zones[:args.zones_per_split]
        logger.info("\nProcessing %s split (%d zones)...", split_name, len(zones))
        stats = generate_labels_for_zones(
            args.data_root, zones, args.seq_len, args.stride,
            label_cfg, output_dir, split_name,
        )
        all_stats[split_name] = stats

    # Save summary
    summary_path = output_dir / "label_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    logger.info("\nSaved label summary: %s", summary_path)
    logger.info("Done. Labels saved to %s/", output_dir)


if __name__ == "__main__":
    main()
