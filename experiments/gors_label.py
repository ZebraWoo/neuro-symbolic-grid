#!/usr/bin/env python3
"""
GORS (Grid Operational Risk Score) Pseudo-Risk Label Generation.

No fake decision intents. The risk target y ∈ [0,1] is derived naturally
from multi-modal volatility in the PSML data:

  y = σ(w1·|Δnet_load|/σ_nl + w2·|Δrenewable|/σ_r + w3·wind_anomaly)

where σ is sigmoid, mapping unbounded volatility → [0,1] risk score.

Usage:
  python experiments/gors_label.py --zones-per-split 3
"""

from __future__ import annotations

import numpy as np

# Column indices in 11-dim PSML data
COL_LOAD = 0
COL_WIND = 1
COL_SOLAR = 2
COL_WIND_SPEED = 8
COL_TEMP = 10

from experiments.gors_config import gors_cfg, GORSLabelConfig


def compute_gors_targets(
    windows_t: np.ndarray,     # (N, seq_len, 11)
    windows_t1: np.ndarray,    # (N, seq_len, 11) — next window
    cfg: GORSLabelConfig = None,
) -> np.ndarray:                # (N,)  risk score ∈ [0,1]
    """
    Compute GORS pseudo-target from data volatility between consecutive windows.

    Risk components:
      1. Net load volatility: |Δnet_load| / σ
      2. Renewable volatility: |Δrenewable| / σ
      3. Weather anomaly: |wind_speed - μ| / σ  (high wind = risk)
      4. Temperature anomaly: |temp - μ| / σ  (extreme temp = risk)

    Combined via sigmoid to [0,1].
    """
    if cfg is None:
        cfg = gors_cfg.labels

    N = windows_t.shape[0]

    # Window-level means
    load_t = windows_t[:, :, COL_LOAD].mean(axis=1)
    load_t1 = windows_t1[:, :, COL_LOAD].mean(axis=1)
    wind_t = windows_t[:, :, COL_WIND].mean(axis=1)
    wind_t1 = windows_t1[:, :, COL_WIND].mean(axis=1)
    solar_t = windows_t[:, :, COL_SOLAR].mean(axis=1)
    solar_t1 = windows_t1[:, :, COL_SOLAR].mean(axis=1)
    wind_speed_t1 = windows_t1[:, :, COL_WIND_SPEED].mean(axis=1)
    temp_t1 = windows_t1[:, :, COL_TEMP].mean(axis=1)

    # Net load = load - renewable
    renewable_t = wind_t + solar_t
    renewable_t1 = wind_t1 + solar_t1
    net_load_t = load_t - renewable_t
    net_load_t1 = load_t1 - renewable_t1

    # ---- Risk Component 1: Net Load Volatility ----
    delta_net_load = np.abs(net_load_t1 - net_load_t)
    sigma_nl = np.std(net_load_t) + 1e-6
    risk_nl = delta_net_load / sigma_nl

    # ---- Risk Component 2: Renewable Volatility ----
    delta_renewable = np.abs(renewable_t1 - renewable_t)
    sigma_r = np.std(renewable_t) + 1e-6
    risk_r = delta_renewable / sigma_r

    # ---- Risk Component 3: Weather Anomaly (Wind Speed) ----
    mu_ws = np.mean(wind_speed_t1)
    sigma_ws = np.std(wind_speed_t1) + 1e-6
    risk_ws = np.abs(wind_speed_t1 - mu_ws) / sigma_ws

    # ---- Risk Component 4: Temperature Anomaly ----
    mu_temp = np.mean(temp_t1)
    sigma_temp = np.std(temp_t1) + 1e-6
    risk_temp = np.abs(temp_t1 - mu_temp) / sigma_temp

    # ---- Combined Risk (pre-sigmoid) ----
    combined = (
        cfg.w_net_load * risk_nl +
        cfg.w_renewable * risk_r +
        cfg.w_weather * 0.6 * risk_ws +
        cfg.w_weather * 0.4 * risk_temp
    )

    # Sigmoid to [0, 1]
    gors = 1.0 / (1.0 + np.exp(-cfg.sigmoid_scale * (combined - 1.0)))

    return gors.astype(np.float32)


def generate_gors_labels(data: np.ndarray, seq_len: int, cfg=None) -> np.ndarray:
    """
    Generate GORS labels for pre-loaded window data.

    Args:
        data: [N, seq_len, 11] normalized PSML windows
    Returns:
        gors: [N] risk scores ∈ [0,1]
    """
    N = data.shape[0]
    if N < 2:
        return np.full(N, 0.5, dtype=np.float32)

    windows_t = data[:-1]
    windows_t1 = data[1:]
    y = compute_gors_targets(windows_t, windows_t1, cfg)
    # Pad last sample
    y_padded = np.zeros(N, dtype=np.float32)
    y_padded[:-1] = y
    y_padded[-1] = y[-1] if len(y) > 0 else 0.5
    return y_padded


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------

import torch
from torch.utils.data import Dataset

# Reuse PreloadedDataset from earlier
FEATURE_COLS = {
    "load_power": 0,
    "wind_power": 1, "solar_power": 2,
    "DHI": 3, "DNI": 4, "GHI": 5, "Solar Zenith Angle": 6,
    "Dew Point": 7, "Wind Speed": 8, "Relative Humidity": 9, "Temperature": 10,
}

MODALITY_SPLITS = {
    "load": [0],
    "renewable": [1, 2],
    "irradiance": [3, 4, 5, 6],
    "weather": [7, 8, 9, 10],
}


class PreloadedDataset(Dataset):
    """Dataset that takes pre-loaded numpy arrays and returns modality dicts."""
    def __init__(self, data: np.ndarray, seq_len: int = 96):
        super().__init__()
        self.data = torch.from_numpy(data).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        window = self.data[idx]
        modalities = {}
        for name, cols in MODALITY_SPLITS.items():
            modalities[name] = window[:, cols]
        return modalities, torch.tensor(0.0)  # target is dummy


class GORSDataset(Dataset):
    """Returns (modalities_dict, gors_score)."""
    def __init__(self, data: np.ndarray, gors: np.ndarray, seq_len: int = 96):
        self.base = PreloadedDataset(data, seq_len=seq_len)
        self.gors = torch.from_numpy(gors).float()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        modalities, _ = self.base[idx]
        return modalities, self.gors[idx]


if __name__ == "__main__":
    # Quick test
    import argparse, logging, sys
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("gors_label")

    parser = argparse.ArgumentParser()
    parser.add_argument("--zones-per-split", type=int, default=0)
    parser.add_argument("--data-root", default=gors_cfg.data.root)
    parser.add_argument("--seq-len", type=int, default=96)
    args = parser.parse_args()

    # Generate for train split
    zones = gors_cfg.data.train_zones
    if args.zones_per_split > 0:
        zones = zones[:args.zones_per_split]

    logger.info("Generating GORS labels for %d zones...", len(zones))

    # Quick test with random data to show distribution
    dummy = np.abs(np.random.randn(1000, 96, 11).astype(np.float32))
    dummy[:, :, 0] *= 100  # load scale
    dummy[:, :, 1] *= 50   # wind scale
    dummy[:, :, 2] *= 30   # solar scale
    gors = generate_gors_labels(dummy, 96)
    logger.info("GORS distribution (dummy): min=%.3f max=%.3f mean=%.3f median=%.3f",
                gors.min(), gors.max(), gors.mean(), np.median(gors))
    logger.info("Risk buckets: low(<0.3)=%.1f%% mid(0.3-0.7)=%.1f%% high(>0.7)=%.1f%%",
                100*(gors < 0.3).mean(), 100*((gors >= 0.3) & (gors <= 0.7)).mean(), 100*(gors > 0.7).mean())
    logger.info("Done.")
