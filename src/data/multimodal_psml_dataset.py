"""
PSML minute-level data as multimodal inputs for MultimodalControlNetwork.

Modalities (per 技术文档 / CSV columns):
  - load:       load_power
  - renewable:  wind_power, solar_power
  - irradiance: DHI, DNI, GHI, Solar Zenith Angle
  - weather:    Dew Point, Wind Speed, Relative Humidity, Temperature

Target: normalized load_power at the first timestep AFTER the input window (next-step prediction).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

MODALITY_COLUMNS: Dict[str, List[str]] = {
    "load": ["load_power"],
    "renewable": ["wind_power", "solar_power"],
    "irradiance": ["DHI", "DNI", "GHI", "Solar Zenith Angle"],
    "weather": ["Dew Point", "Wind Speed", "Relative Humidity", "Temperature"],
}

MODALITY_DIMS: Dict[str, int] = {name: len(cols) for name, cols in MODALITY_COLUMNS.items()}

ALL_FEATURE_COLUMNS: List[str] = [c for cols in MODALITY_COLUMNS.values() for c in cols]


def discover_zone_files(data_root: Path, zones: Optional[List[str]] = None) -> Dict[str, Path]:
    """Resolve zone name -> CSV path."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"PSML data root not found: {data_root}")

    available = {p.stem: p for p in data_root.glob("*.csv") if p.name != "desktop.ini"}
    if not available:
        raise FileNotFoundError(f"No CSV files under {data_root}")

    if zones is None:
        return dict(sorted(available.items()))

    resolved: Dict[str, Path] = {}
    for z in zones:
        key = z if z in available else z.rstrip(".csv")
        if key not in available:
            matches = [k for k in available if k.startswith(z) or z in k]
            if len(matches) == 1:
                key = matches[0]
            elif matches:
                key = matches[0]
                logger.warning("Ambiguous zone %s, using %s", z, key)
            else:
                raise ValueError(f"Zone not found: {z}. Available: {list(available.keys())[:8]}...")
        resolved[key] = available[key]
    return resolved


def _normalize_frame(df: pd.DataFrame, method: str) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """Per-zone normalization; returns normalized frame and stats for load denorm."""
    if method == "zscore":
        mean = df.mean()
        std = df.std() + 1e-8
        normed = (df - mean) / std
        stats = {"mean": mean.values.astype(np.float32), "std": std.values.astype(np.float32)}
    elif method == "minmax":
        mn = df.min()
        mx = df.max()
        normed = (df - mn) / (mx - mn + 1e-8)
        stats = {"min": mn.values.astype(np.float32), "max": mx.values.astype(np.float32)}
    else:
        normed = df.copy()
        stats = {}
    return normed.astype(np.float32), stats


def load_psml_zone_frames(
    data_root: str,
    zones: Optional[List[str]] = None,
    normalize: str = "zscore",
    max_rows_per_zone: Optional[int] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict], Dict[str, int]]:
    """Load and normalize all zones once (shared by train/val splits)."""
    data_root_p = Path(data_root)
    zone_paths = discover_zone_files(data_root_p, zones)
    zone_frames: Dict[str, pd.DataFrame] = {}
    zone_stats: Dict[str, Dict] = {}
    load_col_idx: Dict[str, int] = {}

    for zone_name, csv_path in zone_paths.items():
        logger.info("Loading %s from %s", zone_name, csv_path.name)
        df = pd.read_csv(
            csv_path,
            usecols=lambda c: c == "time" or c in ALL_FEATURE_COLUMNS,
            nrows=max_rows_per_zone,
        )
        if "time" in df.columns:
            df = df.set_index("time")
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

        missing = [c for c in ALL_FEATURE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{zone_name} missing columns: {missing}")

        df = df[ALL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").dropna()
        df, stats = _normalize_frame(df, normalize)
        zone_frames[zone_name] = df
        zone_stats[zone_name] = stats
        load_col_idx[zone_name] = ALL_FEATURE_COLUMNS.index("load_power")
        logger.info("  rows=%d, features=%d", len(df), df.shape[1])

    return zone_frames, zone_stats, load_col_idx


class MultimodalPSMLDataset(Dataset):
    """
    Sliding-window multimodal samples from PSML Minute-level Load and Renewable CSVs.

    Each item:
      modalities: {name: FloatTensor [seq_len, dim]}
      target: FloatTensor [1]  — next-step load_power (normalized, same scale as input load channel)
    """

    def __init__(
        self,
        data_root: str,
        zones: Optional[List[str]] = None,
        seq_len: int = 96,
        stride: int = 96,
        normalize: str = "zscore",
        max_rows_per_zone: Optional[int] = None,
        indices: Optional[List[Tuple[str, int]]] = None,
        zone_frames: Optional[Dict[str, pd.DataFrame]] = None,
        zone_stats: Optional[Dict[str, Dict]] = None,
        load_col_idx: Optional[Dict[str, int]] = None,
    ):
        self.data_root = Path(data_root)
        self.seq_len = seq_len
        self.stride = stride
        self.normalize = normalize

        if zone_frames is not None:
            self.zone_frames = zone_frames
            self.zone_stats = zone_stats or {}
            self.load_col_idx = load_col_idx or {
                z: ALL_FEATURE_COLUMNS.index("load_power") for z in zone_frames
            }
        else:
            self.zone_frames, self.zone_stats, self.load_col_idx = load_psml_zone_frames(
                data_root, zones, normalize, max_rows_per_zone
            )

        if indices is not None:
            self.indices = indices
        else:
            self.indices = self._build_indices()

        if not self.indices:
            raise ValueError("No training samples; try smaller seq_len or more data.")

        logger.info("MultimodalPSMLDataset: %d samples, seq_len=%d", len(self.indices), seq_len)

    def _build_indices(self) -> List[Tuple[str, int]]:
        out: List[Tuple[str, int]] = []
        for zone_name, df in self.zone_frames.items():
            n = len(df)
            # window [start, start+seq_len), target at row start+seq_len
            last_start = n - self.seq_len - 1
            for start in range(0, last_start + 1, self.stride):
                out.append((zone_name, start))
        return out

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        zone_name, start = self.indices[idx]
        df = self.zone_frames[zone_name]
        end = start + self.seq_len
        target_row = end  # next step after window

        block = df.iloc[start:end]
        target_load = float(df.iloc[target_row, self.load_col_idx[zone_name]])

        modalities: Dict[str, torch.Tensor] = {}
        for mod_name, cols in MODALITY_COLUMNS.items():
            arr = block[cols].values.astype(np.float32)
            modalities[mod_name] = torch.from_numpy(arr)

        target = torch.tensor([target_load], dtype=torch.float32)
        return modalities, target


def multimodal_psml_collate(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """Stack a batch for DataLoader."""
    modalities: Dict[str, torch.Tensor] = {}
    for mod_name in MODALITY_COLUMNS:
        modalities[mod_name] = torch.stack([b[0][mod_name] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)  # (B, 1)
    return modalities, targets


def create_multimodal_dataloaders(
    data_root: str,
    zones: Optional[List[str]] = None,
    seq_len: int = 96,
    stride: int = 96,
    batch_size: int = 32,
    normalize: str = "zscore",
    max_rows_per_zone: Optional[int] = None,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    train_sampler=None,
) -> Tuple[torch.utils.data.DataLoader, Optional[torch.utils.data.DataLoader]]:
    """Build train / val DataLoaders with a deterministic index split."""
    zone_frames, zone_stats, load_col_idx = load_psml_zone_frames(
        data_root, zones, normalize, max_rows_per_zone
    )
    shared_kw = dict(
        data_root=data_root,
        zones=zones,
        seq_len=seq_len,
        stride=stride,
        normalize=normalize,
        zone_frames=zone_frames,
        zone_stats=zone_stats,
        load_col_idx=load_col_idx,
    )

    full = MultimodalPSMLDataset(**shared_kw)

    n = len(full)
    if n < 2 or val_ratio <= 0:
        train_loader = torch.utils.data.DataLoader(
            full,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=multimodal_psml_collate,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            drop_last=len(full) >= batch_size,
        )
        return train_loader, None

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(1, int(n * val_ratio))
    val_idx = perm[:n_val].tolist()
    train_idx = perm[n_val:].tolist()

    train_indices = [full.indices[i] for i in train_idx]
    val_indices = [full.indices[i] for i in val_idx]

    train_ds = MultimodalPSMLDataset(**shared_kw, indices=train_indices)
    val_ds = MultimodalPSMLDataset(**shared_kw, indices=val_indices)

    loader_kw = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=multimodal_psml_collate,
        pin_memory=num_workers >= 0,
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=len(train_ds) >= batch_size,
        persistent_workers=num_workers > 0,
        **loader_kw,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        shuffle=False,
        **loader_kw,
    )
    return train_loader, val_loader
