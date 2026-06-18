"""
PSML 多模态数据加载与数据集分割工具
=====================================
用于 TechDoc Unified Framework Demo 的数据准备。

将 PSML 11 维特征按模态拆分为 4 组:
  load(1) + renewable(2) + irradiance(4) + weather(4) = 11

核心分割函数:
  - split_train_val_indices()    训练/验证集索引随机分割
  - build_multimodal_windows()   滑动窗口 + 模态拆分 + next-step target
  - load_multi_zone_multimodal() 一站式多区域加载
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# 模态定义
# ============================================================================

MODALITY_COLUMNS: Dict[str, List[str]] = {
    "load": ["load_power"],
    "renewable": ["wind_power", "solar_power"],
    "irradiance": ["DHI", "DNI", "GHI", "Solar Zenith Angle"],
    "weather": ["Dew Point", "Wind Speed", "Relative Humidity", "Temperature"],
}

MODALITY_DIMS: Dict[str, int] = {name: len(cols) for name, cols in MODALITY_COLUMNS.items()}

ALL_FEATURE_COLS: List[str] = [c for cols in MODALITY_COLUMNS.values() for c in cols]

_COL_IDX: Dict[str, int] = {c: i for i, c in enumerate(ALL_FEATURE_COLS)}


# ============================================================================
# 数据加载器
# ============================================================================

class LoadRenewableDataLoader:
    """扫描并加载 PSML 分钟级电网区域 CSV 数据。"""

    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.zones = self._discover_zones()
        logger.info("发现 %d 个电网区域", len(self.zones))

    def _discover_zones(self) -> Dict[str, Path]:
        zones = {}
        for csv_file in self.data_root.glob("*.csv"):
            if csv_file.name == "desktop.ini":
                continue
            zones[csv_file.stem] = csv_file
        return zones

    def load_zone(self, zone_name: str) -> pd.DataFrame:
        if zone_name not in self.zones:
            raise ValueError(f"区域 {zone_name} 不存在")
        fp = self.zones[zone_name]
        logger.info("加载 %s (%.2f GB)", zone_name, fp.stat().st_size / 1e9)
        df = pd.read_csv(fp)
        if "time" in df.columns:
            df = df.set_index("time")
        df = df.select_dtypes(include=["number"])
        return df


class TimeSeriesDataset:
    """滑动窗口数据集 (用于单区域简单加载)。"""

    def __init__(self, data: pd.DataFrame, seq_len: int = 96,
                 stride: int = 96, normalize: str = "zscore"):
        self.seq_len = seq_len
        self.stride = stride
        data = data.select_dtypes(include=["number"])
        if normalize == "zscore":
            self.data = (data - data.mean()) / (data.std() + 1e-8)
        elif normalize == "minmax":
            self.data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            self.data = data
        self.data = self.data.dropna()
        self.indices = list(range(0, len(self.data) - seq_len + 1, stride))

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, dict]:
        s, e = self.indices[idx], self.indices[idx] + self.seq_len
        return self.data.iloc[s:e].values.astype(np.float32), {}


# ============================================================================
# 数据集分割函数
# ============================================================================

def split_train_val_indices(
    n_samples: int,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """随机分割训练/验证集索引。

    Args:
        n_samples: 总样本数
        train_ratio: 训练集占比 (默认 0.8)
        seed: 随机种子

    Returns:
        train_idx, val_idx: 训练/验证集索引数组
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_samples)
    n_train = max(1, int(n_samples * train_ratio))
    return perm[:n_train], perm[n_train:]


def load_and_normalize_zone(
    data_root: str, zone_name: str,
    normalize: str = "zscore",
    max_rows: Optional[int] = None,
) -> pd.DataFrame:
    """加载单个区域并做归一化，列对齐 ALL_FEATURE_COLS。"""
    loader = LoadRenewableDataLoader(data_root)
    df = loader.load_zone(zone_name)
    avail = [c for c in ALL_FEATURE_COLS if c in df.columns]
    df = df[avail].apply(pd.to_numeric, errors="coerce").dropna()
    if max_rows is not None:
        df = df.iloc[:max_rows]
    if normalize == "zscore":
        mean, std = df.mean(), df.std() + 1e-8
        df = ((df - mean) / std).astype(np.float32)
    elif normalize == "minmax":
        mn, mx = df.min(), df.max()
        df = ((df - mn) / (mx - mn + 1e-8)).astype(np.float32)
    else:
        df = df.astype(np.float32)
    return df


def build_multimodal_windows(
    df_norm: pd.DataFrame,
    seq_len: int = 96,
    stride: int = 96,
    max_windows: int = 500,
    data_fraction: float = 1.0,
) -> Tuple[List[np.ndarray], List[dict], List[float], List[float], List[float]]:
    """从归一化 DataFrame 构建多模态滑动窗口。

    Returns:
        windows_11d:      [(seq_len, 11)]  SNN 分支输入
        modalities_list:   [{mod_name: (seq_len, d_m)}]  多模态分支输入
        next_loads:        [float]  next-step load_power 目标
        grid_loads:        [float]  窗口平均 load_power
        grid_rens:         [float]  窗口平均 wind+solar
    """
    n_total = len(df_norm) - seq_len - 1
    if n_total < 1:
        raise RuntimeError(f"数据不足: 需要 > {seq_len} 行")

    n_windows = n_total // stride
    n_use = min(n_windows, max_windows)
    if 0.0 < data_fraction < 1.0:
        n_use = min(n_use, max(1, int(n_windows * data_fraction)))

    windows_11d, mods_list = [], []
    next_loads, grid_loads, grid_rens = [], [], []

    for i in range(n_use):
        start = i * stride
        end = start + seq_len
        block = df_norm.iloc[start:end]

        arr = np.zeros((seq_len, 11), dtype=np.float32)
        for j, col in enumerate(ALL_FEATURE_COLS):
            if col in block.columns:
                arr[:, j] = block[col].values.astype(np.float32)
        windows_11d.append(arr)

        mod_dict = {}
        for mod_name, cols in MODALITY_COLUMNS.items():
            ci = [_COL_IDX[c] for c in cols if c in _COL_IDX]
            mod_dict[mod_name] = arr[:, ci]
        mods_list.append(mod_dict)

        target_row = end
        if target_row < len(df_norm) and "load_power" in df_norm.columns:
            next_loads.append(float(df_norm.iloc[target_row]["load_power"]))
        else:
            next_loads.append(0.0)

        grid_loads.append(float(arr[:, 0].mean()))
        grid_rens.append(float(arr[:, 1:3].sum(axis=1).mean()))

    logger.info("构建 %d 个多模态窗口 (seq_len=%d, fraction=%.2f)",
                n_use, seq_len, data_fraction)
    return windows_11d, mods_list, next_loads, grid_loads, grid_rens


def load_multi_zone_multimodal(
    data_root: str,
    zones: List[str],
    seq_len: int = 96,
    stride: int = 96,
    normalize: str = "zscore",
    max_windows_per_zone: int = 500,
    data_fraction: float = 1.0,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict:
    """一站式: 多区域加载 → 多模态窗口 → 分割 → 返回全部数据。

    Returns 字典:
      - zones, n_windows, train_idx, val_idx
      - windows_11d, modalities, next_loads, grid_loads, grid_rens (每区域 list)
    """
    all_w11, all_mod, all_nl = [], [], []
    all_gl, all_gr = [], []

    for zone in zones:
        df_norm = load_and_normalize_zone(data_root, zone, normalize)
        w11, mods, nls, gls, grs = build_multimodal_windows(
            df_norm, seq_len, stride, max_windows_per_zone, data_fraction,
        )
        all_w11.append(np.stack(w11, axis=0))
        all_mod.append(mods)
        all_nl.append(np.array(nls, dtype=np.float32))
        all_gl.append(np.array(gls, dtype=np.float32))
        all_gr.append(np.array(grs, dtype=np.float32))

    n_windows = min(d.shape[0] for d in all_w11)
    train_idx, val_idx = split_train_val_indices(n_windows, train_ratio, seed)

    logger.info("多区域多模态数据就绪: zones=%s, windows/zone=%d, train=%d, val=%d",
                zones, n_windows, len(train_idx), len(val_idx))

    return {
        "zones": zones,
        "windows_11d": all_w11, "modalities": all_mod,
        "next_loads": all_nl, "grid_loads": all_gl, "grid_rens": all_gr,
        "train_idx": train_idx, "val_idx": val_idx, "n_windows": n_windows,
    }
