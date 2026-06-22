"""
PSML 分钟级负荷和可再生能源数据加载与数据集分割工具
========================================================
用于 Neuro-Symbolic Grid Control Demo 的数据准备。

数据源: PSML (Power System Multi-Level) 分钟级数据集
每个 CSV 包含 11 维特征: load_power, wind_power, solar_power,
  DHI, DNI, GHI, Dew Point, Solar Zenith Angle, Wind Speed,
  Relative Humidity, Temperature

核心分割函数:
  - split_dataset_indices()    TimeSeriesDataset 索引分割
  - multi_zone_windows()       多区域批量加载 + data_fraction 控量
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    """将长时间序列切分为固定长度的滑动窗口样本。"""

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
        logger.info("创建数据集: %d 个样本, seq_len=%d", len(self.indices), seq_len)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, dict]:
        s, e = self.indices[idx], self.indices[idx] + self.seq_len
        return self.data.iloc[s:e].values.astype(np.float32), {}


# ============================================================================
# 数据集分割函数
# ============================================================================

def split_dataset_indices(
    dataset: TimeSeriesDataset,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """将 TimeSeriesDataset 的样本索引随机分割为训练集和验证集。

    Args:
        dataset: 已构建的 TimeSeriesDataset 实例
        train_ratio: 训练集占比 (默认 0.8)
        seed: 随机种子

    Returns:
        train_indices, val_indices
    """
    n = len(dataset)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = max(1, int(n * train_ratio))
    return perm[:n_train].tolist(), perm[n_train:].tolist()


def multi_zone_windows(
    data_root: str,
    zones: List[str],
    seq_len: int = 96,
    stride: int = 96,
    normalize: str = "zscore",
    max_windows_per_zone: int = 500,
    data_fraction: float = 1.0,
) -> List[np.ndarray]:
    """加载多区域数据，返回每区域的窗口数组。

    Args:
        data_root: PSML 数据根目录
        zones: 区域名列表
        seq_len: 序列长度
        stride: 窗口步长
        normalize: 归一化方式
        max_windows_per_zone: 每区域最大窗口数
        data_fraction: 数据使用比例 (0~1)

    Returns:
        List[np.ndarray]: 每区域 (n_windows, seq_len, n_features)
    """
    loader = LoadRenewableDataLoader(data_root)
    all_data = []

    for zone in zones:
        df = loader.load_zone(zone)
        dataset = TimeSeriesDataset(df, seq_len=seq_len, stride=stride,
                                    normalize=normalize)
        n_total = len(dataset)
        n_use = min(n_total, max_windows_per_zone)
        if 0.0 < data_fraction < 1.0:
            n_use = min(n_use, max(1, int(n_total * data_fraction)))

        windows = [dataset[i][0] for i in range(n_use)]
        all_data.append(np.stack(windows, axis=0))
        logger.info("Zone=%s windows=%d (from %d total)", zone, n_use, n_total)

    return all_data
