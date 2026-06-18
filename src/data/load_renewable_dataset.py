"""
分钟级负荷和可再生能源数据加载器
用于Spikformer预训练模型的数据准备
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoadRenewableDataLoader:
    """
    加载和处理PSML分钟级负荷和可再生能源数据
    
    数据结构：
    - CAISO_zone_*.csv: CAISO电网4个区域的负荷和可再生能源数据
    - ERCOT_zone_*.csv: ERCOT电网8个区域的数据
    - MISO_zone_*.csv: MISO电网6个区域的数据
    """
    
    def __init__(self, data_root: str):
        """
        Args:
            data_root: PSML数据集根目录 (通常是 /home/wuzuoxu/Data/PSML/Minute-level Load and Renewable/)
        """
        self.data_root = Path(data_root)
        self.zones = self._discover_zones()
        logger.info(f"发现 {len(self.zones)} 个电网区域")
    
    def _discover_zones(self) -> Dict[str, Path]:
        """扫描所有可用的电网区域数据文件"""
        zones = {}
        for csv_file in self.data_root.glob("*.csv"):
            if csv_file.name == "desktop.ini":
                continue
            # 文件名格式: GRID_zone_N_.csv (e.g., CAISO_zone_1_.csv)
            key = csv_file.stem  # 移除.csv后缀
            zones[key] = csv_file
        return zones
    
    def load_zone(self, zone_name: str) -> pd.DataFrame:
        """
        加载单个电网区域的数据
        
        Args:
            zone_name: 区域标识 (e.g., 'CAISO_zone_1_')
            
        Returns:
            DataFrame: 时间序列数据（索引为时间戳，列为特征）
        """
        if zone_name not in self.zones:
            raise ValueError(f"区域 {zone_name} 不存在。可用区域: {list(self.zones.keys())}")
        
        file_path = self.zones[zone_name]
        logger.info(f"加载 {zone_name} 数据... (文件大小: {file_path.stat().st_size / 1e9:.2f} GB)")
        
        # 加载CSV（可能很大，使用分块加载）
        df = pd.read_csv(file_path, index_col=0)
        
        # 处理时间索引
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning(f"{zone_name} 索引转换失败，保持原始格式")
        
        logger.info(f"加载完成: 形状 {df.shape}, 特征列: {list(df.columns)[:5]}...")
        return df
    
    def load_all_zones(self) -> Dict[str, pd.DataFrame]:
        """加载所有可用区域的数据"""
        all_data = {}
        for zone_name in sorted(self.zones.keys()):
            try:
                all_data[zone_name] = self.load_zone(zone_name)
            except Exception as e:
                logger.error(f"加载 {zone_name} 失败: {e}")
        return all_data
    
    def get_feature_statistics(self, df: pd.DataFrame) -> Dict[str, float]:
        """计算特征统计信息用于归一化"""
        stats = {
            'mean': df.mean().values,
            'std': df.std().values,
            'min': df.min().values,
            'max': df.max().values,
        }
        return stats
    
    def normalize_minmax(self, df: pd.DataFrame) -> pd.DataFrame:
        """Min-Max归一化到[0, 1]范围"""
        return (df - df.min()) / (df.max() - df.min() + 1e-8)
    
    def normalize_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score标准化"""
        return (df - df.mean()) / (df.std() + 1e-8)


class TimeSeriesDataset:
    """
    将长时间序列转换为固定长度的样本
    用于模型训练
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 seq_len: int = 1440,  # 24小时 (分钟级)
                 stride: int = 360,     # 6小时步长
                 normalize: str = 'zscore'):
        """
        Args:
            data: 时间序列数据
            seq_len: 序列长度（分钟数）
            stride: 滑动窗口步长
            normalize: 归一化方法 ('zscore', 'minmax', 'none')
        """
        self.seq_len = seq_len
        self.stride = stride
        
        # 只选择数值列
        data = data.select_dtypes(include=['number'])
        
        # 归一化
        if normalize == 'zscore':
            self.data = (data - data.mean()) / (data.std() + 1e-8)
        elif normalize == 'minmax':
            self.data = (data - data.min()) / (data.max() - data.min() + 1e-8)
        else:
            self.data = data
        
        # 去掉缺失值
        self.data = self.data.dropna()
        
        # 生成滑动窗口样本索引
        self.indices = []
        for i in range(0, len(self.data) - seq_len + 1, stride):
            self.indices.append(i)
        
        logger.info(f"创建数据集: {len(self.indices)} 个样本，序列长度 {seq_len}")
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, dict]:
        """
        获取单个样本
        
        Returns:
            x: (seq_len, num_features) 的时间序列
            info: 包含样本元数据的字典
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.seq_len
        
        x = self.data.iloc[start_idx:end_idx].values.astype(np.float32)
        
        info = {
            'start_time': self.data.index[start_idx],
            'end_time': self.data.index[end_idx - 1],
            'zone_idx': idx % len(self.data),
        }
        
        return x, info
    
    def get_batch(self, indices: List[int]) -> Tuple[np.ndarray, List[dict]]:
        """获取一个批次的样本"""
        samples = []
        infos = []
        for idx in indices:
            x, info = self[idx]
            samples.append(x)
            infos.append(info)
        
        # 堆叠为批次张量 (batch_size, seq_len, num_features)
        batch_x = np.stack(samples, axis=0)
        return batch_x, infos


def create_pretrain_dataloader(data_root: str,
                               seq_len: int = 1440,
                               stride: int = 360,
                               batch_size: int = 32,
                               normalize: str = 'zscore',
                               zones: List[str] = None) -> Tuple[TimeSeriesDataset, Dict[str, np.ndarray]]:
    """
    创建用于预训练的数据加载器
    
    Args:
        data_root: 数据根目录
        seq_len: 序列长度（分钟）
        stride: 滑动窗口步长
        batch_size: 批次大小
        normalize: 归一化方法
        zones: 要使用的特定区域列表，None表示全部
        
    Returns:
        dataset: 时间序列数据集
        statistics: 所有区域的统计信息
    """
    loader = LoadRenewableDataLoader(data_root)
    
    # 加载指定的区域数据
    if zones is None:
        zones_to_load = list(loader.zones.keys())
    else:
        zones_to_load = zones
    
    all_data = []
    all_stats = {}
    
    for zone_name in zones_to_load:
        df = loader.load_zone(zone_name)
        all_data.append(df)
        all_stats[zone_name] = loader.get_feature_statistics(df)
    
    # 合并所有区域数据
    combined_data = pd.concat(all_data, ignore_index=False)
    
    # 创建数据集
    dataset = TimeSeriesDataset(combined_data, seq_len, stride, normalize)
    
    return dataset, all_stats


if __name__ == "__main__":
    # 测试脚本
    data_root = "/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable"
    
    # 创建数据加载器
    loader = LoadRenewableDataLoader(data_root)
    print(f"可用区域: {list(loader.zones.keys())}")
    
    # 加载单个区域
    df = loader.load_zone('CAISO_zone_1_')
    print(f"\n数据形状: {df.shape}")
    print(f"首行:\n{df.head()}")
    
    # 创建数据集
    dataset = TimeSeriesDataset(df, seq_len=1440, stride=360)
    print(f"\n数据集大小: {len(dataset)}")
    
    # 获取样本
    x, info = dataset[0]
    print(f"样本形状: {x.shape}")
    print(f"样本时间范围: {info['start_time']} 到 {info['end_time']}")
