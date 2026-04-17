#!/usr/bin/env python
"""
Spikformer 预训练脚本 - 带完整的评估指标和可视化
支持 AUC、Precision、Recall、F1、Silhouette等指标
"""

import argparse
import torch
import logging
from pathlib import Path
import sys

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spikformer_pretrain import SpikformerPretrainModel
from src.training.pretrain_training_with_metrics import TrainerWithMetrics, create_dataloaders
from src.data.load_renewable_dataset import LoadRenewableDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Spikformer预训练 - 带评估指标')
    
    # 数据参数
    parser.add_argument('--data-root', type=str, 
                       default='/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable',
                       help='数据集根目录')
    parser.add_argument('--zones', type=str, nargs='+', 
                       default=['CAISO_zone_1_', 'CAISO_zone_2_'],
                       help='要使用的电网区域')
    
    # 模型参数
    parser.add_argument('--hidden-dim', type=int, default=128, help='隐藏维度')
    parser.add_argument('--embedding-dim', type=int, default=64, help='嵌入维度')
    
    # 训练参数
    parser.add_argument('--batch-size', type=int, default=16, help='批次大小')
    parser.add_argument('--seq-len', type=int, default=720, help='序列长度')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='学习率')
    parser.add_argument('--num-epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/spikformer_with_metrics',
                       help='检查点保存目录')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='计算设备')
    
    args = parser.parse_args()
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    if device == 'cuda':
        logger.info(f"GPU信息: {torch.cuda.get_device_name(0)}")
    
    # 创建数据加载器
    logger.info("加载数据...")
    train_loader, val_loader = create_dataloaders(
        args.data_root,
        args.zones,
        batch_size=args.batch_size,
        val_split=0.2
    )
    
    logger.info(f"训练集大小: {len(train_loader.dataset)}")
    logger.info(f"验证集大小: {len(val_loader.dataset)}")
    
    # 创建模型
    logger.info("创建模型...")
    model = SpikformerPretrainModel(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim
    )
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建训练器
    logger.info("初始化训练器...")
    trainer = TrainerWithMetrics(model, device=device, learning_rate=args.learning_rate)
    
    # 开始训练
    logger.info("开始训练...")
    logger.info(f"配置: epochs={args.num_epochs}, batch_size={args.batch_size}, "
               f"lr={args.learning_rate}, hidden_dim={args.hidden_dim}")
    
    metrics = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ 训练完成！")
    logger.info("=" * 70)
    logger.info(f"检查点目录: {args.checkpoint_dir}")
    logger.info(f"结果目录: ./results")
    logger.info("\n生成的可视化文件:")
    logger.info("  1. loss_curve.png - 损失曲线")
    logger.info("  2. metrics_curve.png - AUC、Precision、Recall、F1曲线")
    logger.info("  3. clustering_metrics.png - Silhouette和接近度曲线")


if __name__ == "__main__":
    main()
