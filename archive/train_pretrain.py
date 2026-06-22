#!/usr/bin/env python3
"""
Spikformer预训练模型快速启动脚本
分钟级负荷数据的动态特征表示学习
"""

import sys
import argparse
import logging
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Spikformer预训练模型训练脚本"
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        default='/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable',
        help='PSML数据集根目录'
    )
    
    parser.add_argument(
        '--zones',
        type=str,
        nargs='+',
        default=['CAISO_zone_1_', 'CAISO_zone_2_', 'ERCOT_zone_1_', 'ERCOT_zone_2_'],
        help='要使用的电网区域列表'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='批次大小'
    )
    
    parser.add_argument(
        '--seq-len',
        type=int,
        default=1440,
        help='序列长度（分钟数，默认24小时）'
    )
    
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='隐藏层维度'
    )
    
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=128,
        help='嵌入维度'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='学习率'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='训练轮数'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/spikformer_pretrain',
        help='检查点保存目录'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cuda', 'cpu'],
        help='计算设备'
    )
    
    args = parser.parse_args()
    
    # 导入训练模块
    import torch
    from src.training.pretrain_training import (
        create_dataloaders,
        PretrainingTrainer,
    )
    from src.models.spikformer_pretrain import SpikformerPretrainModel
    
    # 确定设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"使用设备: {device}")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader = create_dataloaders(
        data_root=args.data_root,
        zones=args.zones,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_split=0.8,
    )
    
    # 获取输入维度
    sample_batch = next(iter(train_loader))
    if isinstance(sample_batch, (list, tuple)):
        sample_batch = sample_batch[0]
    if isinstance(sample_batch, list):
        sample_batch = sample_batch[0]
    input_dim = sample_batch.shape[-1]
    logger.info(f"输入维度: {input_dim}")
    
    # 创建模型
    logger.info("创建预训练模型...")
    model = SpikformerPretrainModel(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        num_encoder_layers=4,
    )
    
    # 模型统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    # 创建训练器
    trainer = PretrainingTrainer(
        model=model,
        device=device,
        learning_rate=args.learning_rate,
    )
    
    # 开始训练
    logger.info("=" * 50)
    logger.info("开始预训练")
    logger.info("=" * 50)
    
    train_losses, val_losses = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.checkpoint_dir,
    )
    
    logger.info("=" * 50)
    logger.info("预训练完成！")
    logger.info(f"模型保存到: {args.checkpoint_dir}")
    logger.info("=" * 50)
    
    # 总结
    print("\n" + "=" * 50)
    print("预训练总结")
    print("=" * 50)
    print(f"最终训练损失: {train_losses[-1]:.4f}")
    print(f"最终验证损失: {val_losses[-1]:.4f}")
    print(f"模型保存位置: {args.checkpoint_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
