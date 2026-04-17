"""
训练可视化脚本 - 绘制训练过程中的损失曲线
"""

import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

# 设置风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(checkpoint_dir):
    """从检查点目录绘制训练曲线"""
    checkpoint_path = Path(checkpoint_dir)
    
    # 查找最新的检查点
    checkpoints = sorted(checkpoint_path.glob('checkpoint_epoch_*.pt'))
    
    if not checkpoints:
        print(f"❌ 在 {checkpoint_dir} 中找不到检查点")
        return False
    
    # 加载最后一个检查点获取训练损失
    import torch
    latest_checkpoint = checkpoints[-1]
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    if 'train_losses' not in checkpoint or 'val_losses' not in checkpoint:
        print("❌ 检查点中没有损失信息")
        return False
    
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    
    # 创建图表
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 图1: 训练和验证损失
    axes[0].plot(train_losses, label='训练损失', linewidth=2, marker='o', markersize=4)
    axes[0].plot(val_losses, label='验证损失', linewidth=2, marker='s', markersize=4)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # 图2: 损失下降百分比
    improvement = [(train_losses[0] - loss) / train_losses[0] * 100 for loss in train_losses]
    axes[1].fill_between(range(len(improvement)), improvement, alpha=0.3)
    axes[1].plot(improvement, linewidth=2, marker='o', markersize=4, color='green')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('改善百分比 (%)', fontsize=12)
    axes[1].set_title('训练损失改善情况', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = checkpoint_path / 'training_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到: {output_path}")
    
    # 打印统计信息
    print(f"\n📊 训练统计:")
    print(f"  初始损失: {train_losses[0]:.4f}")
    print(f"  最终损失: {train_losses[-1]:.4f}")
    print(f"  改善幅度: {(train_losses[0] - train_losses[-1]) / train_losses[0] * 100:.2f}%")
    print(f"  最低验证损失: {min(val_losses):.4f} (Epoch {np.argmin(val_losses) + 1})")
    
    plt.show()
    return True


def create_summary_report(checkpoint_dir):
    """创建训练总结报告"""
    checkpoint_path = Path(checkpoint_dir)
    
    import torch
    checkpoints = sorted(checkpoint_path.glob('checkpoint_epoch_*.pt'))
    
    if not checkpoints:
        print(f"❌ 在 {checkpoint_dir} 中找不到检查点")
        return
    
    latest_checkpoint = checkpoints[-1]
    checkpoint = torch.load(latest_checkpoint, map_location='cpu')
    
    # 创建报告
    report = {
        'model': 'SpikformerPretrainModel',
        'task': '动态特征表示学习',
        'total_epochs': checkpoint.get('epoch', len(checkpoint.get('train_losses', []))) + 1,
        'initial_train_loss': float(checkpoint['train_losses'][0]) if checkpoint.get('train_losses') else None,
        'final_train_loss': float(checkpoint['train_losses'][-1]) if checkpoint.get('train_losses') else None,
        'best_val_loss': float(min(checkpoint['val_losses'])) if checkpoint.get('val_losses') else None,
        'best_val_epoch': int(np.argmin(checkpoint['val_losses']) + 1) if checkpoint.get('val_losses') else None,
    }
    
    # 保存报告
    report_path = checkpoint_path / 'training_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("\n📋 训练报告:")
    print(f"  模型: {report['model']}")
    print(f"  任务: {report['task']}")
    print(f"  总轮数: {report['total_epochs']}")
    print(f"  初始损失: {report['initial_train_loss']:.4f}")
    print(f"  最终损失: {report['final_train_loss']:.4f}")
    print(f"  最佳验证损失: {report['best_val_loss']:.4f} (第{report['best_val_epoch']}轮)")
    print(f"\n✅ 报告已保存到: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练可视化工具")
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints/spikformer_pretrain_low_mem',
        help='检查点目录'
    )
    
    args = parser.parse_args()
    
    print("🎨 生成训练可视化...")
    print()
    
    success = plot_training_curves(args.checkpoint_dir)
    if success:
        create_summary_report(args.checkpoint_dir)
