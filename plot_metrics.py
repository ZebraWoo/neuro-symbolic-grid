#!/usr/bin/env python
"""
训练指标可视化脚本
用于绘制AUC、Precision、Recall、F1、Silhouette等指标的训练曲线
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置中文字体和风格
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_all_metrics(metrics_file='./results/metrics.json', output_dir='./results'):
    """绘制所有指标"""
    
    # 加载指标数据
    metrics_path = Path(metrics_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_path.exists():
        logger.error(f"❌ 指标文件不存在: {metrics_path}")
        logger.info("📌 请先运行 train_with_metrics.py 进行训练")
        return False
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"✅ 已加载指标文件: {metrics_path}")
    print("\n📊 检测到的指标：")
    for key in metrics.keys():
        if metrics[key]:
            print(f"  ✓ {key}: {len(metrics[key])} 个数据点")
        else:
            print(f"  ✗ {key}: 无数据")
    print()
    
    # 1. 损失曲线（单独一个图）
    if metrics['train_loss'] and metrics['val_loss']:
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax.plot(epochs, metrics['train_loss'], 'o-', label='训练损失', linewidth=2.5, markersize=6)
        ax.plot(epochs, metrics['val_loss'], 's-', label='验证损失', linewidth=2.5, markersize=6)
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title('训练和验证损失曲线', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / '1_loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("✅ 损失曲线已保存: loss_curve.png")
    
    # 2. 评估指标 - 2x2 子图
    metrics_to_plot = [
        ('val_auc', 'AUC (Area Under Curve)', 'AUC分数', '#1f77b4'),
        ('val_precision', 'Precision (精确率)', 'Precision', '#ff7f0e'),
        ('val_recall', 'Recall (召回率)', 'Recall', '#2ca02c'),
        ('val_f1', 'F1-Score', 'F1 Score', '#d62728'),
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (key, title, ylabel, color) in enumerate(metrics_to_plot):
        ax = axes[idx]
        if metrics[key]:
            epochs = range(1, len(metrics[key]) + 1)
            ax.plot(epochs, metrics[key], 'o-', linewidth=2.5, markersize=6, color=color)
            ax.set_ylim([0, 1.05])
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='基准')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # 显示最佳值和最终值
            best_val = max(metrics[key])
            best_epoch = metrics[key].index(best_val) + 1
            final_val = metrics[key][-1]
            
            textstr = f'最佳: {best_val:.4f} (Epoch {best_epoch})\n最终: {final_val:.4f}'
            ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'{key}\n未计算', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    
    plt.suptitle('分类评估指标', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '2_metrics_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✅ 评估指标曲线已保存: metrics_curve.png")
    
    # 3. 聚类和异常检测指标
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Silhouette Score
    if metrics['val_silhouette']:
        epochs = range(1, len(metrics['val_silhouette']) + 1)
        axes[0].plot(epochs, metrics['val_silhouette'], 'o-', linewidth=2.5, markersize=6, color='purple')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_ylim([-1.05, 1.05])
        axes[0].set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        best_val = max(metrics['val_silhouette'])
        best_epoch = metrics['val_silhouette'].index(best_val) + 1
        
        textstr = f'最佳: {best_val:.4f} (Epoch {best_epoch})'
        axes[0].text(0.98, 0.05, textstr, transform=axes[0].transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        axes[0].text(0.5, 0.5, 'Silhouette Score\n未计算', ha='center', va='center', fontsize=12)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    
    axes[0].set_title('Silhouette Score (聚类质量)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    
    # Proximity
    if metrics['train_proximity'] and metrics['val_proximity']:
        epochs = range(1, len(metrics['train_proximity']) + 1)
        axes[1].plot(epochs, metrics['train_proximity'], 'o-', label='训练接近度', 
                    linewidth=2.5, markersize=6)
        axes[1].plot(epochs, metrics['val_proximity'], 's-', label='验证接近度', 
                    linewidth=2.5, markersize=6)
        axes[1].set_ylabel('Proximity Score', fontsize=11, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, '接近度 (Proximity)\n未计算', ha='center', va='center', fontsize=12)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    axes[1].set_title('接近度 (异常检测指标)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    
    plt.suptitle('聚类和异常检测指标', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / '3_clustering_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✅ 聚类指标曲线已保存: clustering_metrics.png")
    
    # 4. 统计信息总结
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    summary_text = "📊 训练统计总结\n" + "=" * 60 + "\n\n"
    
    # 损失统计
    if metrics['train_loss']:
        summary_text += "📈 损失指标:\n"
        summary_text += f"  • 初始训练损失: {metrics['train_loss'][0]:.6f}\n"
        summary_text += f"  • 最终训练损失: {metrics['train_loss'][-1]:.6f}\n"
        summary_text += f"  • 改善幅度: {(metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100:.2f}%\n"
        summary_text += f"  • 最低验证损失: {min(metrics['val_loss']):.6f}\n"
        summary_text += f"  • 最低值出现轮数: Epoch {metrics['val_loss'].index(min(metrics['val_loss'])) + 1}\n\n"
    
    # AUC统计
    if metrics['val_auc']:
        summary_text += "🎯 AUC指标:\n"
        summary_text += f"  • 最高AUC: {max(metrics['val_auc']):.4f}\n"
        summary_text += f"  • 最低AUC: {min(metrics['val_auc']):.4f}\n"
        summary_text += f"  • 平均AUC: {np.mean(metrics['val_auc']):.4f}\n"
        summary_text += f"  • 最终AUC: {metrics['val_auc'][-1]:.4f}\n\n"
    
    # Precision/Recall/F1统计
    if metrics['val_precision']:
        summary_text += "📐 分类指标:\n"
        summary_text += f"  • Precision: {metrics['val_precision'][-1]:.4f} (最高: {max(metrics['val_precision']):.4f})\n"
    if metrics['val_recall']:
        summary_text += f"  • Recall: {metrics['val_recall'][-1]:.4f} (最高: {max(metrics['val_recall']):.4f})\n"
    if metrics['val_f1']:
        summary_text += f"  • F1-Score: {metrics['val_f1'][-1]:.4f} (最高: {max(metrics['val_f1']):.4f})\n\n"
    
    # Silhouette统计
    if metrics['val_silhouette']:
        summary_text += "🔷 聚类质量 (Silhouette):\n"
        summary_text += f"  • 最高分数: {max(metrics['val_silhouette']):.4f}\n"
        summary_text += f"  • 最低分数: {min(metrics['val_silhouette']):.4f}\n"
        summary_text += f"  • 平均分数: {np.mean(metrics['val_silhouette']):.4f}\n"
        summary_text += f"  • 最终分数: {metrics['val_silhouette'][-1]:.4f}\n\n"
    
    # 接近度统计
    if metrics['val_proximity']:
        summary_text += "📍 接近度 (异常检测):\n"
        summary_text += f"  • 训练接近度: {metrics['train_proximity'][-1]:.4f}\n"
        summary_text += f"  • 验证接近度: {metrics['val_proximity'][-1]:.4f}\n"
        summary_text += f"  • 平均验证接近度: {np.mean(metrics['val_proximity']):.4f}\n\n"
    
    summary_text += "=" * 60 + "\n"
    summary_text += "✅ 训练完成！所有指标已保存。"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='left', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / '4_summary_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("✅ 总结报告已保存: summary_report.png")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='训练指标可视化')
    parser.add_argument('--metrics-file', type=str, default='./results/metrics.json',
                       help='指标JSON文件路径')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='输出目录')
    
    args = parser.parse_args()
    
    print("🎨 开始生成训练指标可视化...\n")
    
    success = plot_all_metrics(args.metrics_file, args.output_dir)
    
    if success:
        print("\n✅ 可视化完成！")
        print(f"📁 输出目录: {args.output_dir}")
        print("\n生成的图表:")
        print("  1. 1_loss_curve.png - 训练和验证损失曲线")
        print("  2. 2_metrics_curve.png - AUC、Precision、Recall、F1曲线")
        print("  3. 3_clustering_metrics.png - Silhouette和接近度曲线")
        print("  4. 4_summary_report.png - 统计总结报告")
    else:
        print("\n❌ 可视化失败")


if __name__ == "__main__":
    main()
