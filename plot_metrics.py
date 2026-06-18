#!/usr/bin/env python
"""
Training metrics visualization script.
Plots training curves for AUC, Precision, Recall, F1, Silhouette, and summary stats.
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

# Plot style
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plot_all_metrics(metrics_file='./results/metrics.json', output_dir='./results'):
    """Plot all available metrics."""
    
    # Load metrics
    metrics_path = Path(metrics_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not metrics_path.exists():
        logger.error(f"[ERROR] Metrics file not found: {metrics_path}")
        logger.info("[INFO] Run train_with_metrics.py first")
        return False
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    logger.info(f"[OK] Loaded metrics file: {metrics_path}")
    print("\nDetected metrics:")
    for key in metrics.keys():
        if metrics[key]:
            print(f"  - {key}: {len(metrics[key])} data points")
        else:
            print(f"  - {key}: no data")
    print()
    
    # 1. Loss curves
    if metrics['train_loss'] and metrics['val_loss']:
        fig, ax = plt.subplots(figsize=(12, 6))
        epochs = range(1, len(metrics['train_loss']) + 1)
        ax.plot(epochs, metrics['train_loss'], 'o-', label='Train Loss', linewidth=2.5, markersize=6)
        ax.plot(epochs, metrics['val_loss'], 's-', label='Validation Loss', linewidth=2.5, markersize=6)
        ax.set_xlabel('Epoch', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss', fontsize=13, fontweight='bold')
        ax.set_title('Training and Validation Loss', fontsize=15, fontweight='bold')
        ax.legend(fontsize=12, loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / 'loss_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info("[OK] Saved: loss_curve.png")
    
    # 2. Evaluation metrics
    metrics_to_plot = [
        ('val_auc', 'AUC (Area Under Curve)', 'AUC Score', '#1f77b4'),
        ('val_precision', 'Precision', 'Precision', '#ff7f0e'),
        ('val_recall', 'Recall', 'Recall', '#2ca02c'),
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
            ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.3, label='Baseline')
            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
            
            # Show best and final values
            best_val = max(metrics[key])
            best_epoch = metrics[key].index(best_val) + 1
            final_val = metrics[key][-1]
            
            textstr = f'Best: {best_val:.4f} (Epoch {best_epoch})\nFinal: {final_val:.4f}'
            ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, f'{key}\nNot available', ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    
    plt.suptitle('Classification Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("[OK] Saved: metrics_curve.png")
    
    # 3. Clustering and anomaly metrics
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
        
        textstr = f'Best: {best_val:.4f} (Epoch {best_epoch})'
        axes[0].text(0.98, 0.05, textstr, transform=axes[0].transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    else:
        axes[0].text(0.5, 0.5, 'Silhouette Score\nNot available', ha='center', va='center', fontsize=12)
        axes[0].set_xticks([])
        axes[0].set_yticks([])
    
    axes[0].set_title('Silhouette Score', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    
    # Proximity
    if metrics['train_proximity'] and metrics['val_proximity']:
        epochs = range(1, len(metrics['train_proximity']) + 1)
        axes[1].plot(epochs, metrics['train_proximity'], 'o-', label='Train Proximity', 
                    linewidth=2.5, markersize=6)
        axes[1].plot(epochs, metrics['val_proximity'], 's-', label='Validation Proximity', 
                    linewidth=2.5, markersize=6)
        axes[1].set_ylabel('Proximity Score', fontsize=11, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'Proximity\nNot available', ha='center', va='center', fontsize=12)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    axes[1].set_title('Proximity', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    
    plt.suptitle('Clustering and Anomaly Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'clustering_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("[OK] Saved: clustering_metrics.png")
    
    # 4. Summary report
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    summary_text = "Training Summary Report\n" + "=" * 60 + "\n\n"
    
    # Loss statistics
    if metrics['train_loss']:
        improvement = 0.0
        if metrics['train_loss'][0] != 0:
            improvement = (metrics['train_loss'][0] - metrics['train_loss'][-1]) / metrics['train_loss'][0] * 100
        summary_text += "Loss Metrics:\n"
        summary_text += f"  - Initial train loss: {metrics['train_loss'][0]:.6e}\n"
        summary_text += f"  - Final train loss: {metrics['train_loss'][-1]:.6e}\n"
        summary_text += f"  - Improvement: {improvement:.2f}%\n"
        summary_text += f"  - Best validation loss: {min(metrics['val_loss']):.6e}\n"
        summary_text += f"  - Best epoch: {metrics['val_loss'].index(min(metrics['val_loss'])) + 1}\n\n"
    
    # AUC statistics
    if metrics['val_auc']:
        summary_text += "AUC Metrics:\n"
        summary_text += f"  - Best AUC: {max(metrics['val_auc']):.4f}\n"
        summary_text += f"  - Worst AUC: {min(metrics['val_auc']):.4f}\n"
        summary_text += f"  - Mean AUC: {np.mean(metrics['val_auc']):.4f}\n"
        summary_text += f"  - Final AUC: {metrics['val_auc'][-1]:.4f}\n\n"
    
    # Precision / Recall / F1 statistics
    if metrics['val_precision']:
        summary_text += "Classification Metrics:\n"
        summary_text += f"  - Precision: {metrics['val_precision'][-1]:.4f} (best: {max(metrics['val_precision']):.4f})\n"
    if metrics['val_recall']:
        summary_text += f"  - Recall: {metrics['val_recall'][-1]:.4f} (best: {max(metrics['val_recall']):.4f})\n"
    if metrics['val_f1']:
        summary_text += f"  - F1-Score: {metrics['val_f1'][-1]:.4f} (best: {max(metrics['val_f1']):.4f})\n\n"
    
    # Silhouette statistics
    if metrics['val_silhouette']:
        summary_text += "Silhouette Metrics:\n"
        summary_text += f"  - Best score: {max(metrics['val_silhouette']):.4f}\n"
        summary_text += f"  - Worst score: {min(metrics['val_silhouette']):.4f}\n"
        summary_text += f"  - Mean score: {np.mean(metrics['val_silhouette']):.4f}\n"
        summary_text += f"  - Final score: {metrics['val_silhouette'][-1]:.4f}\n\n"
    
    # Proximity statistics
    if metrics['val_proximity']:
        summary_text += "Proximity Metrics:\n"
        summary_text += f"  - Final train proximity: {metrics['train_proximity'][-1]:.4f}\n"
        summary_text += f"  - Final validation proximity: {metrics['val_proximity'][-1]:.4f}\n"
        summary_text += f"  - Mean validation proximity: {np.mean(metrics['val_proximity']):.4f}\n\n"
    
    summary_text += "=" * 60 + "\n"
    summary_text += "Training complete. All metrics have been saved."
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', horizontalalignment='left', family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'summary_report.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("[OK] Saved: summary_report.png")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Training metrics visualization')
    parser.add_argument('--metrics-file', type=str, default='./results/metrics.json',
                       help='Metrics JSON path')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Output directory')
    
    args = parser.parse_args()
    
    print("Generating training metric visualizations...\n")
    
    success = plot_all_metrics(args.metrics_file, args.output_dir)
    
    if success:
        print("\nVisualization complete.")
        print(f"Output directory: {args.output_dir}")
        print("\nGenerated files:")
        print("  1. loss_curve.png - Training and validation loss")
        print("  2. metrics_curve.png - AUC, Precision, Recall, F1")
        print("  3. clustering_metrics.png - Silhouette and proximity")
        print("  4. summary_report.png - Summary report")
    else:
        print("\nVisualization failed")


if __name__ == "__main__":
    main()
