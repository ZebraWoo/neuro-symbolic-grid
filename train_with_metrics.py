#!/usr/bin/env python
"""
Spikformer pretraining script with full metrics and visualization support.
"""

import argparse
import torch
import logging
from pathlib import Path
import sys

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.spikformer_pretrain import SpikformerPretrainModel
from src.training.pretrain_training_with_metrics import TrainerWithMetrics, create_dataloaders
from src.data.load_renewable_dataset import LoadRenewableDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Spikformer pretraining with evaluation metrics')
    
    # Data arguments
    parser.add_argument('--data-root', type=str, 
                       default='/home/wuzuoxu/Data/PSML/Minute-level Load and Renewable',
                       help='Dataset root path')
    parser.add_argument('--zones', type=str, nargs='+', 
                       default=['CAISO', 'ERCOT', 'MISO', 'NYISO', 'PJM', 'SPP'],
                       help='Grid zones or zone prefixes to use')
    
    # Model arguments
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--embedding-dim', type=int, default=64, help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=720, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num-epochs', type=int, default=8, help='Number of epochs')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/spikformer_with_metrics',
                       help='Checkpoint directory')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto', help='Compute device')
    
    args = parser.parse_args()
    
    # Resolve device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create data loaders
    logger.info("Loading data...")
    train_loader, val_loader = create_dataloaders(
        args.data_root,
        args.zones,
        batch_size=args.batch_size,
        val_split=0.2,
        seq_len=args.seq_len,
    )
    
    logger.info(f"Train set size: {len(train_loader.dataset)}")
    logger.info(f"Validation set size: {len(val_loader.dataset)}")
    
    # Build model
    logger.info("Building model...")
    model = SpikformerPretrainModel(
        input_dim=11,
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim
    )
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    logger.info("Initializing trainer...")
    trainer = TrainerWithMetrics(model, device=device, learning_rate=args.learning_rate)
    
    # Start training
    logger.info("Starting training...")
    logger.info(f"Config: epochs={args.num_epochs}, batch_size={args.batch_size}, "
               f"lr={args.learning_rate}, hidden_dim={args.hidden_dim}")
    
    metrics = trainer.fit(
        train_loader,
        val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("[OK] Training completed")
    logger.info("=" * 70)
    logger.info(f"Checkpoint directory: {args.checkpoint_dir}")
    logger.info("Results directory: ./results")
    logger.info("\nGenerated plot files:")
    logger.info("  1. loss_curve.png - loss curve")
    logger.info("  2. metrics_curve.png - AUC, Precision, Recall, F1")
    logger.info("  3. clustering_metrics.png - Silhouette and proximity")


if __name__ == "__main__":
    main()
