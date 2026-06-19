"""
Main training script for the Catheter Transformer model.
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
import argparse
from torch.optim.lr_scheduler import StepLR
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import CatheterTransformer
from src.dataset import load_data, create_data_loaders
from src.training import train_model
from src.utils import plot_metrics, setup_device, save_config

def main(args):
    # Setup
    device = setup_device(args.gpu_id)
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data_dir}...")
    X, Y = load_data(args.data_dir, mode=args.mode)
    print(f"Data shape: X={X.shape}, Y={Y.shape}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        X, Y, 
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        augment=args.augment,
        num_workers=args.num_workers,
        seed=args.seed
    )
    print(f"Data split: Train={len(train_loader.dataset)}, "
          f"Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    
    # Initialize model
    model = CatheterTransformer(
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        patch_size=args.patch_size,
        dropout_head=args.dropout_head,
        in_channels=args.in_channels,
        signal_len=args.signal_len
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Setup optimizer, criterion, scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    criterion = nn.MSELoss()
    scheduler = StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    
    # Save config
    config = vars(args)
    save_config(config, args.save_dir)
    
    # Train
    print("\nStarting training...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        save_dir=args.save_dir,
        use_amp=args.use_amp
    )
    
    # Plot and save metrics
    plot_metrics(history, args.plot_dir)
    print(f"\nTraining complete! Results saved to {args.save_dir}")
    
    # Final evaluation
    print("\nFinal evaluation on test set:")
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'))
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_maes, test_l2 = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test MAE (X, Y, Z): {test_maes[0]:.4f}, {test_maes[1]:.4f}, {test_maes[2]:.4f}")
    print(f"Test L2 Error: {test_l2:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Catheter Transformer")
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, 
                       default='/home/TrainingSets/FullStdMode', # Then change with BA and HBA mode data
                       help='Directory containing data')
    parser.add_argument('--mode', type=str, default='BA', choices=['BA', 'Standard'],
                       help='MPI mode: BA or Standard')
    parser.add_argument('--in_channels', type=int, default=2,
                       help='Number of input channels')
    parser.add_argument('--signal_len', type=int, default=80787,
                       help='Length of input signal')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=512,
                       help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--dropout_head', type=float, default=0.0,
                       help='Head dropout rate')
    parser.add_argument('--patch_size', type=int, default=32,
                       help='Patch size for 1D convolution')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Training set ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Validation set ratio')
    parser.add_argument('--scheduler_step', type=int, default=10,
                       help='Scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5,
                       help='Scheduler gamma')
    parser.add_argument('--augment', action='store_true', default=True,
                       help='Use data augmentation')
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision')
    parser.add_argument('--num_workers', type=int, default=2,
                       help='Number of data loading workers')
    
    # System parameters
    parser.add_argument('--gpu_id', type=int, default=0,
                       help='GPU ID to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--plot_dir', type=str, default='./plots',
                       help='Directory to save plots')
    
    args = parser.parse_args()
    main(args)
