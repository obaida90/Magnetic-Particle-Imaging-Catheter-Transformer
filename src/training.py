"""
Training and evaluation utilities for the Transformer model.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import time
import json
import os

def train_epoch(model, loader, optimizer, criterion, scaler=None, device='cuda'):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    for x, y in tqdm(loader, desc="Training"):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        
        with autocast():
            pred = model(x)
            loss = criterion(pred, y)
        
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        total_loss += loss.item()
        all_preds.append(pred.detach().cpu())
        all_targets.append(y.detach().cpu())
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    maes = [
        mean_absolute_error(all_targets[:, i], all_preds[:, i]) 
        for i in range(3)
    ]
    l2_error = torch.mean(torch.norm(all_targets - all_preds, dim=1)).item()
    
    return avg_loss, maes, l2_error

def evaluate(model, loader, criterion, device='cuda'):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            
            with autocast():
                pred = model(x)
                loss = criterion(pred, y)
            
            total_loss += loss.item()
            all_preds.append(pred.detach().cpu())
            all_targets.append(y.detach().cpu())
    
    avg_loss = total_loss / len(loader)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    
    maes = [
        mean_absolute_error(all_targets[:, i], all_preds[:, i]) 
        for i in range(3)
    ]
    l2_error = torch.mean(torch.norm(all_targets - all_preds, dim=1)).item()
    
    return avg_loss, maes, l2_error

def train_model(model, train_loader, val_loader, test_loader, 
                optimizer, criterion, scheduler, device='cuda',
                epochs=200, save_dir='checkpoints', use_amp=True):
    """
    Complete training loop with checkpointing and metrics logging.
    """
    os.makedirs(save_dir, exist_ok=True)
    scaler = GradScaler() if use_amp else None
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [], 'val_loss': [], 'test_loss': [],
        'train_mae_x': [], 'train_mae_y': [], 'train_mae_z': [], 'train_l2': [],
        'val_mae_x': [], 'val_mae_y': [], 'val_mae_z': [], 'val_l2': [],
        'test_mae_x': [], 'test_mae_y': [], 'test_mae_z': [], 'test_l2': [],
        'learning_rate': []
    }
    
    for epoch in range(epochs):
        current_lr = optimizer.param_groups[0]['lr']
        history['learning_rate'].append(current_lr)
        start_time = time.time()
        
        # Training
        train_loss, train_maes, train_l2 = train_epoch(
            model, train_loader, optimizer, criterion, scaler, device
        )
        
        # Validation and test
        val_loss, val_maes, val_l2 = evaluate(model, val_loader, criterion, device)
        test_loss, test_maes, test_l2 = evaluate(model, test_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['test_loss'].append(test_loss)
        history['train_l2'].append(train_l2)
        history['val_l2'].append(val_l2)
        history['test_l2'].append(test_l2)
        history['train_mae_x'].append(train_maes[0])
        history['train_mae_y'].append(train_maes[1])
        history['train_mae_z'].append(train_maes[2])
        history['val_mae_x'].append(val_maes[0])
        history['val_mae_y'].append(val_maes[1])
        history['val_mae_z'].append(val_maes[2])
        history['test_mae_x'].append(test_maes[0])
        history['test_mae_y'].append(test_maes[1])
        history['test_mae_z'].append(test_maes[2])
        
        # Print results
        print(f"\nEpoch {epoch+1:03d}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f"Train L2: {train_l2:.4f}, Val L2: {val_l2:.4f}, Test L2: {test_l2:.4f}")
        print(f"MAE X/Y/Z - Train: {train_maes[0]:.4f}/{train_maes[1]:.4f}/{train_maes[2]:.4f}")
        print(f"MAE X/Y/Z - Val: {val_maes[0]:.4f}/{val_maes[1]:.4f}/{val_maes[2]:.4f}")
        print(f"Time: {time.time()-start_time:.1f}s | LR: {current_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'metrics': {
                    'val_maes': val_maes,
                    'val_l2': val_l2,
                    'test_maes': test_maes,
                    'test_l2': test_l2
                }
            }
            torch.save(checkpoint, os.path.join(save_dir, 'best_model.pth'))
            print("*** New best model saved! ***")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(save_dir, 'final_model.pth'))
    
    # Save history
    for key in history:
        np.save(os.path.join(save_dir, f"{key}.npy"), np.array(history[key]))
    
    return history
