"""
Data augmentation and dataset handling for MPI signals.
"""

import torch
from torch.utils.data import Dataset
import numpy as np

class AugmentedMPIDataset(Dataset):
    """
    MPI dataset with data augmentation for training.
    
    Args:
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target values
        augment (bool): Whether to apply augmentation
        noise_level (float): Standard deviation of Gaussian noise
        scale_range (tuple): Range for random amplitude scaling
    """
    def __init__(self, X, Y, augment=True, noise_level=0.05, scale_range=(0.8, 1.2)):
        self.X = X
        self.Y = Y
        self.augment = augment
        self.noise_level = noise_level
        self.scale_range = scale_range
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.Y[idx]
        
        if self.augment:
            # Apply amplitude scaling
            scale_factor = torch.empty(1).uniform_(*self.scale_range)
            x = x * scale_factor
            
            # Add Gaussian noise
            noise = torch.randn_like(x) * self.noise_level * torch.abs(x).mean()
            x = x + noise
            
            # Ensure non-negative values for physical quantities
            x = torch.clamp(x, min=0)
            
        return x, y

def load_data(data_dir, mode='BA'):
    """
    Load MPI data from numpy files.
    
    Args:
        data_dir (str): Directory containing the data
        mode (str): 'BA' or 'Standard'
    
    Returns:
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target values
    """
    X = np.load(os.path.join(data_dir, "X_train_scaled.npy"))
    Y = np.load(os.path.join(data_dir, "Y_train_scaled.npy"))
    
    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)
    
    return X, Y

def create_data_loaders(X, Y, batch_size=128, train_ratio=0.7, val_ratio=0.15, 
                       augment=True, num_workers=2, seed=42):
    """
    Create train, validation, and test data loaders.
    
    Args:
        X (torch.Tensor): Input features
        Y (torch.Tensor): Target values
        batch_size (int): Batch size
        train_ratio (float): Training set ratio
        val_ratio (float): Validation set ratio
        augment (bool): Apply augmentation to training set
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
    
    Returns:
        train_loader, val_loader, test_loader (DataLoader)
    """
    from torch.utils.data import DataLoader, random_split
    
    dataset_size = len(X)
    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    # Create augmented dataset for training
    augmented_dataset = AugmentedMPIDataset(X, Y, augment=augment)
    
    # Split data
    train_data, val_test_data = random_split(
        augmented_dataset, 
        [train_size, dataset_size - train_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    # Validation and test sets (no augmentation)
    val_data, test_data = random_split(
        AugmentedMPIDataset(val_test_data.dataset.X[val_test_data.indices], 
                           val_test_data.dataset.Y[val_test_data.indices], 
                           augment=False),
        [val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
