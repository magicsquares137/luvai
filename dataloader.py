import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class GPTDataset(Dataset):
    """Dataset for GPT training with controlled iterations"""
    
    def __init__(self, data_path, block_size, max_iters=None, batch_size=12):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size
        self.total_size = len(self.data) - self.block_size
        self.batch_size = batch_size
        
        # If max_iters is set, limit the dataset size to just what we need
        if max_iters is not None:
            self.size = max_iters * batch_size
        else:
            self.size = self.total_size
            
        print(f"Dataset has {self.total_size:,} potential samples")
        print(f"Using {self.size:,} samples ({self.size/self.total_size:.2%} of dataset)")
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # Always sample randomly from the full dataset
        # This ensures we use the entire dataset even with limited iterations
        real_idx = torch.randint(0, self.total_size, (1,)).item()
            
        # Get block starting at idx
        x = torch.from_numpy((self.data[real_idx:real_idx+self.block_size]).astype(np.int64))
        # Target is the same sequence shifted by 1
        y = torch.from_numpy((self.data[real_idx+1:real_idx+1+self.block_size]).astype(np.int64))
        return x, y

def create_dataloaders(
    data_dir,
    dataset_name,
    block_size,
    batch_size,
    distributed=False,
    world_size=1,
    rank=0,
    seed=1337,
    shuffle=True,
    num_workers=4,
    pin_memory=True
):
    """Create train and validation dataloaders with proper tracking of epochs"""
    
    train_dataset = GPTDataset(
        os.path.join(data_dir, dataset_name, 'train.bin'),
        block_size
    )
    
    val_dataset = GPTDataset(
        os.path.join(data_dir, dataset_name, 'val.bin'),
        block_size
    )
    
    # Create samplers for distributed training
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )
        # We usually don't need to shuffle the validation set
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed
        )
    else:
        train_sampler = None
        val_sampler = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None and shuffle),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Important for FSDP to have consistent batch sizes
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return train_loader, val_loader

# Example usage in main training script:
"""
# Replace the existing data loading code with:

train_loader, val_loader = create_dataloaders(
    data_dir='data',
    dataset_name=dataset,
    block_size=block_size,
    batch_size=batch_size,
    distributed=distributed,
    world_size=world_size,
    rank=rank,
    seed=1337 + seed_offset,
    shuffle=True,
    num_workers=4,
    pin_memory=(device_type == 'cuda')
)

# Then in the training loop:
for epoch in range(num_epochs):
    # Set epoch for distributed sampler
    if distributed:
        train_loader.sampler.set_epoch(epoch)
    
    # Training loop
    model.train()
    for i, (X, Y) in enumerate(train_loader):
        # Move data to device
        X, Y = X.to(device), Y.to(device)
        
        # Forward, backward, optimize steps...
        
    # Validation loop
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            # Evaluation steps...
"""