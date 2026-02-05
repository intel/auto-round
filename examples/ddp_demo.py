"""
Simple End-to-End DDP (DistributedDataParallel) Example

This script demonstrates how to train a simple neural network using PyTorch DDP
for distributed training across multiple GPUs.

Usage:
    # Using torchrun (recommended) - specify custom port to avoid conflicts
    torchrun --nproc_per_node=2 --master_port=29501 ddp_demo.py

    # Or using torch.multiprocessing.spawn (for single machine)
    python ddp_demo.py

Troubleshooting:
    # If you get "address already in use" error, use a different port:
    torchrun --nproc_per_node=2 --master_port=29502 ddp_demo.py

    # Or kill existing processes:
    pkill -f ddp_demo.py
"""

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# Simple synthetic dataset for demonstration
class SimpleDataset(Dataset):
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate random data and labels
        data = torch.randn(self.input_dim)
        label = torch.randn(5)  # 5 output classes
        return data, label


# Simple neural network model
class SimpleModel(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")

    # Initialize process group
    # Use 'nccl' backend for GPU training, 'gloo' for CPU
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()


def train_ddp(rank, world_size, epochs=5, batch_size=32):
    """
    Main training function for DDP.

    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        epochs: Number of training epochs
        batch_size: Batch size per process
    """
    print(f"Running DDP training on rank {rank}/{world_size}")

    # Setup distributed environment
    setup(rank, world_size)

    # Set device for this process
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # Create model and move to device
    model = SimpleModel().to(device)

    # Wrap model with DDP
    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[rank])
    else:
        ddp_model = DDP(model)

    # Create dataset and distributed sampler
    dataset = SimpleDataset(size=10)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(epochs):
        # Set epoch for sampler (important for proper shuffling)
        sampler.set_epoch(epoch)

        epoch_loss = 0.0
        num_batches = 0

        ddp_model.train()
        for batch_idx, (data, labels) in enumerate(dataloader):
            logger.info(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx} processing")
            # Move data to device
            data = data.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        # Calculate average loss
        avg_loss = epoch_loss / num_batches

        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}] completed, Average Loss: {avg_loss:.4f}")

    # Save checkpoint (only on rank 0)
    if rank == 0:
        checkpoint_path = "/tmp/ddp_model_checkpoint.pth"
        torch.save(
            {
                "epoch": epochs,
                "model_state_dict": ddp_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Checkpoint saved to {checkpoint_path}")

    # Wait for all processes to finish
    dist.barrier()

    # Cleanup
    cleanup()
    print(f"Rank {rank} finished training")


def main_spawn():
    """Main function using mp.spawn for single-machine multi-GPU training."""
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 2

    if world_size < 2:
        print("Warning: Running with only 1 process. DDP benefits are minimal.")
        world_size = 2  # Still spawn 2 processes for demonstration

    print(f"Starting DDP training with {world_size} processes")

    mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)

    print("Training completed!")


def main_torchrun():
    """Main function for torchrun-based execution."""
    # When using torchrun, environment variables are already set
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    print(f"Torchrun mode: rank={rank}, local_rank={local_rank}, world_size={world_size}")

    # Setup distributed environment
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend)

    # Set device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Train
    model = SimpleModel().to(device)
    if torch.cuda.is_available():
        ddp_model = DDP(model, device_ids=[local_rank])
    else:
        ddp_model = DDP(model)

    # Create dataset with distributed sampler
    dataset = SimpleDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    for epoch in range(5):
        sampler.set_epoch(epoch)
        epoch_loss = 0.0

        for batch_idx, (data, labels) in enumerate(dataloader):
            data = data.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0 and rank == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        if rank == 0:
            print(f"Epoch {epoch+1} completed, Avg Loss: {epoch_loss/len(dataloader):.4f}")

    dist.destroy_process_group()
    print(f"Rank {rank} finished!")


if __name__ == "__main__":
    # Check if running with torchrun (environment variables will be set)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        print("Detected torchrun environment, using torchrun mode")
        main_torchrun()
    else:
        print("Using mp.spawn mode for single-machine training")
        main_spawn()
