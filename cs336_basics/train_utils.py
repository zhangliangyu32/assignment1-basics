import numpy as np
import torch
import os
from typing import BinaryIO, IO
import tqdm
import wandb
import cs336_basics.utils as utils
def data_loader(x: np.ndarray, batch_size: int = 16, context_len: int = 10, device: str = "cpu"):
    """
    Data loader for training a model with context windows.
    
    Args:
        x (np.ndarray): Input data, 1 dimensional numpy arrays consisting of token ids.
        batch_size (int): Number of samples per batch.
        context_len (int): Length of the context window.
        device (torch.device): Device to load the tensors onto.
    """
    device = torch.device(device)
    n = len(x)
    start_indices = np.random.randint(0, n - context_len, size=batch_size)
    x_batch = np.array([x[i:i + context_len] for i in start_indices])
    x_batch = torch.tensor(x_batch, device=device)
    target_batch = np.array([x[i + 1:i + context_len + 1] for i in start_indices])
    target_batch = torch.tensor(target_batch, device=device)
    return x_batch, target_batch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    """
    Save the model and optimizer state dicts to disk.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        iteration: The current iteration number.
        out: The output file path to save the checkpoint.
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, os.path.join(out))

def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load the model and optimizer state dicts from disk.

    Args:
        model: The model to load the state dict into.
        optimizer: The optimizer to load the state dict into.
        checkpoint_path: The path to the checkpoint file.
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]

