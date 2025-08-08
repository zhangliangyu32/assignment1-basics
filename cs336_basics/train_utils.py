import numpy as np
import torch
import os
import io
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
    # deal with the corner case when computing validation loss
    if n == context_len: 
        start_indices = np.array([0] * batch_size)
    else:
        start_indices = np.random.randint(0, n - context_len, size=batch_size)
    x_batch = np.array([x[i:i + context_len] for i in start_indices])
    x_batch = torch.tensor(x_batch, device=device)
    target_batch = np.array([x[i + 1:i + context_len + 1] for i in start_indices])
    target_batch = torch.tensor(target_batch, device=device)
    return x_batch, target_batch

# def compute_validation_loss(model: torch.nn.Module, x_val: np.ndarray, context_len: int = 256, device: str = "cpu"):
#     """
#     Compute the validation loss for the model.
    
#     Args:
#         model (torch.nn.Module): The model to evaluate.
#         x_val (np.ndarray): Validation data, 1 dimensional numpy arrays consisting of token ids.
#         context_len (int): Length of the context window.
#         device (str): Device to load the tensors onto.
    
#     Returns:
#         float: The average validation loss.
#     """
#     model.eval()
#     device = torch.device(device)
#     total_loss = 0.0
#     n_batches = len(x_val) // context_len
#     with torch.no_grad():
#         for i in tqdm.tqdm(range(n_batches)):
#             x_batch, target_batch = data_loader(x_val[i * context_len:(i + 1) * context_len], batch_size=1, context_len=context_len, device=device)
#             logits = model(x_batch)
#             loss = utils.cross_entropy_loss(logits, target_batch)
#             total_loss += loss.item()
#     return total_loss / n_batches

def compute_validation_loss(model: torch.nn.Module, x_val: np.ndarray, context_len: int = 256, n_batches: int = 500, batch_size: int = 32, device: str = "cpu"):
    """
    Compute the validation loss for the model.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        x_val (np.ndarray): Validation data, 1 dimensional numpy arrays consisting of token ids.
        context_len (int): Length of the context window.
        device (str): Device to load the tensors onto.
    
    Returns:
        float: The average validation loss.
    """
    model.eval()
    device = torch.device(device)
    total_loss = 0.0
    with torch.no_grad():
        for i in tqdm.tqdm(range(n_batches)):
            x_batch, target_batch = data_loader(x_val, batch_size=batch_size, context_len=context_len, device=device)
            logits = model(x_batch)
            loss = utils.cross_entropy_loss(logits, target_batch)
            total_loss += loss.item()
    return total_loss / n_batches

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
        src: The path to the checkpoint file or a file-like object.
    """
    # If src is a file-like object that's not seekable, load it into a BytesIO buffer
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]