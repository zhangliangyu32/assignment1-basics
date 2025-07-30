import torch
from einops import rearrange, einsum, reduce
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val, _ = torch.max(x, dim=dim, keepdim=True)
    x_stabilized = x - max_val

    exps = torch.exp(x_stabilized)

    sum_exps = torch.sum(exps, dim=dim, keepdim=True)

    return exps / sum_exps

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss between logits and targets.
    Args:
        logits (torch.Tensor): Logits from the model.
        targets (torch.Tensor): Ground truth labels.
    Returns:
        torch.Tensor: Computed cross-entropy loss.
    """
    max_val, _ = torch.max(logits, dim=-1, keepdim=True)
    logits_stabilized = logits - max_val
    vector_loss = -logits_stabilized + torch.logsumexp(logits_stabilized, dim=-1, keepdim=True)
    vector_loss = torch.gather(vector_loss, dim=-1, index=targets.unsqueeze(-1))
    average_loss = torch.mean(vector_loss)
    return average_loss.squeeze()
