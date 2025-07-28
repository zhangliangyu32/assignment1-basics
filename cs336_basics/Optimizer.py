from collections.abc import Iterable, Callable
from typing import Iterator, Optional
import torch
import math
import pickle
class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = {"lr": lr}
        super(SGD, self).__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0) 
                grad = p.grad.data
                p.data -= lr / (math.sqrt(t + 1)) * grad
                state["t"] = t + 1
        return loss

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError("Invalid learning rate: {}".format(alpha))
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super(AdamW, self).__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                grad = p.grad.data
                exp_avg.mul_(betas[0]).add_(grad, alpha=1 - betas[0])
                exp_avg_sq.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])
                denom = (exp_avg_sq.sqrt() + eps)
                corrected_lr = lr * math.sqrt(1 - betas[1] ** state["step"]) / (1 - betas[0] ** state["step"])
                p.data.addcdiv_(exp_avg, denom, value=-corrected_lr)
                if weight_decay != 0:
                    p.data.add_(-weight_decay * lr * p.data)
        return loss

    
if __name__ == "__main__":
    # Example usage
    weights = torch.nn.Parameter(5 * torch.randn(10, 10))
    opt = SGD([weights], lr=1000)
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(loss.cpu().item())
        loss.backward()
        opt.step()

def cos_lr_scheduler(t: int, alpha_max: float, alpha_min: float, warmup_steps: int, total_steps: int) -> float:
    """
    A cosine learning rate scheduler with warmup.
    
    Args:
        t (int): The current training step.
        alpha_max (float): The maximum learning rate.
        alpha_min (float): The minimum learning rate.
        warmup_steps (int): The number of warmup steps.
        total_steps (int): The total number of training steps.

    Returns:
        float: The learning rate for the current step.
    """
    if t < warmup_steps:
        return alpha_max * (t / warmup_steps)
    elif t <= total_steps:
        return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + math.cos(math.pi * (t - warmup_steps) / (total_steps - warmup_steps)))
    else:
        return alpha_min
    
def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_norm: float,
    eps: float = 1e-6,
) -> None:
    """
    Clips gradients of an iterable of parameters.

    Args:
        parameters (Iterable[torch.nn.Parameter]): Iterable of parameters to clip gradients for.
        max_norm (float): Maximum norm of the gradients.
        norm_type (float): Type of the norm to use for clipping.
    """
    print(max_norm)
    grad_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            grad_norm  += param.grad.data.norm("fro") ** 2
    grad_norm = grad_norm ** 0.5
    if grad_norm > max_norm:
        for param in parameters:
            if param.grad is not None:
                param.grad.data.mul_(max_norm / (grad_norm + eps))
    return