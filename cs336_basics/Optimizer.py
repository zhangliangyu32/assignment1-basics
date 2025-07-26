from collections.abc import Iterable, Callable
from typing import Iterator, Optional
import torch
import math

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
