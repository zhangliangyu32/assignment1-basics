import numpy as np
from torch import nn
from cs336_basics.utils import cross_entropy_loss
from cs336_basics.train_utils import data_loader, save_checkpoint, load_checkpoint, compute_validation_loss
from cs336_basics.Optimizer import AdamW, ConstantLRScheduler, CosineLRScheduler
import swanlab
from tqdm import tqdm
import os
def train(x_train: np.ndarray, x_val: np.ndarray, model: nn.Module, config: dict):
    batch_size = config.get("batch_size", 32)
    context_len = config.get("context_len", 256)
    device = config.get("device", "cpu")
    max_iterations = config.get("max_iterations", 40000)
    weight_decay = config.get("weight_decay", 0.01)
    save_until = config.get("save_until", 2000)
    save_path = config.get("save_path", "/root/assignment1-basics/checkpoints/")
    val_until = config.get("val_until", 1000)
    training_note = config.get("note", "No note provided")
    training_project = config.get("project", "cs336_basics")
    if config.get("lr_scheduler_cosine", False) is False:
        lr = config.get("learning_rate", 1e-3)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        lr_scheduler = ConstantLRScheduler(optimizer, iter=0, lr=lr)
        run = swanlab.init(project=training_project, 
        config={"lr": lr, "batch_size": batch_size, "weight_decay": weight_decay, "note": training_note})
    else:
        max_lr = config.get("max_learning_rate", 1e-3)
        min_lr = config.get("min_learning_rate", 1e-6)
        warmup_end = config.get("warmup_end", 4000)
        cosine_end = config.get("cosine_end", max_iterations)
        optimizer = AdamW(model.parameters(), lr=max_lr, weight_decay=weight_decay)
        lr_scheduler = CosineLRScheduler(optimizer, iter=0, max_lr=max_lr, min_lr=min_lr, warmup_end=warmup_end, cosine_end=cosine_end)
        run = swanlab.init(project=training_project, 
                     config={"lr": "cosine scheduler", "batch_size": batch_size, "weight_decay": weight_decay, "note": training_note})

    
    model.train()
    for iteration in tqdm(range(max_iterations)):
        x_batch, target_batch = data_loader(x_train, batch_size=batch_size, context_len=context_len, device=device)
        logits = model(x_batch)
        loss = cross_entropy_loss(logits, target_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if (iteration + 1) % 100 == 0:
            run.log({"train_loss": loss.item(), "learning_rate": optimizer.param_groups[0]['lr']})
            print(f"Iteration {iteration + 1}, Training Loss: {loss.item():.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        if (iteration + 1) % save_until == 0:
            save_checkpoint(model, optimizer, iteration, os.path.join(save_path, f"checkpoint_{iteration + 1}.pt"))

        if (iteration + 1) % val_until == 0:
            val_loss = compute_validation_loss(model, x_val, context_len=context_len, device=device)
            run.log({"val_loss": val_loss})
            print(f"Iteration {iteration + 1}, Validation Loss: {val_loss:.4f}")
    run.finish()
    save_checkpoint(model, optimizer, iteration, os.path.join(save_path, f"checkpoint_final.pt"))
    return

if __name__ == "__main__":
    config = {
        "batch_size": 32,
        "context_len": 256,
        "max_iterations": 40000,
        "weight_decay": 0.00,
        "save_until": 100000,
        "save_path": "/root/assignment1-basics/checkpoints/",
        "val_until": 2000,
        "learning_rate": 3e-3,
        "lr_scheduler_cosine": False,
        "device": "cuda:0",
        "train_data_path": "/root/autodl-tmp/data/tiny_stories_train.npy",
        "val_data_path": "/root/autodl-tmp/data/tiny_stories_valid.npy",
    }
    from cs336_basics.Modules import TransformerLM
    model = TransformerLM(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, device=config["device"])
    x_train = np.memmap(config["train_data_path"], dtype=np. uint16, mode='r').astype(np.int64)
    x_val = np.memmap(config["val_data_path"], dtype=np.uint16, mode='r').astype(np.int64)
    train(x_train, x_val, model, config)
