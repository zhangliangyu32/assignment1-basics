
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
        "max_learning_rate": 1e-3,
        "min_learning_rate": 1e-6,
        "warmup_end": 1000,
        "cosine_end": 40000,
        "device": "cuda:0",
        "train_data_path": "/root/autodl-tmp/data/tiny_stories_train.npy",
        "val_data_path": "/root/autodl-tmp/data/tiny_stories_valid.npy",
        "ablation_mode": "no_norm", # None, post_norm, no_RoPE, SiLU
    }
from cs336_basics.Modules import TransformerLM, TransformerLMwithAblation
import numpy as np
from cs336_basics.train import train
x_train = np.memmap(config["train_data_path"], dtype=np. uint16, mode='r').astype(np.int64)
x_val = np.memmap(config["val_data_path"], dtype=np.uint16, mode='r').astype(np.int64)

# config["learning_rate"] = 1e-4
# model = TransformerLM(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, device=config["device"])
# train(x_train, x_val, model, config)

# config["learning_rate"] = 2e-3
# config["batch_size"] = 64
# config["max_iterations"] = 20000
# model = TransformerLM(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, device=config["device"])
# train(x_train, x_val, model, config)

# config["learning_rate"] = 5e-4
# config["batch_size"] = 16
# config["max_iterations"] = 80000
# model = TransformerLM(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, device=config["device"])
# train(x_train, x_val, model, config)


config["note"] = "no_norm ablation"
model = TransformerLMwithAblation(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, ablation_mode=config["ablation_mode"], device=config["device"])
train(x_train, x_val, model, config)

config["learning_rate"] = 1e-4
model = TransformerLMwithAblation(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, ablation_mode=config["ablation_mode"], device=config["device"])
train(x_train, x_val, model, config)

config["learning_rate"] = 1e-3
config["ablation_mode"] = "post_norm"
model = TransformerLMwithAblation(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, ablation_mode=config["ablation_mode"], device=config["device"])
train(x_train, x_val, model, config)

config["ablation_mode"] = "no_RoPE"
model = TransformerLMwithAblation(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, ablation_mode=config["ablation_mode"], device=config["device"])
train(x_train, x_val, model, config)

config["ablation_mode"] = "SiLU"
model = TransformerLMwithAblation(vocab_size=10000, context_length=config["context_len"], num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, ablation_mode=config["ablation_mode"], device=config["device"])
train(x_train, x_val, model, config)
