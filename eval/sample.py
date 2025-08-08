from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.Modules import TransformerLM
from cs336_basics.Optimizer import AdamW
from cs336_basics.train_utils import load_checkpoint

import torch
import numpy as np
import os

model = TransformerLM(vocab_size=32000, context_length=256, num_layers=4, d_model=512, num_heads=16, d_ff=2048, rope_theta=10000, device="cuda:0")
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.00)

load_checkpoint(os.path.join("/root/assignment1-basics/checkpoints/", "checkpoint_final.pt"), model, optimizer)
model.eval()
tokenizer_owt = Tokenizer.from_files(
    vocab_filepath="owt_vocabulary.pkl",
    merges_filepath="owt_merges.pkl",
    special_tokens=["<|endoftext|>"]
)
for i in range(4):
    prompt = "Once"
    prefix=torch.tensor(tokenizer_owt.encode(prompt), device="cuda:0")
    text = model.decode(prefix, max_length=256, eof_id=tokenizer_owt.eot_id, temperature=0.8, top_p=0.8)
    print(f"Sample {i+1}: {tokenizer_owt.decode(text.squeeze().cpu().numpy())}")
