from cs336_basics.Tokenizer import Tokenizer
from cs336_basics.Modules import TransformerLM
from cs336_basics.Optimizer import AdamW
from cs336_basics.train_utils import load_checkpoint

import torch
import numpy as np
import os

model = TransformerLM(vocab_size=10000, context_length=256, num_layers=4, d_model=512, num_heads=16, d_ff=1344, rope_theta=10000, device="cuda:0")
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.00)

load_checkpoint(os.path.join("/root/assignment1-basics/checkpoints/", "checkpoint_final.pt"), model, optimizer)
model.eval()
tokenizer_tiny_stories = Tokenizer.from_files(
    vocab_filepath="tiny_stories_vocabulary.pkl",
    merges_filepath="tiny_stories_merges.pkl",
    special_tokens=["<|endoftext|>"]
)
for i in range(4):
    prompt = "Once"
    prefix=torch.tensor(tokenizer_tiny_stories.encode(prompt), device="cuda:0")
    text = model.decode(prefix, max_length=256, eof_id=tokenizer_tiny_stories.eot_id, temperature=0.8, top_p=0.8)
    print(f"Sample {i+1}: {tokenizer_tiny_stories.decode(text.squeeze().cpu().numpy())}")
