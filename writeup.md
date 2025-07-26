Problem (unicode1):

1. '\x00'
2. It is the invisible null character when printed.
3. It reads '\x00' when appears as in a string and is invisible when printed.

Problem (unicode2):

1. The vocabulary is smaller. And thus aviod the sparsity issue and achieve better space efficiency for common text.
2. The UTF-8 encoding may use multiple bytes to represent a character. A failure case can be "你好吗？".
3. "\xe4\xbd". "\xe4" signals the beginning of a 3-byte character.

Problem (train_bpe_tinystories):

1. It uses 43GB memories and takes 50 mins to finish. The longest token is the word "accomplishment", which makes perfect sense.
2. The acquire method for thread.lock takes most of the running time. This is because the implementation is serial.

Problem (train_bpe_expts_owt):

1. It uses 231GB memories and takes about 50 hours to finish. The longest token is the word "---", which makes perfect sense.

2.

Problem (tokenizer_experiments):

1. The compression ratios of the tiny stories tokenizer and the owt tokenizer are 4.094558901215002 and --- , respectively.

2. 

3. The throughput of the two tokenizers are 8206.930357155328 and ----. It will take 114095 and ---- seconds if we use the two tokenizers to tokenize the Pile dataset, respectively.

4. .

Problem (transformer_accounting): 

1. The Token Embedding layer has vocab_size * d_model learnable parameters.

    For a transformer block, there are 

    - two RMSNorm layer, each has d_model learnable parameters
    - one FFN layer with 3 * d_model * d_ff learnable parameters
    - one MultiHeadSelfAttention layer with d_model * d_k * num_heads + d_model * d_k * num_heads + d_model * d_v * num_heads + d_model * d_v * num_heads learnable parameters. Note that d_v * num_heads = d_k * num_heads = d_model.
    
    The final output layer with d_model * vocab_size learnable parameters.

    In total, there are vocab_size * d_model + num_layers * (2 * d_model + 3 * d_model * d_ff + 4 * d_model^2) + d_model * vocab_size = 2127056000=2.1B parameters.
    It takes about 8.5GB memory to load this model.

2. For each transformer block, the FF layer need to perform 3 matmuls with 4 * context_len  * d_model * d_ff + 2 * context_len * d_model * d_ff FLOPs. (6 ldf)

    In the MHA layer, we need to 

    - Compute Q, K, V, each need a matrix multiplication with context_len * (2 * d_model * (num_heads * d_k)) FLOPs (assuming d_k=d_v). (6 ld^2)
    - Compute Q^T*K, needing a matmul with num_heads * (2 * context_len * d_k * context_len) FLOPs. (2 l^2d)
    - Compute (Q^T*K) * V, needing a matmul with num_heads * (2 * context_len * context_len * d_v) FLOPs. (2 l^2d)
    - Compute the output, needing a matmul with  (2 * context_len * (num_heads * d_v) * d_model). (2 ld^2)

    For the final output layer, we need a matmul with 2 * vocab_len * d_model * context_len FLOPs. (100 l^2d)

    In total, there are 4.51 * 1e12.

3. Each transformer block needs $6 ldf+8ld^2+4l^2d$ FLOPs, where MHA takes $8ld^2+4l^2d$ FLOPs and the FFN takes $24ld^2$ (suppose $f = 4l$). The final output layer needs $2 vdl$ FLOPs.
The transformer blocks require the most FLOPs, and the FLOPs of FFN is about 2 times of the FLOPs of the MHA.

4. Clear from the computations in 3.

5. Clear from the computations in 3.

Problem (learning_rate_tuning):

* lr=10, slow decay;
* lr=100, fast decay;
* lr=1000, diverges.



