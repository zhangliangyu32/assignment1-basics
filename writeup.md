Problem (unicode1):

1. '\x00'
2. It is the invisible null character when printed.
3. It reads '\x00' when appearing as in a string and is invisible when printed.

Problem (unicode2):

1. The vocabulary is smaller. And thus avoid the sparsity issue and achieve better space efficiency for common text.
2. The UTF-8 encoding may use multiple bytes to represent a character. A failure case can be "你好吗？".
3. "\xe4\xbd". "\xe4" signals the beginning of a 3-byte character.

Problem (train_bpe_tinystories):

1. It uses 6GB memory and takes only 28 seconds to finish training (damek's implementations is so fast!!!). The longest token is the word "accomplishment", which makes perfect sense.
2. The functions for reading files takes most of the running time. This is because the main performance bottleneck is to load the dataset.

Problem (train_bpe_expts_owt):

1. It uses 55GB memory and takes about 1.5 hours to finish. The longest token is a string consisting of repeading number patterns, which is a little bit confusing.

2. The owt tokenizer has a richer vocabulary. 

Problem (tokenizer_experiments):

1. The compression ratios of the tiny stories tokenizer and the owt tokenizer are 4.094558901215002 and 4.368112784973083 , respectively.

2. The compression ratio is lower (3.1813436083310203).

3. The throughput of the two tokenizers are 2858193.9135480467 and 1965804.69142808. It will take 310150 and 450819 seconds if we use the two tokenizers to tokenize the Pile dataset, respectively.

4. uint16 achieve optimal space complexity while 2^16=65536>32000.

Problem (transformer_accounting): 

1. The Token Embedding layer has vocab_size * d_model learnable parameters.

    For a transformer block, there are 

    - two RMSNorm layer, each has d_model learnable parameters
    - one FFN layer with 3 * d_model * d_ff learnable parameters
    - one MultiHeadSelfAttention layer with d_model * d_k * num_heads + d_model * d_k * num_heads + d_model * d_v * num_heads + d_model * d_v * num_heads learnable parameters. Note that d_v * num_heads = d_k * num_heads = d_model.
    
    The final output layer has d_model * vocab_size learnable parameters.

    In total, there are vocab_size * d_model + num_layers * (2 * d_model + 3 * d_model * d_ff + 4 * d_model^2) + d_model * vocab_size + d = 2127057600=2.1B parameters.
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

Problem (adamwAccounting):

1. We first define some notations: $d=$d_model, $h=$num_heads, $v=$vocab_size, $l=$context_len, $n=$num_layers, $B=$batch_size.
   For the parameters,
   * The token embedding layer has $dv$ parameters.
   * The final Norm layer and output embeddings layer have $d$ and $dv$ parameters, respectively.
   * In the transformer block, each of the two Norm layers has $d$ parameters, the MHA layer has $4d^2$ parameters, and the FFN has $12 d^2$ parameters.
   
   In total, there are $2dv+d+n(16d^2+2d)$ parameters.
   
   The memory consumption of saving the gradients is equal to that of saving the parameters.

   For the optimizer's state, we need to save two moving averages, which costs twice the memory cost of saving the parameters.

   For the activations,
   * The final Norm layer yields $BLd$ activations, and the output embedding layer yields $BLv$ activations.
   * In the transformer block, each of the two Norm layers has $Bld$ activations. For MHA, the Q, K, V projections yields $3Bld$ activations. $Q^\top K$ matrix multiply yields $Bhl^2$ activations, softmax yields $Bld$ activations, weighted sum of values yields $Bld$ activations, and the final output yields $Bld$ activations. For FFN, each of silu, $W_1$ matrix multiply and $W_2$ matrix multiply yields $4Bld$ activations.
   * To compute cross-entropy on logits we need to compute $Bl$ activations.
  
   In total, there are $BLd+Blv+Bl+n(20Bld+bl^2)$ activations.

   So the final memory use is $8dv+4d+4n(16d^2+2d)+BLd+Blv+Bl+n(20Bld+BHl^2)$ float 32 parameters.

2. The expression is $8.51\times 10^9+2.88\times 10^9 B$, or $31.7$GB+$10.7B$GB. The maximal batch_size is approximately 4.

3. Each forward step takes about $4.51\times 10^{12}$ FLOPs. So each adamW step takes about $13.5\times 10^{12}$ FLOPs (ignoring some minor operations.)

4. It will take approximately 7 years to finish the training

Proble (generate): 

Here is a sample from the LM trained on TinyStories. We use top_p=0.8 and temprerature=0.8.

Once upon a time, there was a little boy named Tim. Tim was a very good boy, but he was very spoiled. He always wanted more and more. One day, Tim's mom took him to the store. She wanted to buy a toy for him. Tim was very happy.
At the store, Tim saw many toys. He saw a big doll, a soft bear, and a funny car. But he did not have enough money. Tim felt sad and his mom said, "Don't be sad, Tim. You can buy the car." Tim's mom bought the car for him.
When they got home, Tim played with his new toy. He was very happy. But then, something unexpected happened. The car started to move on its own! It drove around the room, and Tim was very surprised. The car drove around the room, and Tim laughed and clapped his hands.




