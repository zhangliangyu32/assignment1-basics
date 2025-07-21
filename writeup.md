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

