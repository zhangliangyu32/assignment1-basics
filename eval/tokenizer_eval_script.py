from cs336_basics.Tokenizer import Tokenizer
import time



tokenizer_owt = Tokenizer.from_files(vocab_filepath="tiny_stories_vocabulary.pkl", merges_filepath="tiny_stories_merges.pkl", special_tokens=["<|endoftext|>"])

max_len = 1
max_len_token = None
for id, token in tokenizer_owt.vocab.items():
    if len(token) > max_len:
        max_len = len(token)
        max_len_token = token
print(f"Max length token: {max_len_token}, Length: {max_len}")

len_threshold = 20
long_subword = []
for id, token in tokenizer_owt.vocab.items():
    if len(token) > len_threshold:
        long_subword.append(token)
print("Long subwords: ", long_subword)


with open("sampled_tiny_stories.txt", "r") as f:
    text = f.read()

# Print the number of bytes in the file
text_bytes = text.encode('utf-8')
print(f"Number of bytes in sampled_tiny_stories.txt: {len(text_bytes)}")
print(f"Number of characters: {len(text)}")

time_start = time.time()
owt_ids = tokenizer_owt.encode(text)
time_ed = time.time()
print("time taken to tokenize sampled owt", time_ed - time_start)
print("length of ids", len(owt_ids))
print("sampled ids", owt_ids[:50])

print("compression ratio", len(text_bytes) / len(owt_ids))
print("throughput", len(text_bytes) / (time_ed - time_start))

