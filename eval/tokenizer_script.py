from cs336_basics.Tokenizer import Tokenizer
import time
tokenizer_tiny_stories = Tokenizer.from_files(vocab_filepath="tiny_stories_vocabulary.pkl", merges_filepath="tiny_stories_merges.pkl", special_tokens=["<|endoftext|>"])

with open("sampled_tiny_stories.txt", "r") as f:
    text = f.read()

# Print the number of bytes in the file
text_bytes = text.encode('utf-8')
print(f"Number of bytes in sampled_tiny_stories.txt: {len(text_bytes)}")
print(f"Number of characters: {len(text)}")

time_start = time.time()
tiny_stories_ids = tokenizer_tiny_stories.encode(text)
time_ed = time.time()
print("time taken to tokenize sampled tiny stories", time_ed - time_start)
print("length of ids", len(tiny_stories_ids))
print("sampled ids", tiny_stories_ids[:50])

print("compression ratio", len(text_bytes) / len(tiny_stories_ids))
print("throughput", len(text_bytes) / (time_ed - time_start))

