from cs336_basics.Tokenizer import Tokenizer

tokenizer_owt = Tokenizer.from_files(
    vocab_filepath="owt_vocabulary.pkl",
    merges_filepath="owt_merges.pkl",
    special_tokens=["<|endoftext|>"]
)
tokenizer_owt.encode_to_file(load_path="/root/autodl-tmp/data/owt_train.txt", save_path="/root/autodl-tmp/data/owt_train.npy")

# tokenizer_tiny_stories = Tokenizer.from_files(
#     vocab_filepath="tiny_stories_vocabulary.pkl",
#     merges_filepath="tiny_stories_merges.pkl",
#     special_tokens=["<|endoftext|>"]
# )
# tokenizer_tiny_stories.encode_to_file(
#     load_path="/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt",
#     save_path="/root/autodl-tmp/data/tiny_stories_train.npy")
