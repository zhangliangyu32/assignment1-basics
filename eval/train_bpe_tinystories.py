import cProfile
import pickle
from cs336_basics.train_bpe import train_bpe
cProfile.run('vocabulary, merges=train_bpe("data/TinyStoriesV2-GPT4-train.txt", 10000, ["<|endoftext|>"])')
pickle.dump(vocabulary, open("./vocabulary.pkl", "wb"))
pickle.dump(merges, open("./merges.pkl", "wb"))