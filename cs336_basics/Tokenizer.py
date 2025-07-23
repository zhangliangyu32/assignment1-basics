import pickle
import regex as re
from tqdm import tqdm
from typing import Iterable, Iterator
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer(object):
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.revvocab = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens
        return
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as f:
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        special_tokens = special_tokens
        return cls(vocab, merges, special_tokens)
    
    def encode(self, text: str) -> list[int]:
        def to_bytes_list(word: str) -> list[bytes]:
            l = list(tuple(word.encode("utf-8")))
            l = [bytes([x]) for x in l]
            return l
        
        # Handle case when special_tokens is None
        if self.special_tokens is None or len(self.special_tokens) == 0:
            # No special tokens, just process normally
            pre_tokens = []
            for m in re.finditer(PAT, text):
                word = m.group(0)
                pre_tokens.append(to_bytes_list(word))
        else:
            # Use capturing groups to preserve special tokens during splitting
             # Sort special tokens by length (longest first) to avoid partial matches
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "(" + "|".join(map(re.escape, sorted_special_tokens)) + ")"
            chunks = re.split(pattern, text)
            pre_tokens = []
            for chunk in chunks:
                if chunk in self.special_tokens:
                    # Handle special tokens directly
                    pre_tokens.append([chunk.encode("utf-8")])
                elif chunk:  # Skip empty chunks
                    for m in re.finditer(PAT, chunk):
                        word = m.group(0)
                        pre_tokens.append(to_bytes_list(word))
        
        tokens_ids = []
        for pre_token in pre_tokens:
            # Apply merges
            for merge in self.merges:
                i = 0
                while i < len(pre_token) - 1:
                    if pre_token[i] == merge[0] and pre_token[i + 1] == merge[1]:
                        pre_token[i] = merge[0] + merge[1]
                        del pre_token[i + 1]
                    else:
                        i += 1
            # Convert to token IDs
            token_ids = []
            for token in pre_token:
                if token in self.revvocab:
                    token_ids.append(self.revvocab[token])
                # If token not in vocab, we might need to handle unknown tokens
            tokens_ids.extend(token_ids)
        return tokens_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    
    def decode(self, ids: list[int]) -> str:
        """
        Decode a list of token IDs back to a string.
        """
        tokens = [self.vocab[i] for i in ids]
        text = b''.join(tokens).decode('utf-8', errors='replace')
        return text
