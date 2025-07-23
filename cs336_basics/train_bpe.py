import os
from typing import BinaryIO
import multiprocessing as mp
from collections import defaultdict, Counter
import regex as re
from tqdm import tqdm
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str], num_processes: int = 8) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a Byte Pair Encoding (BPE) model on the input file.
    
    Args:
        input_path (str): Path to the input text file.
        vocab_size (int): Desired size of the vocabulary.
        special_tokens (list[str]): List of special tokens to include in the vocabulary.
    """
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # Step 1: Initialize Vocabulary
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    for token_bytes in special_token_bytes:
        if token_bytes not in vocab.values():
            vocab[next_id] = token_bytes
            next_id += 1
    
    revvocab = {v: k for k, v in vocab.items()}

    # Step 2: Pre-tokenization
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str) -> tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)
    
    def bytes_tuple_to_ints_tuple(bytes_tuple: tuple[bytes]) -> tuple[int]:
        l = [revvocab[b] for b in bytes_tuple]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in tqdm(chunks, desc="Pre-tokenizing", unit="chunk"):
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[bytes_tuple_to_ints_tuple(to_bytes_tuple(word))] += 1   # key of pre_tokens_cnt is a tuple of ints

    # Step 3: Compute BPE Merges
    merges = []
    pair_counts = defaultdict(int)

    # Count all adjacent byte pairs
    for token, cnt in pre_tokens_cnt.items():
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1]) # pair of ints
            pair_counts[pair] += cnt

    i = 0  # Start from the next available ID
    for i in tqdm(range(vocab_size - len(vocab))):
        i += 1
        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [(vocab[k[0]], vocab[k[1]]) for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)
        best_pair = (revvocab[best_pair[0]], revvocab[best_pair[1]])  # Convert back to ints
        a, b = best_pair

        # Create new token
        new_token = next_id
        vocab[new_token] = vocab[a] + vocab[b]
        revvocab[vocab[new_token]] = new_token
        next_id += 1
        change = []
        # Apply the merge to all pre-tokenized sequences
        # 收集变更
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        pair_counts[best_pair] -= cnt
                        if pair_counts[best_pair] <= 0:
                            del pair_counts[best_pair]
                        if i > 0:
                            pair_counts[(token[i - 1], token[i])] -= cnt
                            if pair_counts[(token[i - 1], token[i])] <= 0:
                                del pair_counts[(token[i - 1], token[i])]
                            pair_counts[(token[i - 1], new_token)] = pair_counts.get((token[i - 1], new_token), 0) + cnt
                        if i + 2 < len(token):
                            pair_counts[(token[i + 1], token[i + 2])] -= cnt
                            if pair_counts[(token[i + 1], token[i + 2])] <= 0:
                                del pair_counts[(token[i + 1], token[i + 2])]
                            pair_counts[(new_token, token[i + 2])] = pair_counts.get((new_token, token[i + 2]), 0) + cnt
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                change.append((token, new_pre_token, cnt))

        for old_token, new_pre_token, cnt in change:
            del pre_tokens_cnt[old_token]
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt

        # 应用变       

        # Record the merge
        merges.append((vocab[a], vocab[b]))

    return vocab, merges