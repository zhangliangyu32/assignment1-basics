import os
from typing import BinaryIO
import multiprocessing as mp
from collections import defaultdict, Counter
import regex as re
from tqdm import tqdm
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


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

    # Step 2: Pre-tokenization
    pre_tokens_cnt = defaultdict(int)

    def to_bytes_tuple(word: str) -> tuple[bytes]:
        l = list(tuple(word.encode("utf-8")))
        l = [bytes([x]) for x in l]
        return tuple(l)

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    chunks = re.split("|".join(map(re.escape, special_tokens)), text)
    
    for chunk in tqdm(chunks):
        for m in re.finditer(PAT, chunk):
            word = m.group(0)
            pre_tokens_cnt[to_bytes_tuple(word)] += 1   # key of pre_tokens_cnt e.g. (b'H', b'e', b'l', b'l', b'o')

    # Step 3: Compute BPE Merges
    merges = []

    for i in tqdm(range(vocab_size- len(vocab))):
    # while len(vocab) < vocab_size:
        pair_counts = defaultdict(int)

        # Count all adjacent byte pairs
        for token, cnt in pre_tokens_cnt.items():
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                pair_counts[pair] += cnt

        if not pair_counts:
            break  # No more pairs to merge

        # Find the most frequent pair(s)
        max_count = max(pair_counts.values())
        candidates = [k for k, v in pair_counts.items() if v == max_count]
        best_pair = max(candidates)

        a, b = best_pair

        # Create new token
        new_token = a + b
        vocab[next_id] = new_token
        next_id += 1

        # Apply the merge to all pre-tokenized sequences
        # 收集变更
        changes = []
        for token, cnt in pre_tokens_cnt.items():
            # Find all occurrences of the `best_pair` in `token`
            indices = [i for i in range(len(token) - 1) if token[i:i + 2] == best_pair]
            if indices:
                # Replace each occurrence with `new_token`
                new_pre_token = []
                i = 0
                while i < len(token):
                    if i in indices:
                        new_pre_token.append(new_token)
                        i += 2
                    else:
                        new_pre_token.append(token[i])
                        i += 1
                new_pre_token = tuple(new_pre_token)
                changes.append((token, new_pre_token, cnt))

        # 应用变更
        for old_token, new_pre_token, cnt in changes:
            pre_tokens_cnt[new_pre_token] = pre_tokens_cnt.get(new_pre_token, 0) + cnt
            del pre_tokens_cnt[old_token]

        # Record the merge
        merges.append((a, b))

    return vocab, merges