import numpy as np
from numba import jit
from math import log

"""
Lempel-Ziv complexity (LZ complexity) implementation.
Code adapted from https://github.com/raphaelvallat/antropy
"""

@jit("uint32(uint32[:])", nopython=True)
def _lz_complexity(binary_string):
    complexity = 1
    prefix_len = 1
    len_substring = 1
    max_len_substring = 1
    pointer = 0

    while prefix_len + len_substring <= len(binary_string):
        if binary_string[pointer + len_substring - 1] == binary_string[prefix_len + len_substring - 1]:
            len_substring += 1
        else:
            max_len_substring = max(len_substring, max_len_substring)
            pointer += 1
            if pointer == prefix_len:
                complexity += 1
                prefix_len += max_len_substring
                pointer = 0
                max_len_substring = 1
            len_substring = 1

    if len_substring != 1:
        complexity += 1

    return complexity

def lziv_complexity(sequence, normalize=False):
    assert isinstance(sequence, (str, list, np.ndarray))
    assert isinstance(normalize, bool)

    if isinstance(sequence, (list, np.ndarray)):
        arr = np.asarray(sequence)
        # Check if already binary (only 0 and 1)
        unique_vals = np.unique(arr)
        if not np.all(np.isin(unique_vals, [0, 1])):
            # Binarise around the mean
            mean = np.mean(arr)
            arr = (arr > mean).astype("uint32")
        sequence = arr
    else:
        # For string, check if only '0' and '1'
        unique_chars = set(sequence)
        if not unique_chars.issubset({'0', '1'}):
            # Convert to array of ints, binarise around mean
            arr = np.array([ord(c) for c in sequence])
            mean = np.mean(arr)
            arr = (arr > mean).astype("uint32")
            sequence = arr

    if isinstance(sequence, (list, np.ndarray)):
        sequence = np.asarray(sequence)
        if sequence.dtype.kind in "bfi":
            s = sequence.astype("uint32")
        else:
            s = np.fromiter(map(ord, "".join(sequence.astype(str))), dtype="uint32")
    else:
        s = np.fromiter(map(ord, sequence), dtype="uint32")

    if normalize:
        n = len(s)
        base = sum(np.bincount(s) > 0)
        base = 2 if base < 2 else base
        return _lz_complexity(s) / (n / log(n, base))
    else:
        return _lz_complexity(s)

# === Minimal test case ===
if __name__ == "__main__":
    s = '1001111011000010'
    # s = np.random.randn(1000)  # Random sequence
    print("LZ complexity (raw):", lziv_complexity(s))
    print("LZ complexity (normalized):", lziv_complexity(s, normalize=True))
