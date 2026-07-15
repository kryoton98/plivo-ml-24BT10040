"""
Byte-level BPE tokenizer (pure standard library).

Design:
  * Base vocabulary = the 256 raw bytes  ->  ANY UTF-8 text encodes, always.
  * 768 learned merges                   ->  vocab_size == 1024.
  * decode(encode(text)) == text exactly (lossless: every token maps back to
    a fixed byte string, and the concatenation is the original UTF-8 bytes).

Exposes:
    load()                      -> Tokenizer  (no args; extra args optional)
    Tokenizer.encode(str)       -> list[int]
    Tokenizer.decode(list[int]) -> str
    Tokenizer.vocab_size        -> int
    train(text_file_path)       -> Tokenizer  (also writes merges.json)

All state lives in merges.json next to this file, resolved via __file__.
"""

import json
import os
import re
import sys
from collections import Counter

VOCAB_SIZE = 1024
NUM_MERGES = VOCAB_SIZE - 256  # 768

MERGES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merges.json")

# ---------------------------------------------------------------------------
# Pre-tokenization
# ---------------------------------------------------------------------------
# Splits text into chunks so merges never cross word/space boundaries (this is
# what keeps training fast and stops tokens like "the_dog" from forming).
# stdlib `re` has no \p{...}, but \w is Unicode-aware by default in Py3, so
# Devanagari letters land in the \w class alongside ASCII letters.
# The three branches are exhaustive over every character (\s, \w, or neither),
# so "".join(findall(text)) == text for all inputs. encode() verifies anyway.
_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)|\ ?\w+|\ ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)


def _chunks(text):
    parts = _PAT.findall(text)
    if "".join(parts) != text:  # paranoia: never lose a byte
        return [text]
    return parts


# ---------------------------------------------------------------------------
# Core BPE helpers
# ---------------------------------------------------------------------------
def _pairs(word):
    return zip(word, word[1:])


def _merge_word(word, pair, new_id):
    """Replace every non-overlapping occurrence of `pair` in `word`."""
    out = []
    i = 0
    n = len(word)
    a, b = pair
    while i < n:
        if i < n - 1 and word[i] == a and word[i + 1] == b:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return out


class Tokenizer:
    def __init__(self, merges):
        # merges: list of (a, b) in the exact order learned.
        self.merges = [tuple(m) for m in merges]
        # rank[(a, b)] = position in the learned order (lower == applied first)
        self.ranks = {pair: i for i, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

        # id -> bytes, for decoding
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]

        self._cache = {}  # chunk(str) -> list[int]

    # -- encode ------------------------------------------------------------
    def _encode_chunk(self, chunk):
        hit = self._cache.get(chunk)
        if hit is not None:
            return hit

        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # find the pair present in `ids` with the lowest merge rank, i.e.
            # apply merges in exactly the order they were learned
            best = None
            best_rank = None
            for p in _pairs(ids):
                r = self.ranks.get(p)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = p, r
            if best is None:
                break
            ids = _merge_word(ids, best, 256 + best_rank)

        if len(self._cache) < 200_000:
            self._cache[chunk] = ids
        return ids

    def encode(self, text):
        out = []
        for chunk in _chunks(text):
            out.extend(self._encode_chunk(chunk))
        return out

    # -- decode ------------------------------------------------------------
    def decode(self, ids):
        buf = []
        for i in ids:
            piece = self.vocab.get(int(i))
            if piece is None:  # unknown id -> skip rather than crash
                continue
            buf.append(piece)
        return b"".join(buf).decode("utf-8", errors="replace")

    # -- persistence -------------------------------------------------------
    def save(self, path=None):
        path = path or MERGES_FILE
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"type": "bpe", "vocab_size": self.vocab_size,
                 "merges": [list(m) for m in self.merges]},
                f,
            )
        return path


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(text_file_path, num_merges=NUM_MERGES, out_path=None, verbose=True):
    """Learn `num_merges` BPE merges from a text file and save merges.json.

    Efficiency: text is collapsed to a chunk->frequency table, then each merge
    only touches the chunks that actually contain the winning pair (tracked in
    an inverted index), so it's ~O(corpus) once + O(affected) per merge rather
    than a full re-scan every iteration. A 7MB corpus trains in well under a
    minute.
    """
    with open(text_file_path, "r", encoding="utf-8", errors="strict") as f:
        text = f.read()

    freq = Counter(_chunks(text))
    words = [list(w.encode("utf-8")) for w in freq]
    counts = list(freq.values())

    # global pair stats + inverted index pair -> {word indices}
    stats = Counter()
    where = {}
    for wi, (w, c) in enumerate(zip(words, counts)):
        for p in _pairs(w):
            stats[p] += c
            where.setdefault(p, set()).add(wi)

    merges = []
    for step in range(num_merges):
        if not stats:
            break
        pair = max(stats, key=lambda p: (stats[p], -p[0], -p[1]))
        if stats[pair] < 2:
            break
        new_id = 256 + len(merges)
        merges.append(pair)

        for wi in list(where.get(pair, ())):
            w = words[wi]
            c = counts[wi]
            for p in _pairs(w):  # retract old contributions
                stats[p] -= c
                if stats[p] <= 0:
                    del stats[p]
                s = where.get(p)
                if s is not None:
                    s.discard(wi)

            nw = _merge_word(w, pair, new_id)
            words[wi] = nw

            for p in _pairs(nw):  # add new ones
                stats[p] += c
                where.setdefault(p, set()).add(wi)

        stats.pop(pair, None)
        where.pop(pair, None)

        if verbose and (step + 1) % 100 == 0:
            print(f"  merge {step + 1}/{num_merges}", file=sys.stderr)

    tok = Tokenizer(merges)
    tok.save(out_path)
    if verbose:
        n_bytes = len(text.encode("utf-8"))
        n_tok = len(tok.encode(text))
        print(
            f"trained {len(merges)} merges | vocab_size={tok.vocab_size} | "
            f"compression {n_bytes / max(n_tok, 1):.2f}x bytes/token",
            file=sys.stderr,
        )
    return tok


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load(path=None):
    """Return the tokenizer used by train.py / evaluate.py. Called with no args."""
    path = path or MERGES_FILE
    if os.path.isdir(path):
        path = os.path.join(path, "merges.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run `python tokenizer.py <corpus.txt>` to train it."
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Tokenizer(data["merges"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python tokenizer.py <corpus.txt>", file=sys.stderr)
        raise SystemExit(1)
    train(sys.argv[1])