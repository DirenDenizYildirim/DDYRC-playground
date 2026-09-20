"""Measurements taken on the raw bytes of memory.

Nothing here knows about organisms, lineages or fitness.  Every quantity is a
statistic of the byte array plus counters from the interpreter.
"""
import zlib

import numpy as np

LOG2 = np.log(2.0)


def shannon_entropy_bits(mem):
    """Shannon entropy of the byte distribution, in bits per byte (0..8)."""
    counts = np.bincount(mem.reshape(-1), minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / LOG2)


def compressed_bits_per_byte(mem, level=9):
    """zlib-compressed size in bits per byte.

    An upper bound on the per-byte code length of a real (if weak) universal
    model, so it picks up repeated substrings that the order-0 entropy cannot.
    """
    raw = mem.reshape(-1).tobytes()
    if not raw:
        return 0.0
    return float(len(zlib.compress(raw, level)) * 8) / len(raw)


def high_order_entropy(mem, level=9):
    """(order-0 entropy) - (compressed bits/byte).

    Near zero for i.i.d. bytes; rises when the same substrings recur, which is
    what a replicator takeover looks like from the outside.
    """
    h = shannon_entropy_bits(mem)
    c = compressed_bits_per_byte(mem, level)
    return h, c, h - c


def _ext_starts(mem, w, wrap, tape_len=None):
    """Extended buffer plus the start offsets of every admissible w-window."""
    mem = mem.reshape(-1)
    n = mem.size
    if wrap:
        ext = np.concatenate([mem, mem[: w - 1]])
        starts = np.arange(n, dtype=np.int64)
    else:
        ext = mem
        starts = np.arange(n - w + 1, dtype=np.int64)
        if tape_len is not None:
            starts = starts[(starts % tape_len) <= tape_len - w]
    return ext, starts


def window_matrix(mem, w, wrap, tape_len=None):
    """(n_windows, w) uint8 matrix of every admissible window."""
    ext, starts = _ext_starts(mem, w, wrap, tape_len)
    out = np.empty((starts.size, w), dtype=np.uint8)
    for j in range(w):
        out[:, j] = ext[starts + j]
    return out


def window_keys_u64(mem, w, wrap, tape_len=None):
    """Windows of w <= 8 bytes packed little-endian into uint64 keys."""
    assert w <= 8
    ext, starts = _ext_starts(mem, w, wrap, tape_len)
    keys = np.zeros(starts.size, dtype=np.uint64)
    for j in range(w):
        keys |= ext[starts + j].astype(np.uint64) << np.uint64(8 * j)
    return keys


def count_windows_u64(mem, w, wrap, tape_len=None):
    """Sorted unique window keys and their multiplicities."""
    return np.unique(window_keys_u64(mem, w, wrap, tape_len), return_counts=True)


def top_windows(mem, w, wrap, tape_len=None, top=10):
    """The ``top`` most frequent w-byte windows as (bytes, count) pairs."""
    mat = np.ascontiguousarray(window_matrix(mem, w, wrap, tape_len))
    recs = mat.view(np.dtype((np.void, w))).reshape(-1)
    uniq, counts = np.unique(recs, return_counts=True)
    if uniq.size == 0:
        return []
    order = np.argsort(counts, kind="stable")[::-1][:top]
    return [(uniq[i].tobytes(), int(counts[i])) for i in order]


def key_to_bytes(key, w):
    k = int(key)
    return bytes((k >> (8 * j)) & 255 for j in range(w))


_PRINTABLE = bytes(range(33, 127))


def printable(raw):
    """Readable rendering: printable ASCII verbatim, everything else as '.'.

    The hex form is logged alongside, so the ambiguity between a no-op '.'
    byte and a literal '.' instruction never has to be resolved by eye.
    """
    return "".join(chr(b) if b in _PRINTABLE else ("_" if b == 32 else ".")
                   for b in raw)


class PersistenceTracker:
    """A(t): cumulative count of distinct windows that have ever persisted.

    A window is *persistent* once its count has been >= ``c_min`` for ``tau``
    consecutive snapshots.  A(t) only ever grows; ``n_persistent`` is how many
    windows are persistent right now.

    Note the streak is over *snapshots*, so what "tau" means in epochs depends
    on the snapshot interval.
    """

    def __init__(self, c_min=8, tau=5):
        self.c_min = int(c_min)
        self.tau = int(tau)
        self._keys = np.zeros(0, dtype=np.uint64)     # sorted, streak > 0
        self._streaks = np.zeros(0, dtype=np.int64)
        self.ever = np.zeros(0, dtype=np.uint64)      # sorted

    def update(self, keys, counts):
        """``keys`` must be sorted ascending (np.unique output)."""
        q = keys[counts >= self.c_min]
        if q.size == 0:
            self._keys = np.zeros(0, dtype=np.uint64)
            self._streaks = np.zeros(0, dtype=np.int64)
            return len(self.ever), 0
        streaks = np.ones(q.size, dtype=np.int64)
        if self._keys.size:
            idx = np.searchsorted(self._keys, q)
            idx_c = np.clip(idx, 0, self._keys.size - 1)
            hit = self._keys[idx_c] == q
            streaks[hit] = self._streaks[idx_c[hit]] + 1
        self._keys, self._streaks = q, streaks
        now_persistent = q[streaks >= self.tau]
        if now_persistent.size:
            self.ever = np.union1d(self.ever, now_persistent)
        return len(self.ever), int(now_persistent.size)

    @property
    def a_t(self):
        return int(self.ever.size)
