"""Measurements taken on the raw bytes of memory.

Nothing here knows about organisms, lineages or fitness.  Every quantity is a
statistic of the byte array plus counters from the interpreter.
"""
import zlib

import brotli
import numpy as np

LOG2 = np.log(2.0)

# A 2**24-byte window covers every soup here; zlib's 32 KiB window does not,
# so zlib cannot see a repeat spanning more than half of a 64 KiB ring.
# Quality 6 is the default: quality 2 (what cubff reports) is weak enough that
# it compresses a one-byte crystal *worse* than its order-0 entropy, making
# high-order entropy negative; quality 11 costs 7.6 s on a 2 MiB soup.  Both
# are still logged -- q2 for comparability with cubff's own numbers.
BROTLI_LGWIN = 24
COMPRESSORS = {"brotli6": 6, "brotli2": 2, "brotli11": 11}


def shannon_entropy_bits(mem):
    """Shannon entropy of the byte distribution, in bits per byte (0..8)."""
    counts = np.bincount(mem.reshape(-1), minlength=256).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / LOG2)


def compressed_bits_per_byte(mem, method="brotli6", level=9):
    """Compressed size in bits per byte.

    An upper bound on the per-byte code length of a real (if weak) universal
    model, so it picks up repeated substrings that the order-0 entropy cannot.

    ``brotli6`` (the default), ``brotli2`` (cubff's setting), ``brotli11``,
    ``zlib`` and ``zstd`` are accepted.  ``zlib`` is kept only so the earlier
    runs stay comparable.
    """
    raw = mem.reshape(-1).tobytes()
    if not raw:
        return 0.0
    if method in COMPRESSORS:
        out = brotli.compress(raw, quality=COMPRESSORS[method],
                              lgwin=BROTLI_LGWIN)
    elif method == "zlib":
        out = zlib.compress(raw, level)
    elif method == "zstd":
        import zstandard
        out = zstandard.ZstdCompressor(
            level=19, write_content_size=False).compress(raw)
    else:
        raise ValueError("unknown compressor: %s" % method)
    return float(len(out) * 8) / len(raw)


def high_order_entropy(mem, method="brotli6", level=9):
    """(order-0 entropy H) - (compressed bits/byte).

    Near zero for i.i.d. bytes; rises when the same substrings recur.  It is
    near zero for a one-byte monoculture too, because H has fallen to meet the
    compressed size -- which is why H is always reported beside it.
    """
    h = shannon_entropy_bits(mem)
    c = compressed_bits_per_byte(mem, method, level)
    return h, c, h - c


def byte_frequencies(mem):
    counts = np.bincount(mem.reshape(-1), minlength=256).astype(np.float64)
    total = counts.sum()
    return counts / total if total else counts


def expected_iid_counts(keys, w, freqs, n_windows):
    """Expected multiplicity of each packed w-window under an i.i.d. model.

    The model has the *current* byte frequencies, so it already knows that the
    soup is 82% '<'; a window only looks surprising if its arrangement is
    surprising, not merely its letters.
    """
    logp = np.log(np.maximum(freqs, 1e-300))
    total = np.zeros(keys.size, dtype=np.float64)
    for j in range(w):
        b = ((keys >> np.uint64(8 * j)) & np.uint64(255)).astype(np.int64)
        total += logp[b]
    return n_windows * np.exp(total)


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

    A window *qualifies* at a snapshot when its count is at least ``c_min``
    and, if ``null_ratio`` is set, at least ``null_ratio`` times the count an
    i.i.d. model with the soup's current byte frequencies would predict.  It
    becomes *persistent* once it has qualified continuously for ``tau_epochs``
    epochs.  A(t) only ever grows; ``n_persistent`` is how many windows are
    persistent right now.

    ``tau_epochs`` is in epochs, not snapshots, so changing the logging cadence
    does not change what A(t) counts.  (Qualification is still only observable
    at snapshots, so a streak is assumed to hold between two snapshots that
    both qualify -- that assumption is unavoidable and is the same one the
    snapshot-counting version made.)

    The null filter is what separates novelty from combinatorics.  In a soup
    that is 82% one byte, a window of eight copies of that byte occurs tens of
    thousands of times by chance; without the filter A(t) counts every such
    arrangement as a discovery.
    """

    def __init__(self, c_min=8, tau_epochs=250, null_ratio=0.0, window=8):
        self.c_min = int(c_min)
        self.tau_epochs = int(tau_epochs)
        self.null_ratio = float(null_ratio)
        self.window = int(window)
        self._keys = np.zeros(0, dtype=np.uint64)     # sorted, qualifying
        self._since = np.zeros(0, dtype=np.int64)     # epoch each streak began
        self.ever = np.zeros(0, dtype=np.uint64)      # sorted

    def qualifying(self, keys, counts, freqs=None, n_windows=None):
        q = keys[counts >= self.c_min]
        if q.size == 0 or self.null_ratio <= 0.0 or freqs is None:
            return q
        c = counts[counts >= self.c_min]
        expected = expected_iid_counts(q, self.window, freqs, n_windows)
        return q[c >= self.null_ratio * expected]

    def update(self, keys, counts, epoch, freqs=None, n_windows=None):
        """``keys`` must be sorted ascending (np.unique output)."""
        q = self.qualifying(keys, counts, freqs, n_windows)
        if q.size == 0:
            self._keys = np.zeros(0, dtype=np.uint64)
            self._since = np.zeros(0, dtype=np.int64)
            return len(self.ever), 0
        since = np.full(q.size, epoch, dtype=np.int64)
        if self._keys.size:
            idx = np.searchsorted(self._keys, q)
            idx_c = np.clip(idx, 0, self._keys.size - 1)
            hit = self._keys[idx_c] == q
            since[hit] = self._since[idx_c[hit]]
        self._keys, self._since = q, since
        now_persistent = q[(epoch - since) >= self.tau_epochs]
        if now_persistent.size:
            self.ever = np.union1d(self.ever, now_persistent)
        return len(self.ever), int(now_persistent.size)

    @property
    def a_t(self):
        return int(self.ever.size)
