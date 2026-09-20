"""Metrics: entropy, compression, window counting, A(t)."""
import numpy as np

from soup import metrics


def test_entropy_of_a_constant_array_is_zero():
    assert metrics.shannon_entropy_bits(np.zeros(1000, np.uint8)) == 0.0


def test_entropy_of_a_uniform_byte_array_is_eight_bits():
    mem = np.tile(np.arange(256, dtype=np.uint8), 100)
    assert abs(metrics.shannon_entropy_bits(mem) - 8.0) < 1e-12


def test_entropy_of_a_two_valued_array_is_one_bit():
    mem = np.array([0, 1] * 500, dtype=np.uint8)
    assert abs(metrics.shannon_entropy_bits(mem) - 1.0) < 1e-12


def test_random_bytes_have_near_zero_high_order_entropy():
    mem = np.random.default_rng(0).integers(0, 256, 65536, dtype=np.uint8)
    h, c, ho = metrics.high_order_entropy(mem)
    assert abs(ho) < 0.05


def test_a_repeating_pattern_has_large_high_order_entropy():
    mem = np.tile(np.frombuffer(b"[-}}}}]+++[.>}]", np.uint8), 5000)
    h, c, ho = metrics.high_order_entropy(mem)
    assert ho > 2.0
    assert c < 0.2


def test_compressed_bits_per_byte_of_zeros_is_tiny():
    assert metrics.compressed_bits_per_byte(np.zeros(65536, np.uint8)) < 0.05


# --- window counting ---------------------------------------------------------

def test_window_counts_without_wrap():
    mem = np.frombuffer(b"abcabc", np.uint8)
    keys, counts = metrics.count_windows_u64(mem, 3, wrap=False)
    assert counts.sum() == 4                    # abc bca cab abc
    assert keys.size == 3                       # ... of which 3 are distinct
    abc = metrics.window_keys_u64(np.frombuffer(b"abc", np.uint8), 3, False)[0]
    assert counts[np.searchsorted(keys, abc)] == 2


def test_window_counts_with_wrap():
    mem = np.frombuffer(b"abcabc", np.uint8)
    keys, counts = metrics.count_windows_u64(mem, 3, wrap=True)
    assert counts.sum() == 6                    # one window per position
    assert keys.size == 3                       # abc bca cab, twice each
    assert set(counts.tolist()) == {2}


def test_windows_do_not_cross_tape_boundaries_in_bff_mode():
    mem = np.frombuffer(b"aaaabbbb", np.uint8)  # two 4-byte "tapes"
    keys, counts = metrics.count_windows_u64(mem, 3, wrap=False, tape_len=4)
    assert counts.sum() == 4                    # 2 per tape, none across
    got = {metrics.key_to_bytes(k, 3) for k in keys}
    assert got == {b"aaa", b"bbb"}


def test_key_round_trip():
    raw = b"\x01\x02\x03\x04\x05\x06\x07\x08"
    key = metrics.window_keys_u64(np.frombuffer(raw, np.uint8), 8, False)[0]
    assert metrics.key_to_bytes(key, 8) == raw


def test_top_windows_ranks_by_count():
    mem = np.frombuffer(b"XY" * 100 + b"ZZZZ", np.uint8)
    top = metrics.top_windows(mem, 2, wrap=False, top=2)
    assert top[0][0] in (b"XY", b"YX")
    assert top[0][1] >= 99


def test_printable_rendering():
    assert metrics.printable(b"[.>}]") == "[.>}]"
    assert metrics.printable(b"\x00\x08\xff") == "..."
    assert metrics.printable(b" ") == "_"


# --- A(t) --------------------------------------------------------------------

def mk(counts_by_key):
    keys = np.array(sorted(counts_by_key), dtype=np.uint64)
    counts = np.array([counts_by_key[int(k)] for k in keys], dtype=np.int64)
    return keys, counts


def test_a_window_needs_tau_epochs_to_become_persistent():
    t = metrics.PersistenceTracker(c_min=8, tau_epochs=30)
    assert t.update(*mk({1: 10}), 0) == (0, 0)
    assert t.update(*mk({1: 10}), 20) == (0, 0)
    assert t.update(*mk({1: 10}), 30) == (1, 1)


def test_tau_is_measured_in_epochs_not_snapshots():
    """The same history logged at two cadences must give the same A(t)."""
    coarse = metrics.PersistenceTracker(c_min=8, tau_epochs=100)
    fine = metrics.PersistenceTracker(c_min=8, tau_epochs=100)
    for e in (0, 50, 100, 150, 200):
        coarse.update(*mk({1: 10}), e)
    for e in range(0, 201, 10):
        fine.update(*mk({1: 10}), e)
    assert coarse.a_t == fine.a_t == 1


def test_a_broken_streak_restarts_the_clock():
    t = metrics.PersistenceTracker(c_min=8, tau_epochs=30)
    t.update(*mk({1: 10}), 0)
    t.update(*mk({1: 10}), 20)
    t.update(*mk({1: 2}), 25)          # falls below c_min
    assert t.update(*mk({1: 10}), 30) == (0, 0)
    assert t.update(*mk({1: 10}), 55) == (0, 0)
    assert t.update(*mk({1: 10}), 60) == (1, 1)


def test_counts_below_c_min_never_persist():
    t = metrics.PersistenceTracker(c_min=8, tau_epochs=0)
    for e in range(10):
        assert t.update(*mk({1: 7}), e * 10) == (0, 0)


def test_a_t_is_cumulative_and_never_decreases():
    t = metrics.PersistenceTracker(c_min=8, tau_epochs=10)
    t.update(*mk({1: 10}), 0)
    t.update(*mk({1: 10}), 10)
    assert t.a_t == 1
    t.update(*mk({2: 10}), 20)         # window 1 disappears entirely
    t.update(*mk({2: 10}), 30)
    assert t.a_t == 2                  # cumulative: still counts window 1
    assert t.update(*mk({}), 40)[1] == 0
    assert t.a_t == 2


def test_distinct_windows_are_counted_separately():
    t = metrics.PersistenceTracker(c_min=2, tau_epochs=0)
    a_t, now = t.update(*mk({1: 5, 2: 5, 3: 1}), 0)
    assert (a_t, now) == (2, 2)


# --- the null filter ---------------------------------------------------------

def test_null_filter_rejects_windows_a_biased_coin_would_produce():
    """In memory that is 82% one byte, a run of that byte is not a discovery."""
    mem = np.full(65536, ord("<"), dtype=np.uint8)
    mem[::9] = ord(",")                      # ~11% commas, i.i.d.-ish
    keys, counts = metrics.count_windows_u64(mem, 8, wrap=True)
    freqs = metrics.byte_frequencies(mem)
    naive = metrics.PersistenceTracker(c_min=8, tau_epochs=0, null_ratio=0.0)
    filt = metrics.PersistenceTracker(c_min=8, tau_epochs=0, null_ratio=5.0)
    n_naive = naive.qualifying(keys, counts).size
    n_filt = filt.qualifying(keys, counts, freqs, mem.size).size
    assert n_naive > 0
    assert n_filt < n_naive


def test_null_filter_keeps_a_planted_multi_byte_pattern():
    rng = np.random.default_rng(0)
    mem = rng.integers(0, 256, 65536, dtype=np.uint8)
    motif = np.frombuffer(b"[-}}}}]+", dtype=np.uint8)
    for i in range(0, 65536, 512):           # 128 copies of one 8-byte motif
        mem[i:i + 8] = motif
    keys, counts = metrics.count_windows_u64(mem, 8, wrap=True)
    freqs = metrics.byte_frequencies(mem)
    filt = metrics.PersistenceTracker(c_min=8, tau_epochs=0, null_ratio=5.0)
    kept = filt.qualifying(keys, counts, freqs, mem.size)
    key = metrics.window_keys_u64(motif, 8, False)[0]
    assert key in kept


def test_expected_iid_counts_match_a_hand_computation():
    freqs = np.zeros(256)
    freqs[0] = 0.5
    freqs[1] = 0.5
    keys = metrics.window_keys_u64(np.zeros(2, dtype=np.uint8), 2, False)[:1]
    got = metrics.expected_iid_counts(keys, 2, freqs, 1000)
    assert abs(float(got[0]) - 250.0) < 1e-9


# --- compressors -------------------------------------------------------------

def test_every_compressor_agrees_that_random_bytes_are_incompressible():
    mem = np.random.default_rng(1).integers(0, 256, 65536, dtype=np.uint8)
    for method in ("brotli6", "brotli2", "zlib", "zstd"):
        assert abs(metrics.compressed_bits_per_byte(mem, method) - 8.0) < 0.05


def test_brotli_sees_repeats_that_zlibs_window_cannot():
    """A 64 KiB ring with a period longer than zlib's 32 KiB window."""
    block = np.random.default_rng(2).integers(0, 256, 32768, dtype=np.uint8)
    mem = np.concatenate([block, block])     # one repeat, 32768 apart
    assert metrics.compressed_bits_per_byte(mem, "brotli6") < 4.2
    assert metrics.compressed_bits_per_byte(mem, "zlib") > 7.9


def test_high_order_entropy_defaults_to_the_long_window_compressor():
    mem = np.random.default_rng(3).integers(0, 256, 4096, dtype=np.uint8)
    h, c, ho = metrics.high_order_entropy(mem)
    assert abs(c - metrics.compressed_bits_per_byte(mem, "brotli6")) < 1e-12
    assert abs(ho - (h - c)) < 1e-12
