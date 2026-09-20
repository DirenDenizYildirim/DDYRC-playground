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


def test_a_window_needs_tau_consecutive_snapshots_to_become_persistent():
    t = metrics.PersistenceTracker(c_min=8, tau=3)
    assert t.update(*mk({1: 10})) == (0, 0)
    assert t.update(*mk({1: 10})) == (0, 0)
    assert t.update(*mk({1: 10})) == (1, 1)


def test_a_broken_streak_restarts_the_count():
    t = metrics.PersistenceTracker(c_min=8, tau=3)
    t.update(*mk({1: 10}))
    t.update(*mk({1: 10}))
    t.update(*mk({1: 2}))          # falls below c_min
    assert t.update(*mk({1: 10})) == (0, 0)
    t.update(*mk({1: 10}))
    assert t.update(*mk({1: 10})) == (1, 1)


def test_counts_below_c_min_never_persist():
    t = metrics.PersistenceTracker(c_min=8, tau=2)
    for _ in range(10):
        assert t.update(*mk({1: 7})) == (0, 0)


def test_a_t_is_cumulative_and_never_decreases():
    t = metrics.PersistenceTracker(c_min=8, tau=2)
    t.update(*mk({1: 10}))
    t.update(*mk({1: 10}))
    assert t.a_t == 1
    t.update(*mk({2: 10}))         # window 1 disappears entirely
    t.update(*mk({2: 10}))
    assert t.a_t == 2              # cumulative: still counts window 1
    assert t.update(*mk({}))[1] == 0   # nothing persistent right now
    assert t.a_t == 2


def test_distinct_windows_are_counted_separately():
    t = metrics.PersistenceTracker(c_min=2, tau=1)
    a_t, now = t.update(*mk({1: 5, 2: 5, 3: 1}))
    assert (a_t, now) == (2, 2)
