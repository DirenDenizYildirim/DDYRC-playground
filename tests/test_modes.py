"""blocks mode, the no-copy language variant, and indel mutation."""
import numpy as np
import pytest

from soup import core
from soup.interp import NO_LABELS, NO_LABELS_2D
from soup.config import Config
from soup.rng import make_state, mutate_indel
from soup.run import Soup


# --- blocks ------------------------------------------------------------------

@pytest.mark.parametrize("d", [1, 2, 8])
def test_partner_is_a_distinct_block_within_d(d):
    state = make_state(4)
    offs = [core.pick_partner(state, 1000, d) for _ in range(4000)]
    assert min(offs) == -d and max(offs) == d
    assert 0 not in offs
    assert len(set(offs)) == 2 * d          # every offset is reachable


def test_partner_offsets_are_uniform():
    state = make_state(5)
    offs = np.array([core.pick_partner(state, 1000, 2) for _ in range(8000)])
    counts = np.array([(offs == o).sum() for o in (-2, -1, 1, 2)])
    assert counts.min() > 1700 and counts.max() < 2300


def test_blocks_pairing_wraps_around_the_ring():
    state = make_state(6)
    n = 16
    seen = set()
    for _ in range(2000):
        j = (0 + core.pick_partner(state, n, 2)) % n
        seen.add(j)
    assert seen == {1, 2, 14, 15}


def test_blocks_epoch_runs_n_over_two_ticks():
    cfg = Config(mode="blocks", N=64, d=2, k=256, mu=0.0, seed=1, epochs=0)
    soup = Soup(cfg)
    soup.advance(3)
    assert soup.stats[core.STAT_RUNS] == 3 * (64 // 2)


def test_blocks_only_ever_touches_nearby_blocks():
    """One tick can only write to the two blocks it paired."""
    cfg = Config(mode="blocks", N=64, d=2, k=8192, mu=0.0, seed=2, epochs=0)
    soup = Soup(cfg)
    before = soup.mem.copy()
    state = np.array([99], dtype=np.uint64)
    stats = np.zeros(core.N_STATS, dtype=np.int64)
    visited = np.zeros(128, dtype=np.int32)
    core.blocks_epoch(soup.mem, soup.buf, 1, 2, 8192, state, stats, visited, 0,
                      NO_LABELS_2D, NO_LABELS)
    changed = np.nonzero((soup.mem != before).any(axis=1))[0]
    assert changed.size <= 2
    if changed.size == 2:
        gap = abs(int(changed[0]) - int(changed[1]))
        assert min(gap, 64 - gap) <= 2


def test_blocks_runs_are_reproducible():
    def go(seed):
        soup = Soup(Config(mode="blocks", N=64, d=2, k=256, seed=seed, epochs=0))
        soup.advance(10)
        return soup.flat.copy()
    assert np.array_equal(go(3), go(3))
    assert not np.array_equal(go(3), go(4))


def test_d_is_validated():
    from soup.config import validate
    with pytest.raises(SystemExit):
        validate(Config(mode="blocks", N=64, d=0))
    with pytest.raises(SystemExit):
        validate(Config(mode="blocks", N=64, d=33))


# --- no-copy variant ---------------------------------------------------------

def test_no_copy_never_records_a_copy():
    soup = Soup(Config(mode="ring", M=4096, R=64, k=2048, seed=1,
                       no_copy=1, epochs=0))
    soup.advance(5)
    assert soup.stats[core.STAT_COPYRUNS] == 0
    assert soup.stats[core.STAT_STEPS] > 0


def test_copies_do_happen_without_the_flag():
    soup = Soup(Config(mode="ring", M=4096, R=64, k=2048, seed=1, epochs=0))
    soup.advance(5)
    assert soup.stats[core.STAT_COPYRUNS] > 0


def test_no_copy_still_costs_a_step():
    from soup.interp import NO_LABELS, NO_LABELS_2D, run_region_full
    buf = np.frombuffer(b"..." + bytes(5), dtype=np.uint8).copy()
    visited = np.zeros(8, dtype=np.int32)
    on = run_region_full(buf.copy(), 0, 0, 0, 100, True, visited, 1,
                         NO_LABELS, True)
    off = run_region_full(buf.copy(), 0, 0, 0, 100, True, visited, 2,
                          NO_LABELS, False)
    assert on[0] == off[0]          # same step count
    assert on[4] == off[4]          # still counted as commands
    assert on[1] == 3 and off[1] == 0


# --- indel mutation ----------------------------------------------------------

def test_indel_shifts_bytes_within_the_radius():
    state = make_state(1)
    mem = np.arange(64, dtype=np.uint8)
    before = mem.copy()
    n = mutate_indel(mem, 1.0, 8, state)     # every position gets an event
    assert n == 64
    assert not np.array_equal(mem, before)


def test_indel_rate_is_as_requested():
    state = make_state(2)
    mem = np.zeros(200_000, dtype=np.uint8)
    rate = 0.0005
    total = sum(mutate_indel(mem, rate, 4, state) for _ in range(10))
    expected = rate * mem.size * 10
    assert abs(total - expected) < 5 * np.sqrt(expected)


def test_indel_moves_data_no_further_than_R():
    state = make_state(3)
    mem = np.zeros(1000, dtype=np.uint8)
    mem[500] = 77
    for _ in range(20):
        mutate_indel(mem, 0.002, 16, state)
    moved = np.nonzero(mem == 77)[0]
    if moved.size:                            # it may have been overwritten
        assert abs(int(moved[0]) - 500) <= 20 * 16


def test_indel_doubles_the_mutation_event_rate_in_a_ring_run():
    def events(indel):
        soup = Soup(Config(mode="ring", M=65536, R=128, k=64, mu=0.001,
                           seed=7, indel=indel, epochs=0))
        soup.advance(20)
        return soup.mutations
    plain, both = events(0), events(1)
    assert 1.7 * plain < both < 2.3 * plain


def test_indel_off_by_default():
    soup = Soup(Config(mode="ring", M=4096, R=32, k=64, mu=0.001, seed=1,
                       epochs=0))
    soup.advance(5)
    assert soup.cfg.indel == 0
