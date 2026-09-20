"""The hand-written self-copying program must copy itself in both modes."""
import numpy as np

from soup import core
from soup.interp import NO_LABELS, NO_LABELS_2D, run_region
from soup.replicator import OFFSET, PROGRAM, as_array

L = len(PROGRAM)


def test_program_is_a_true_replicator_in_a_bare_region():
    buf = np.zeros(128, dtype=np.uint8)
    buf[:L] = as_array()
    steps, ncopy, term = run_region(buf, 0, 0, 0, 8192)
    assert bytes(buf[OFFSET:OFFSET + L]) == PROGRAM
    assert bytes(buf[:L]) == PROGRAM      # the original is intact too
    assert ncopy >= L
    assert steps < 8192


def test_the_copy_is_byte_for_byte_including_the_counter_cell():
    # the '+' run restores the counter before the copy loop reads it, so the
    # child can replicate in turn -- this is what makes it a replicator and
    # not merely a program that copies its own body
    buf = np.zeros(128, dtype=np.uint8)
    buf[:L] = as_array()
    run_region(buf, 0, 0, 0, 8192)
    child = buf[OFFSET:OFFSET + L].copy()
    assert child[0] == PROGRAM[0] == 8

    grandparent = np.zeros(128, dtype=np.uint8)
    grandparent[:L] = child           # run the child on its own
    run_region(grandparent, 0, 0, 0, 8192)
    assert bytes(grandparent[OFFSET:OFFSET + L]) == PROGRAM


def test_replicates_in_bff_mode_through_the_epoch_driver():
    # tape A holds the program, tape B is empty.  The program copies to +64,
    # i.e. into the *second* half of the concatenation, so it only replicates
    # on the epochs where the shuffle puts it first; a handful of epochs is
    # enough for that to happen at least once.
    pop = np.zeros((2, 64), dtype=np.uint8)
    pop[0, :L] = as_array()
    buf = np.zeros(128, dtype=np.uint8)
    perm = np.arange(2, dtype=np.int64)
    visited = np.zeros(128, dtype=np.int32)
    state = np.array([12345], dtype=np.uint64)
    stats = np.zeros(core.N_STATS, dtype=np.int64)
    for _ in range(20):
        core.bff_epoch(pop, buf, perm, 8192, state, stats, visited, _ * 2,
                       NO_LABELS_2D, NO_LABELS)
        if bytes(pop[1, :L]) == PROGRAM:
            break
    assert bytes(pop[0, :L]) == PROGRAM
    assert bytes(pop[1, :L]) == PROGRAM
    assert stats[core.STAT_RUNS] >= 1


def test_replicates_in_ring_mode_through_the_tick_driver():
    m, R = 4096, 128
    ring = np.zeros(m, dtype=np.uint8)
    p = 1000
    ring[p:p + L] = as_array()
    buf = np.zeros(2 * R + 1, dtype=np.uint8)
    stats = np.zeros(core.N_STATS, dtype=np.int64)
    visited = np.zeros(2 * R + 1, dtype=np.int32)
    core.ring_tick_at(ring, buf, p, R, 8192, stats, visited, 1,
                      NO_LABELS, NO_LABELS)
    assert bytes(ring[p:p + L]) == PROGRAM
    assert bytes(ring[p + OFFSET:p + OFFSET + L]) == PROGRAM


def test_replicates_across_the_ring_wraparound():
    m, R = 4096, 128
    ring = np.zeros(m, dtype=np.uint8)
    p = m - 5                                   # program straddles the seam
    idx = (p + np.arange(L)) % m
    ring[idx] = as_array()
    buf = np.zeros(2 * R + 1, dtype=np.uint8)
    stats = np.zeros(core.N_STATS, dtype=np.int64)
    visited = np.zeros(2 * R + 1, dtype=np.int32)
    core.ring_tick_at(ring, buf, p, R, 8192, stats, visited, 1,
                      NO_LABELS, NO_LABELS)
    child = ring[(p + OFFSET + np.arange(L)) % m]
    assert bytes(child) == PROGRAM


def test_planted_replicators_take_over_a_bff_population():
    """Sanity check on the whole stack: a known replicator must spread."""
    from soup.config import Config
    from soup.run import Soup
    cfg = Config(mode="bff", N=64, k=8192, mu=0.0, seed=3, plant=1, epochs=0)
    soup = Soup(cfg)
    before = int((soup.mem[:, :L] == as_array()).all(axis=1).sum())
    soup.advance(40)
    after = int((soup.mem[:, :L] == as_array()).all(axis=1).sum())
    assert before == 1
    assert after > 8, f"replicator failed to spread: {before} -> {after}"
