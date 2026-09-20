"""The two modes.  Both call the same :func:`soup.interp.run_region`.

bff   -- population of N tapes of 64 bytes.  Each epoch: a uniformly random
         perfect matching of the tapes; each pair is concatenated to 128
         bytes, run from ip=0 with head0=head1=0, then split back.

ring  -- one ring of M bytes.  One tick: pick a uniformly random position p,
         run from ip=p with head0=head1=p, confined to the 2R+1 window
         centred on p (wrapping on the ring).  One epoch is M/64 ticks, run
         sequentially.

``stats`` is a length-6 int64 accumulator:
    0 runs, 1 steps, 2 runs-with->=1-copy, 3..5 termination-code counts.
"""
import numpy as np
from numba import njit

from .interp import run_region
from .rng import mutate, rand_below, shuffle

STAT_RUNS, STAT_STEPS, STAT_COPYRUNS = 0, 1, 2
STAT_TERM0, STAT_TERM1, STAT_TERM2 = 3, 4, 5
N_STATS = 6

TAPE_LEN = 64  # bff tape length; also the ring's ticks-per-epoch divisor


@njit(cache=True, nogil=True)
def _record(stats, steps, ncopy, term):
    stats[STAT_RUNS] += 1
    stats[STAT_STEPS] += steps
    if ncopy > 0:
        stats[STAT_COPYRUNS] += 1
    stats[STAT_TERM0 + term] += 1


@njit(cache=True, nogil=True)
def bff_epoch(pop, buf, perm, k, state, stats):
    """One bff epoch: pair up every tape, run each pair, split back."""
    n = pop.shape[0]
    shuffle(perm, state)
    for i in range(0, n - 1, 2):
        a = perm[i]
        b = perm[i + 1]
        buf[0:TAPE_LEN] = pop[a]
        buf[TAPE_LEN:2 * TAPE_LEN] = pop[b]
        steps, ncopy, term = run_region(buf, 0, 0, 0, k)
        pop[a] = buf[0:TAPE_LEN]
        pop[b] = buf[TAPE_LEN:2 * TAPE_LEN]
        _record(stats, steps, ncopy, term)


@njit(cache=True, nogil=True)
def ring_tick_at(ring, buf, p, R, k, stats):
    """Run one ring tick anchored at position ``p``.  Exposed for tests."""
    m = ring.shape[0]
    L = 2 * R + 1
    start = p - R
    while start < 0:
        start += m
    if start + L <= m:
        buf[:] = ring[start:start + L]
    else:
        n1 = m - start
        buf[:n1] = ring[start:]
        buf[n1:] = ring[:L - n1]
    steps, ncopy, term = run_region(buf, R, R, R, k)
    if start + L <= m:
        ring[start:start + L] = buf
    else:
        n1 = m - start
        ring[start:] = buf[:n1]
        ring[:L - n1] = buf[n1:]
    _record(stats, steps, ncopy, term)
    return steps, ncopy, term


@njit(cache=True, nogil=True)
def ring_epoch(ring, buf, ticks, R, k, state, stats):
    m = ring.shape[0]
    for _ in range(ticks):
        p = rand_below(state, m)
        ring_tick_at(ring, buf, p, R, k, stats)


@njit(cache=True, nogil=True)
def bff_chunk(pop, buf, perm, n_epochs, k, mu, state, stats):
    n_mut = 0
    for _ in range(n_epochs):
        bff_epoch(pop, buf, perm, k, state, stats)
        n_mut += mutate(pop.reshape(pop.shape[0] * pop.shape[1]), mu, state)
    return n_mut


@njit(cache=True, nogil=True)
def ring_chunk(ring, buf, n_epochs, ticks, R, k, mu, state, stats):
    n_mut = 0
    for _ in range(n_epochs):
        ring_epoch(ring, buf, ticks, R, k, state, stats)
        n_mut += mutate(ring, mu, state)
    return n_mut
