"""The modes.  All of them call the same :func:`soup.interp.run_region_full`.

bff     population of N tapes of 64 bytes; each epoch draws a uniformly random
        perfect matching, concatenates each pair to 128 bytes, runs it from
        ip = head0 = head1 = 0, and splits it back.

ring    one ring of M bytes.  A tick picks a uniformly random position p and
        runs from ip = head0 = head1 = p, confined to the 2R+1 window centred
        on p.  One epoch is M/64 ticks, run sequentially.

blocks  a ring of N blocks of 64 bytes -- bff's pairing with ring locality.  A
        tick picks a random block i and a partner j drawn uniformly from the
        2*d blocks within +/-d of it (wrapping, i excluded), concatenates the
        two in a random order, runs it bff-style, and writes both back.  One
        epoch is N/2 ticks.

``stats`` is a length-8 int64 accumulator:
    0 runs, 1 steps, 2 runs-with->=1-copy, 3..5 termination-code counts,
    6 revisited steps (steps spent re-running an instruction), 7 executed
    instructions that were not no-ops.
"""
import numpy as np
from numba import njit

from .interp import run_region_full
from .rng import mutate, mutate_indel, rand_below, shuffle

STAT_RUNS, STAT_STEPS, STAT_COPYRUNS = 0, 1, 2
STAT_TERM0, STAT_TERM1, STAT_TERM2 = 3, 4, 5
STAT_REVISITS, STAT_COMMANDS = 6, 7
N_STATS = 8

TAPE_LEN = 64  # bff/blocks tape length; also the ring's ticks-per-epoch divisor


@njit(inline="always")
def _record(stats, steps, ncopy, term, revisits, ncmd):
    stats[STAT_RUNS] += 1
    stats[STAT_STEPS] += steps
    if ncopy > 0:
        stats[STAT_COPYRUNS] += 1
    stats[STAT_TERM0 + term] += 1
    stats[STAT_REVISITS] += revisits
    stats[STAT_COMMANDS] += ncmd


@njit(cache=True, nogil=True)
def bff_epoch(pop, buf, perm, k, state, stats, visited, gen0,
              wrap_heads=True, allow_copy=True):
    """One bff epoch: pair up every tape, run each pair, split back."""
    n = pop.shape[0]
    shuffle(perm, state)
    gen = gen0
    for i in range(0, n - 1, 2):
        a = perm[i]
        b = perm[i + 1]
        buf[0:TAPE_LEN] = pop[a]
        buf[TAPE_LEN:2 * TAPE_LEN] = pop[b]
        gen += 1
        steps, ncopy, term, rev, ncmd = run_region_full(
            buf, 0, 0, 0, k, wrap_heads, visited, gen, allow_copy)
        pop[a] = buf[0:TAPE_LEN]
        pop[b] = buf[TAPE_LEN:2 * TAPE_LEN]
        _record(stats, steps, ncopy, term, rev, ncmd)
    return gen


@njit(cache=True, nogil=True)
def pick_partner(state, n, d):
    """A block index drawn uniformly from the 2d neighbours within +/-d.

    The block itself is excluded, so a tick always pairs two distinct blocks,
    and the ring wraps.  Exposed so the neighbourhood can be tested directly.
    """
    off = rand_below(state, 2 * d)
    if off >= d:
        return off - d + 1     # offsets +1 .. +d
    return off - d             # offsets -d .. -1


@njit(cache=True, nogil=True)
def blocks_epoch(pop, buf, ticks, d, k, state, stats, visited, gen0,
                 wrap_heads=True, allow_copy=True):
    """One blocks epoch: ticks x (pick a block, pick a near neighbour, run)."""
    n = pop.shape[0]
    gen = gen0
    for _ in range(ticks):
        i = rand_below(state, n)
        j = (i + pick_partner(state, n, d)) % n
        if rand_below(state, 2) == 0:
            a, b = i, j
        else:
            a, b = j, i
        buf[0:TAPE_LEN] = pop[a]
        buf[TAPE_LEN:2 * TAPE_LEN] = pop[b]
        gen += 1
        steps, ncopy, term, rev, ncmd = run_region_full(
            buf, 0, 0, 0, k, wrap_heads, visited, gen, allow_copy)
        pop[a] = buf[0:TAPE_LEN]
        pop[b] = buf[TAPE_LEN:2 * TAPE_LEN]
        _record(stats, steps, ncopy, term, rev, ncmd)
    return gen


@njit(cache=True, nogil=True)
def ring_tick_at(ring, buf, p, R, k, stats, visited, gen,
                 wrap_heads=True, allow_copy=True):
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
    steps, ncopy, term, rev, ncmd = run_region_full(
        buf, R, R, R, k, wrap_heads, visited, gen, allow_copy)
    if start + L <= m:
        ring[start:start + L] = buf
    else:
        n1 = m - start
        ring[start:] = buf[:n1]
        ring[:L - n1] = buf[n1:]
    _record(stats, steps, ncopy, term, rev, ncmd)
    return steps, ncopy, term


@njit(cache=True, nogil=True)
def ring_epoch(ring, buf, ticks, R, k, state, stats, visited, gen0,
               wrap_heads=True, allow_copy=True):
    m = ring.shape[0]
    gen = gen0
    for _ in range(ticks):
        p = rand_below(state, m)
        gen += 1
        ring_tick_at(ring, buf, p, R, k, stats, visited, gen,
                     wrap_heads, allow_copy)
    return gen


@njit(cache=True, nogil=True)
def bff_chunk(pop, buf, perm, n_epochs, k, mu, state, stats, visited, gen0,
              wrap_heads=True, allow_copy=True):
    n_mut = 0
    gen = gen0
    flat = pop.reshape(pop.shape[0] * pop.shape[1])
    for _ in range(n_epochs):
        gen = bff_epoch(pop, buf, perm, k, state, stats, visited, gen,
                        wrap_heads, allow_copy)
        n_mut += mutate(flat, mu, state)
    return n_mut, gen


@njit(cache=True, nogil=True)
def blocks_chunk(pop, buf, n_epochs, ticks, d, k, mu, state, stats,
                 visited, gen0, wrap_heads=True, allow_copy=True):
    n_mut = 0
    gen = gen0
    flat = pop.reshape(pop.shape[0] * pop.shape[1])
    for _ in range(n_epochs):
        gen = blocks_epoch(pop, buf, ticks, d, k, state, stats, visited, gen,
                           wrap_heads, allow_copy)
        n_mut += mutate(flat, mu, state)
    return n_mut, gen


@njit(cache=True, nogil=True)
def ring_chunk(ring, buf, n_epochs, ticks, R, k, mu, state, stats,
               visited, gen0, wrap_heads=True, allow_copy=True, indel=False):
    n_mut = 0
    gen = gen0
    for _ in range(n_epochs):
        gen = ring_epoch(ring, buf, ticks, R, k, state, stats, visited, gen,
                         wrap_heads, allow_copy)
        n_mut += mutate(ring, mu, state)
        if indel:
            n_mut += mutate_indel(ring, mu, R, state)
    return n_mut, gen
