"""Bit-exact reproduction of cubff's bff soup.

cubff (github.com/paradigms-of-intelligence/cubff) is the implementation the
"Computational Life" paper used.  This module reproduces its `bff` and
`bff_noheads` languages exactly -- the same SplitMix64 streams for
initialisation, pairing and mutation, and the same order of operations -- so
that a run here can be diffed byte-for-byte against a cubff checkpoint.
`tests/test_compat.py` does exactly that.

The interpreter itself is the same :func:`soup.interp.run_region_full` used by
every other mode; only the soup-level scaffolding differs.  The semantic
differences between cubff and the spec this repository implements are listed in
RESULTS.md; the one that matters is the initial machine state:

    bff          head0 = tape[0] % 128, head1 = tape[1] % 128, pc = 2
    bff_noheads  head0 = head1 = 0, pc = 0          (what our spec says)

Two constants are baked in because cubff bakes them in: tapes are 64 bytes and
the step budget is 8192.
"""
import numpy as np
from numba import njit

from .core import (STAT_COMMANDS, STAT_COPIES, STAT_COPYRUNS, STAT_REVISITS,
                   STAT_RUNS, STAT_STEPS, STAT_TERM0)
from .interp import run_region_full

TAPE = 64
PAIR = 2 * TAPE
CUBFF_STEPS = 8 * 1024
CUBFF_MUTATION_DENOM = 1 << 30          # mutation_prob is a fraction of this
CUBFF_MUTATION_DEFAULT = 1 << 18        # = 2**-12 = 0.000244140625

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(inline="always")
def splitmix64(x):
    # the cast matters: given an int64 numba would sign-extend the shifts and
    # silently produce a different stream for some seeds
    z = np.uint64(x) + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@njit(inline="always")
def seed_of(seed_base, x):
    """cubff's `seed` lambda: SplitMix64(SplitMix64(params.seed) ^ SplitMix64(x))."""
    return splitmix64(np.uint64(seed_base) ^ splitmix64(np.uint64(x)))


def seed_base_for(seed):
    """SplitMix64(params.seed), computed once on the host."""
    return np.uint64(splitmix64(np.uint64(seed)))


def seed_of_host(seed_base, x):
    """`seed_of` for callers outside nopython mode.

    numba hands a uint64 result back as a plain Python int, and a later call
    passing that int into another kernel gets typed by its *value* -- so a
    small seed compiles the kernel for int64 and a large one then fails with
    "int too big to convert".  Casting on the way out pins it to uint64.
    """
    return np.uint64(seed_of(np.uint64(seed_base), np.uint64(x)))


@njit(cache=True, nogil=True)
def cubff_init(pop, seed0):
    """InitPrograms: prog[i] = SplitMix64(64*N*seed + 64*index + i) % 256."""
    n = pop.shape[0]
    base = np.uint64(TAPE) * np.uint64(n) * seed0
    for index in range(n):
        off = base + np.uint64(TAPE) * np.uint64(index)
        for i in range(TAPE):
            pop[index, i] = np.uint8(splitmix64(off + np.uint64(i)) % np.uint64(256))


@njit(cache=True, nogil=True)
def cubff_shuffle(s, epoch, seed_base):
    """do_shuffle: identity, then a descending Fisher-Yates over the whole array.

    The reset matters: cubff rebuilds the identity permutation every epoch and
    shuffles that, rather than reshuffling the previous epoch's pairing.
    """
    n = s.shape[0]
    for i in range(n):
        s[i] = i
    for i in range(n - 1, -1, -1):
        x = np.uint64(epoch) * np.uint64(n) + np.uint64(i)
        j = np.int64(splitmix64(seed_of(seed_base, x)) % np.uint64(i + 1))
        tmp = s[i]
        s[i] = s[j]
        s[j] = tmp


@njit(cache=True, nogil=True)
def cubff_epoch(pop, buf, s, epoch, seed_base, mut_prob, heads_from_tape,
                k, stats, visited, gen0, labels, lab_buf):
    """One cubff epoch: shuffle, then for each pair mutate-then-run-then-store.

    cubff mutates the concatenated pair immediately before executing it, not
    the whole soup after the epoch; every program is in exactly one pair, so
    each byte still gets exactly one mutation trial per epoch.
    """
    n = pop.shape[0]
    cubff_shuffle(s, epoch, seed_base)
    eseed = seed_of(seed_base, np.uint64(epoch))
    mp = np.uint64(mut_prob)
    mask30 = np.uint64(CUBFF_MUTATION_DENOM - 1)
    gen = gen0
    n_mut = 0
    for index in range(n // 2):
        p1 = s[2 * index]
        p2 = s[2 * index + 1]
        buf[0:TAPE] = pop[p1]
        buf[TAPE:PAIR] = pop[p2]
        if labels.shape[0] > 0:
            lab_buf[0:TAPE] = labels[p1]
            lab_buf[TAPE:PAIR] = labels[p2]
        row = (np.uint64(n) * eseed + np.uint64(index)) * np.uint64(PAIR)
        for i in range(PAIR):
            rng = splitmix64(row + np.uint64(i))
            if ((rng >> np.uint64(8)) & mask30) < mp:
                buf[i] = np.uint8(rng & np.uint64(255))
                n_mut += 1
        if heads_from_tape:
            h0 = np.int64(buf[0]) % PAIR
            h1 = np.int64(buf[1]) % PAIR
            pc = 2
        else:
            h0 = 0
            h1 = 0
            pc = 0
        gen += 1
        steps, ncopy, term, revis, ncmd = run_region_full(
            buf, pc, h0, h1, k, True, visited, gen, lab_buf)
        pop[p1] = buf[0:TAPE]
        pop[p2] = buf[TAPE:PAIR]
        if labels.shape[0] > 0:
            labels[p1] = lab_buf[0:TAPE]
            labels[p2] = lab_buf[TAPE:PAIR]
        stats[STAT_RUNS] += 1
        stats[STAT_STEPS] += steps
        if ncopy > 0:
            stats[STAT_COPYRUNS] += 1
        stats[STAT_TERM0 + term] += 1
        stats[STAT_REVISITS] += revis
        stats[STAT_COMMANDS] += ncmd
        stats[STAT_COPIES] += ncopy
    return n_mut, gen


@njit(cache=True, nogil=True)
def cubff_chunk(pop, buf, s, epoch0, n_epochs, seed_base, mut_prob,
                heads_from_tape, k, stats, visited, gen0, labels, lab_buf):
    gen = gen0
    total = 0
    for e in range(n_epochs):
        n_mut, gen = cubff_epoch(pop, buf, s, epoch0 + e, seed_base, mut_prob,
                                 heads_from_tape, k, stats, visited, gen,
                                 labels, lab_buf)
        total += n_mut
    return total, gen
