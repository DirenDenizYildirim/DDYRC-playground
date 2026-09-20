"""A small explicit PRNG so runs are reproducible from the seed alone.

We do not use numpy's or numba's global RNG: the point is that the byte stream
consumed by a run depends only on (seed, algorithm), never on numba/numpy
version, thread count, or how the run is chunked between snapshots.

xorshift64* -- state is a length-1 uint64 array so numba can mutate it.
"""
import numpy as np
from numba import njit

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)


def make_state(seed):
    """SplitMix64-expand ``seed`` into a non-zero xorshift64* state."""
    with np.errstate(over="ignore"):  # wraparound is the point
        z = np.uint64(seed) + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    if z == np.uint64(0):
        z = np.uint64(0x9E3779B97F4A7C15)
    return np.array([z], dtype=np.uint64)


@njit(inline="always")
def next_u64(state):
    x = state[0]
    x ^= x >> np.uint64(12)
    x ^= x << np.uint64(25)
    x ^= x >> np.uint64(27)
    state[0] = x
    return x * np.uint64(2685821657736338717)


@njit(inline="always")
def rand_below(state, n):
    """Uniform-ish integer in [0, n).

    Plain modulo.  The bias is ~n/2**64 and is irrelevant for n <= 2**20.
    """
    return np.int64(next_u64(state) % np.uint64(n))


@njit(inline="always")
def rand_byte(state):
    return np.uint8(next_u64(state) & np.uint64(255))


@njit(inline="always")
def rand_float(state):
    """Uniform in [0, 1) with 53 bits of resolution."""
    return np.float64(next_u64(state) >> np.uint64(11)) * (1.0 / 9007199254740992.0)


@njit(cache=True, nogil=True)
def fill_random(mem, state):
    for i in range(mem.shape[0]):
        mem[i] = rand_byte(state)


@njit(cache=True, nogil=True)
def mutate(mem, mu, state):
    """Flip each byte to a uniformly random byte with probability ``mu``.

    Implemented by sampling geometric gaps instead of drawing one variate per
    byte; statistically identical, ~M/(mu*M) times cheaper.  The replacement
    byte is drawn uniformly from 0..255, so a mutation can be a no-op.
    """
    if mu <= 0.0:
        return 0
    n = mem.shape[0]
    if mu >= 1.0:
        for i in range(n):
            mem[i] = rand_byte(state)
        return n
    log1m = np.log1p(-mu)
    i = -1
    count = 0
    while True:
        u = rand_float(state)
        if u <= 0.0:
            u = 1e-300
        gap = np.int64(np.floor(np.log(u) / log1m)) + 1
        i += gap
        if i >= n:
            break
        mem[i] = rand_byte(state)
        count += 1
    return count


@njit(cache=True, nogil=True)
def shuffle(perm, state):
    """In-place Fisher-Yates."""
    for i in range(perm.shape[0] - 1, 0, -1):
        j = rand_below(state, i + 1)
        tmp = perm[i]
        perm[i] = perm[j]
        perm[j] = tmp
