"""The PRNG, the mutation operator and the pairing shuffle."""
import numpy as np

from soup.rng import fill_random, make_state, mutate, shuffle


def stream(seed, n=1000):
    st = make_state(seed)
    out = np.zeros(n, dtype=np.uint8)
    fill_random(out, st)
    return out


def test_same_seed_gives_the_same_stream():
    assert np.array_equal(stream(7), stream(7))


def test_different_seeds_give_different_streams():
    assert not np.array_equal(stream(7), stream(8))


def test_initial_fill_is_roughly_uniform_over_all_256_values():
    out = stream(1, 256 * 400)
    counts = np.bincount(out, minlength=256)
    assert counts.min() > 250 and counts.max() < 550
    assert abs(out.mean() - 127.5) < 2.0


def test_mutation_rate_matches_mu():
    st = make_state(5)
    mem = np.zeros(100_000, dtype=np.uint8)
    mu = 0.001
    total = sum(mutate(mem, mu, st) for _ in range(20))
    expected = mu * mem.size * 20
    assert abs(total - expected) < 5 * np.sqrt(expected)


def test_mutation_touches_the_reported_number_of_bytes():
    st = make_state(6)
    mem = np.zeros(50_000, dtype=np.uint8)
    n = mutate(mem, 0.01, st)
    # a mutation draws a uniform byte, so ~1/256 of them land on the old value
    changed = int((mem != 0).sum())
    assert 0.95 * n * (255 / 256) < changed <= n


def test_mutation_positions_are_spread_over_the_whole_array():
    st = make_state(7)
    mem = np.zeros(50_000, dtype=np.uint8)
    mutate(mem, 0.02, st)
    hit = np.nonzero(mem)[0]
    assert hit.min() < 2_000 and hit.max() > 48_000


def test_zero_mutation_rate_is_a_noop():
    st = make_state(8)
    mem = np.zeros(1000, dtype=np.uint8)
    assert mutate(mem, 0.0, st) == 0
    assert not mem.any()


def test_shuffle_is_a_permutation():
    st = make_state(9)
    perm = np.arange(1000, dtype=np.int64)
    shuffle(perm, st)
    assert np.array_equal(np.sort(perm), np.arange(1000))
    assert not np.array_equal(perm, np.arange(1000))


def test_shuffle_pairs_are_not_biased_towards_neighbours():
    st = make_state(10)
    perm = np.arange(64, dtype=np.int64)
    adjacent = 0
    for _ in range(200):
        shuffle(perm, st)
        adjacent += int((np.abs(perm[0::2] - perm[1::2]) == 1).sum())
    # 32 pairs x 200 shuffles; chance level is about 2*63/(64*63) per pair
    assert adjacent < 400
