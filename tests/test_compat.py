"""--compat reproduces cubff byte-for-byte.

The fixtures in tests/data are checkpoints written by cubff itself
(github.com/paradigms-of-intelligence/cubff at f212e849, built with
`make CUDA=0`), each the soup after 129 epochs of a 256-tape run.  A
checkpoint is 24 bytes of header followed by the raw soup.

This is the strongest validation in the repository: it pins our interpreter,
our pairing, our mutation and our RNG against the implementation the
"Computational Life" paper used, rather than against our own reading of it.
"""
import os

import numpy as np
import pytest

from soup.config import Config
from soup.run import Soup

DATA = os.path.join(os.path.dirname(__file__), "data")
HEADER = 24
EPOCHS = 129          # checkpoint 0000000128 is the soup after epoch 128 ran
CUBFF_MU = 2.0 ** -12  # cubff's default 1<<18 over a 1<<30 denominator


def reference(lang, seed):
    path = os.path.join(DATA, "cubff_%s_n256_seed%d_epoch128.dat" % (lang, seed))
    return np.fromfile(path, dtype=np.uint8)[HEADER:]


def simulate(compat, seed, epochs=EPOCHS, n=256):
    cfg = Config(mode="bff", compat=compat, N=n, k=8192, mu=CUBFF_MU,
                 seed=seed, epochs=0)
    soup = Soup(cfg)
    soup.advance(epochs)
    return soup


@pytest.mark.parametrize("seed", [3, 11])
@pytest.mark.parametrize("lang,compat", [("bff", "cubff"),
                                         ("bff_noheads", "cubff_noheads")])
def test_soup_matches_cubff_byte_for_byte(lang, compat, seed):
    assert np.array_equal(simulate(compat, seed).flat, reference(lang, seed))


def test_the_two_cubff_languages_actually_differ():
    """Otherwise the test above would prove nothing about the heads rule."""
    assert not np.array_equal(reference("bff", 3), reference("bff_noheads", 3))


def test_heads_are_read_from_the_first_two_bytes_in_cubff_mode():
    from soup.interp import run_region_full
    tape = np.zeros(128, dtype=np.uint8)
    tape[0] = 200          # head0 -> 200 % 128 = 72
    tape[1] = 5            # head1 -> 5
    tape[2] = ord("+")     # pc starts at 2
    buf = tape.copy()
    visited = np.zeros(128, dtype=np.int32)
    run_region_full(buf, 2, int(tape[0]) % 128, int(tape[1]) % 128, 8192,
                    True, visited, 1)
    assert buf[72] == 1    # the '+' landed 72 bytes in, not at 0


def test_compat_is_reproducible_and_seed_dependent():
    a = simulate("cubff", 3, epochs=20)
    b = simulate("cubff", 3, epochs=20)
    c = simulate("cubff", 4, epochs=20)
    assert np.array_equal(a.flat, b.flat)
    assert not np.array_equal(a.flat, c.flat)


def test_compat_mutation_rate_matches_cubff():
    soup = simulate("cubff_noheads", 3, epochs=50)
    expected = 50 * 256 * 64 * CUBFF_MU
    assert abs(soup.mutations - expected) < 5 * np.sqrt(expected)


def test_compat_is_rejected_outside_bff_mode():
    from soup.config import validate
    with pytest.raises(SystemExit):
        validate(Config(mode="ring", compat="cubff"))
