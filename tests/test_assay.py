"""The competition assay: does it measure what it claims to?"""
import csv

import numpy as np
import pytest

from soup import replicator
from soup.assay import LABEL_A, LABEL_B, build, one_replicate, run_assay


def rand_soup(seed, n=64):
    return np.random.default_rng(seed).integers(
        0, 256, size=(n, 64), dtype=np.uint8)


def repl_soup(n=64):
    prog = replicator.as_array()
    mem = np.zeros((n, 64), dtype=np.uint8)
    mem[:, :prog.size] = prog
    return mem


def test_build_splits_the_population_in_half():
    mem, labels = build(rand_soup(1), rand_soup(2), 128,
                        np.random.default_rng(0))
    assert (labels == LABEL_A).sum() == 64 * 64
    assert (labels == LABEL_B).sum() == 64 * 64


def test_every_tape_carries_one_label():
    _, labels = build(rand_soup(1), rand_soup(2), 128,
                      np.random.default_rng(0))
    assert set(np.unique(labels)) == {LABEL_A, LABEL_B}
    for tape in labels:
        assert len(set(tape.tolist())) == 1


def test_tapes_come_from_the_soups_they_are_labelled_with():
    a, b = rand_soup(1), rand_soup(2)
    mem, labels = build(a, b, 128, np.random.default_rng(0))
    a_rows = {row.tobytes() for row in a}
    b_rows = {row.tobytes() for row in b}
    for tape, lab in zip(mem, labels):
        want = a_rows if lab[0] == LABEL_A else b_rows
        assert tape.tobytes() in want


def test_an_odd_population_still_fills_every_slot():
    mem, labels = build(rand_soup(1), rand_soup(2), 65,
                        np.random.default_rng(0))
    assert (labels[:, 0] == LABEL_A).sum() == 33
    assert (labels[:, 0] == LABEL_B).sum() == 32


def rows(a, b, epochs=30, n=128, seed=1000):
    return list(one_replicate(a, b, n, epochs, 10, seed, 0.000244140625, 8192))


def test_the_assay_starts_at_a_half():
    r = rows(rand_soup(1), rand_soup(2))
    assert r[0]["frac_bytes_a"] == 0.5
    assert r[0]["frac_tapes_a"] == 0.5


def test_a_replicator_takes_bytes_from_a_dead_soup():
    """The one case where the answer is known in advance."""
    dead = np.zeros((64, 64), dtype=np.uint8)     # every byte a no-op
    r = rows(dead, repl_soup(), epochs=40)
    assert r[-1]["frac_bytes_a"] < 0.45


def test_a_dead_soup_never_takes_bytes_from_a_replicator():
    dead = np.zeros((64, 64), dtype=np.uint8)
    r = rows(repl_soup(), dead, epochs=40)
    assert r[-1]["frac_bytes_a"] > 0.55


def test_the_neutral_control_stays_near_a_half():
    """A against itself: the only movement should be drift."""
    a = rand_soup(1)
    finals = [rows(a, a, epochs=40, seed=1000 + i)[-1]["frac_bytes_a"]
              for i in range(3)]
    assert abs(np.mean(finals) - 0.5) < 0.05


def test_the_assay_does_not_read_its_own_labels():
    """One population, labels swapped: same memory, mirrored curve.

    If the machine could see a label, relabelling would change where the
    bytes go.  It cannot, so the two runs differ only in the bookkeeping.
    """
    from soup.assay import fractions
    from soup.config import Config
    from soup.run import Soup

    mem, labels = build(rand_soup(1), rand_soup(2), 128,
                        np.random.default_rng(0))
    out = []
    for lab in (labels, np.where(labels == LABEL_A, LABEL_B, LABEL_A)):
        cfg = Config(mode="bff", compat="cubff", N=128, k=8192,
                     mu=0.000244140625, seed=1000, epochs=30, labels=1)
        soup = Soup(cfg)
        soup.mem[...] = mem
        soup.labels[...] = lab
        soup.advance(30)
        out.append((soup.mem.copy(), fractions(soup)))
    assert np.array_equal(out[0][0], out[1][0])
    assert out[0][1][0] == pytest.approx(1.0 - out[1][1][0])
    assert out[0][1][1] == pytest.approx(1.0 - out[1][1][1])


def test_run_assay_writes_one_row_per_rep_and_epoch(tmp_path):
    np.save(tmp_path / "a.npy", rand_soup(1))
    np.save(tmp_path / "b.npy", rand_soup(2))
    out = str(tmp_path / "out")
    path = run_assay(str(tmp_path / "a.npy"), str(tmp_path / "b.npy"), out,
                     n=128, epochs=20, every=10, reps=2, quiet=True)
    with open(path) as fh:
        got = list(csv.DictReader(fh))
    assert [(r["rep"], r["epoch"]) for r in got] == [
        ("0", "0"), ("0", "10"), ("0", "20"),
        ("1", "0"), ("1", "10"), ("1", "20")]


def test_a_ring_dump_is_rejected(tmp_path):
    np.save(tmp_path / "ring.npy", np.zeros(4096, dtype=np.uint8))
    np.save(tmp_path / "b.npy", rand_soup(2))
    with pytest.raises(SystemExit):
        run_assay(str(tmp_path / "ring.npy"), str(tmp_path / "b.npy"),
                  str(tmp_path / "out"), n=128, epochs=10, quiet=True)
