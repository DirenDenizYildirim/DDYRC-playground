"""Provenance labels must be invisible to the machine, and resume must be exact."""
import numpy as np
import pytest

from soup import core
from soup.config import Config
from soup.interp import NO_LABELS, run_region_full
from soup.run import Soup

CUBFF_MU = 2.0 ** -12


def run(seed, labels, epochs=40, n=256, **kw):
    cfg = Config(mode="bff", compat="cubff", N=n, k=8192, mu=CUBFF_MU,
                 seed=seed, epochs=0, labels=labels, **kw)
    soup = Soup(cfg)
    if labels:
        # give every tape its own label so a copy is detectable
        soup.labels[:] = (np.arange(n, dtype=np.uint8) % 251)[:, None]
    soup.advance(epochs)
    return soup


# --- the machine cannot see a label -----------------------------------------

@pytest.mark.parametrize("seed", [1, 2, 3])
def test_labels_do_not_change_memory(seed):
    assert np.array_equal(run(seed, 0).flat, run(seed, 1).flat)


@pytest.mark.parametrize("seed", [1, 2])
def test_labels_do_not_change_any_counter(seed):
    off, on = run(seed, 0), run(seed, 1)
    assert list(off.stats) == list(on.stats)
    assert off.mutations == on.mutations


def test_labels_do_not_change_the_metric_row():
    from soup import metrics
    from soup.run import measure
    rows = []
    for labels in (0, 1):
        soup = run(4, labels)
        trackers = (metrics.PersistenceTracker(8, 250, 5.0, 8),
                    metrics.PersistenceTracker(8, 250, 0.0, 8))
        row = measure(soup, trackers)[0]
        del row["sim_s"]              # wall-clock, not a property of the soup
        rows.append(row)
    # distinct_labels is the one column labels are supposed to change: it is
    # blank when they are off.  Everything else has to match byte for byte.
    assert rows[0].pop("distinct_labels") == ""
    assert rows[1].pop("distinct_labels") == 251
    assert rows[0] == rows[1]


def test_starting_labels_do_not_change_the_trajectory():
    """Two different label assignments, same memory out."""
    a = run(5, 1)
    cfg = Config(mode="bff", compat="cubff", N=256, k=8192, mu=CUBFF_MU,
                 seed=5, epochs=0, labels=1)
    b = Soup(cfg)
    b.labels[:] = 7                      # every byte labelled the same
    b.advance(40)
    assert np.array_equal(a.flat, b.flat)
    assert not np.array_equal(a.labels, b.labels)


# --- labels follow copies ----------------------------------------------------

def test_a_copy_carries_the_source_label():
    buf = np.zeros(16, dtype=np.uint8)
    buf[:3] = np.frombuffer(b"<}.", dtype=np.uint8)
    buf[15] = 200
    labels = np.zeros(16, dtype=np.uint8)
    labels[15] = 42
    visited = np.zeros(16, dtype=np.int32)
    run_region_full(buf, 0, 0, 0, 100, True, visited, 1, labels)
    assert buf[1] == 200          # '.' copied the byte from head0=15 to head1=1
    assert labels[1] == 42        # and its label came with it


def test_arithmetic_keeps_the_cells_own_label():
    buf = np.zeros(8, dtype=np.uint8)
    buf[:2] = np.frombuffer(b"<+", dtype=np.uint8)
    labels = np.arange(8, dtype=np.uint8)
    visited = np.zeros(8, dtype=np.int32)
    run_region_full(buf, 0, 0, 0, 100, True, visited, 1, labels)
    assert buf[7] == 1
    assert list(labels) == list(range(8))


def test_mutation_keeps_the_cells_label():
    soup = run(6, 1, epochs=60)
    # cubff mutates inside the run; labels are only ever moved by a copy, so a
    # label that was never a copy destination still holds its founder's id
    assert soup.mutations > 0
    assert set(np.unique(soup.labels)) <= set(range(251))


def test_labels_are_not_part_of_memory():
    soup = run(7, 1)
    assert soup.labels.shape == soup.mem.shape
    assert soup.labels is not soup.mem


# --- resume from a checkpoint ------------------------------------------------

def test_resume_reproduces_an_uninterrupted_run(tmp_path):
    """cubff's streams are keyed on the epoch, so a resume is exact."""
    straight = run(9, 0, epochs=40)
    first = run(9, 0, epochs=20)
    path = tmp_path / "ck.npy"
    np.save(path, first.mem)
    cfg = Config(mode="bff", compat="cubff", N=256, k=8192, mu=CUBFF_MU,
                 seed=9, epochs=0, load=str(path), start_epoch=20)
    resumed = Soup(cfg)
    assert resumed.epoch == 20
    resumed.advance(20)
    assert np.array_equal(resumed.flat, straight.flat)


def test_resume_rejects_a_mismatched_checkpoint(tmp_path):
    path = tmp_path / "ck.npy"
    np.save(path, np.zeros((128, 64), dtype=np.uint8))
    cfg = Config(mode="bff", compat="cubff", N=256, seed=1, epochs=0,
                 load=str(path), start_epoch=0)
    with pytest.raises(SystemExit):
        Soup(cfg)


def test_missing_checkpoint_is_rejected():
    from soup.config import validate
    with pytest.raises(SystemExit):
        validate(Config(load="/no/such/file.npy"))
