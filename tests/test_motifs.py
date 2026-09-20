"""Motif families and the post-takeover analysers."""
import numpy as np
import pytest

from soup import metrics, motifs, posthoc


def tapes(rows, seed=0):
    """Each row laid at the start of a tape, the rest filled with noise.

    Zero padding would make the all-zero window the commonest one in the
    soup, so the padding has to be random or the fixtures measure the pad.
    """
    rng = np.random.default_rng(seed)
    mem = rng.integers(0, 256, size=(len(rows), 64), dtype=np.uint8)
    for i, r in enumerate(rows):
        mem[i, :len(r)] = np.frombuffer(r, dtype=np.uint8)
    return mem


def rand_soup(seed, n=256):
    return np.random.default_rng(seed).integers(
        0, 256, size=(n, 64), dtype=np.uint8)


def test_edit_distance_matches_a_known_case():
    a = np.frombuffer(b"kitten", dtype=np.uint8)
    b = np.frombuffer(b"sitting", dtype=np.uint8)
    assert motifs.edit_distance(a, b) == 3
    assert motifs.edit_distance(a, a) == 0


def test_edit_distance_is_symmetric():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.integers(0, 256, 16, dtype=np.uint8)
        b = rng.integers(0, 256, 16, dtype=np.uint8)
        assert motifs.edit_distance(a, b) == motifs.edit_distance(b, a)


def test_shifts_of_one_string_land_in_one_family():
    motif = bytes(range(40, 72))                 # 32 distinct bytes
    rows = [motif[s:s + 24] for s in range(8) for _ in range(6)]
    fams = motifs.snapshot_families(tapes(rows), top=50)
    assert fams[0]["coverage"] == pytest.approx(1.0)
    assert fams[0]["size"] >= 8                  # every shift, one family
    # the rest is the random padding (and windows straddling it): every one
    # of those windows occurs exactly once in the soup
    assert all(f["top_count"] == 1 for f in fams[1:])


def test_two_unrelated_motifs_stay_apart():
    a = bytes(range(40, 72))
    b = bytes(range(200, 232))
    rows = [a[:24]] * 24 + [b[:24]] * 24
    fams = motifs.snapshot_families(tapes(rows), top=50)
    assert len(fams) >= 2
    reps = {f["rep"][:8] for f in fams[:2]}
    assert len(reps) == 2


def test_the_clustering_threshold_is_below_the_null():
    """No unrelated pair of windows should reach MAX_DIST."""
    d = motifs.null_distances(rand_soup(1, 512), pairs=500)
    assert d.min() > motifs.MAX_DIST


def test_coverage_is_the_fraction_of_tapes_carrying_the_family():
    motif = bytes(range(40, 72))
    rows = [motif[:24]] * 25 + [bytes(24)] * 75
    _, counts, ids = motifs.window_ids(tapes(rows))
    windows, _, _ = motifs.window_ids(tapes(rows))
    fam = [i for i in range(windows.shape[0])
           if windows[i].tobytes() == motif[:16]]
    assert motifs.coverage(ids, fam) == pytest.approx(0.25)


def test_window_ids_rejects_a_ring():
    with pytest.raises(SystemExit):
        motifs.window_ids(np.zeros(4096, dtype=np.uint8))


def test_track_records_a_sweep(tmp_path):
    """Two snapshots: motif A everywhere, then motif B everywhere."""
    a = bytes(range(40, 72))
    b = bytes(range(150, 182))
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    np.save(snaps / "epoch_00000000.npy", tapes([a[:24]] * 64))
    np.save(snaps / "epoch_00001000.npy", tapes([b[:24]] * 64))
    _, sweeps = motifs.track(str(tmp_path), str(tmp_path / "m"), top=50,
                             quiet=True)
    assert len(sweeps) == 1
    assert sweeps[0]["epoch"] == 1000
    assert sweeps[0]["to_coverage"] == pytest.approx(1.0)


def test_track_records_no_sweep_when_the_leader_holds(tmp_path):
    a = bytes(range(40, 72))
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    for e in (0, 1000, 2000):
        np.save(snaps / ("epoch_%08d.npy" % e), tapes([a[:24]] * 64))
    _, sweeps = motifs.track(str(tmp_path), str(tmp_path / "m"), top=50,
                             quiet=True)
    assert sweeps == []


def test_random_and_random_colonise_equally():
    """The floor: neither side of a random pair has an advantage."""
    ss = posthoc.self_sufficiency(rand_soup(3), samples=400, seed=1)
    assert abs(ss["random_vs_random"]["colonisation"] - 0.5) < 0.05


def test_a_replicator_colonises_a_random_partner():
    """The hand-written replicator is written for the spec's head rule."""
    from soup import replicator
    prog = replicator.as_array()
    mem = np.zeros((64, 64), dtype=np.uint8)
    mem[:, :prog.size] = prog
    ss = posthoc.self_sufficiency(mem, samples=200, seed=1,
                                  heads_from_tape=False)
    assert ss["soup_vs_random"]["copy_rate"] > 0.9
    assert ss["soup_vs_random"]["colonisation"] > 0.55


def test_a_dead_soup_never_gains_ground():
    """All no-ops: it cannot copy, and a random partner can only take from it.

    copy_rate is not directional -- the random partner does the copying --
    so the claim to test is that colonisation never rises above a half.
    """
    ss = posthoc.self_sufficiency(np.zeros((64, 64), dtype=np.uint8),
                                  samples=200, seed=1)
    assert ss["soup_vs_random"]["colonisation"] <= 0.5


def test_the_motif_mask_covers_the_motif_and_nothing_else():
    motif = bytes(range(40, 72))
    mem = tapes([motif[:16] + bytes(48)] * 8)
    windows, _, ids = motifs.window_ids(mem)
    fam = [i for i in range(windows.shape[0])
           if windows[i].tobytes() == motif[:16]]
    mask = posthoc.motif_mask(mem, fam, ids)
    assert mask[:, :16].all()
    assert not mask[:, 16:].any()


def test_the_alignment_finds_a_conserved_run_past_the_seed():
    """A 32-byte conserved block with random flanks."""
    rng = np.random.default_rng(0)
    core = bytes(range(60, 92))
    mem = rng.integers(0, 256, size=(256, 64), dtype=np.uint8)
    mem[:, 16:48] = np.frombuffer(core, dtype=np.uint8)
    windows, counts, ids = motifs.window_ids(mem)
    fams = motifs.snapshot_families(mem, top=50)
    off, ent, n, n_aligned = posthoc.alignment_profile(
        mem, ids, [fams[0]["rep_id"]], span=32)
    lo, hi, width = posthoc.conserved_span(off, ent)
    assert n_aligned == 256
    assert width >= 16              # at least the seed window
    assert hi - lo + 1 <= 32 + 2    # and not beyond the conserved block


def test_snapshot_row_on_a_random_soup_has_no_motif_advantage():
    row = posthoc.snapshot_row(rand_soup(5), 0, samples=128)
    assert abs(row["ss_colonisation"] - 0.5) < 0.08
