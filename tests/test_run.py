"""Reproducibility, configuration, and the end-to-end run driver."""
import csv
import hashlib
import json
import os

import numpy as np
import pytest

from soup.config import Config, build_config
from soup.run import CSV_FIELDS, Soup, main


def digest(soup):
    return hashlib.sha256(soup.flat.tobytes()).hexdigest()


def advanced(mode, seed, epochs, chunks=1, **kw):
    cfg = Config(mode=mode, seed=seed, epochs=epochs, **kw)
    soup = Soup(cfg)
    per = epochs // chunks
    for _ in range(chunks):
        soup.advance(per)
    return soup


RING = dict(M=2048, R=64, k=512)
BFF = dict(N=32, k=512)


def test_ring_runs_are_reproducible_from_the_seed():
    a = advanced("ring", 11, 20, **RING)
    b = advanced("ring", 11, 20, **RING)
    assert digest(a) == digest(b)


def test_bff_runs_are_reproducible_from_the_seed():
    a = advanced("bff", 11, 20, **BFF)
    b = advanced("bff", 11, 20, **BFF)
    assert digest(a) == digest(b)


def test_different_seeds_diverge():
    assert digest(advanced("ring", 1, 20, **RING)) != \
           digest(advanced("ring", 2, 20, **RING))


@pytest.mark.parametrize("mode,kw", [("ring", RING), ("bff", BFF)])
def test_results_do_not_depend_on_how_epochs_are_chunked(mode, kw):
    """Snapshot interval must not perturb the trajectory."""
    one = advanced(mode, 5, 24, chunks=1, **kw)
    many = advanced(mode, 5, 24, chunks=8, **kw)
    assert digest(one) == digest(many)


def test_the_initial_condition_is_uniform_random_bytes():
    soup = advanced("ring", 3, 0, **RING)
    counts = np.bincount(soup.flat, minlength=256)
    assert counts.min() > 0
    assert abs(soup.flat.mean() - 127.5) < 6


def test_zero_mutation_rate_leaves_untouched_regions_alone():
    soup = advanced("ring", 4, 5, mu=0.0, M=2048, R=8, k=4)
    assert soup.mutations == 0


def test_stats_are_accumulated_and_plausible():
    soup = advanced("ring", 6, 4, **RING)
    from soup.core import STAT_RUNS, STAT_STEPS
    assert soup.stats[STAT_RUNS] == 4 * (RING["M"] // 64)
    assert soup.stats[STAT_STEPS] > 0
    assert soup.stats[3:6].sum() == soup.stats[STAT_RUNS]


# --- configuration -----------------------------------------------------------

def test_cli_overrides_beat_the_json_config(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({"mode": "bff", "N": 64, "k": 99}))

    class Args:
        pass
    args = Args()
    args.config = str(cfg_file)
    for f in Config.__dataclass_fields__:
        setattr(args, f, None)
    args.k = 123
    cfg = build_config(args)
    assert (cfg.mode, cfg.N, cfg.k) == ("bff", 64, 123)


def test_bad_config_is_rejected(tmp_path):
    cfg_file = tmp_path / "c.json"
    cfg_file.write_text(json.dumps({"nonsense": 1}))

    class Args:
        pass
    args = Args()
    args.config = str(cfg_file)
    for f in Config.__dataclass_fields__:
        setattr(args, f, None)
    with pytest.raises(SystemExit):
        build_config(args)


def test_ring_rejects_M_smaller_than_the_window():
    class Args:
        pass
    args = Args()
    args.config = None
    for f in Config.__dataclass_fields__:
        setattr(args, f, None)
    args.mode, args.M, args.R = "ring", 64, 128
    with pytest.raises(SystemExit):
        build_config(args)


# --- end to end --------------------------------------------------------------

def test_end_to_end_run_writes_every_output(tmp_path):
    out = str(tmp_path / "run")
    main(["--out", out, "--mode", "ring", "--M", "2048", "--R", "32",
          "--k", "256", "--epochs", "20", "--snapshot-interval", "5",
          "--tape-dump-interval", "10", "--kymo-width", "256", "--seed", "1",
          "--quiet"])
    for name in ("metrics.csv", "patterns.log", "config.json",
                 "summary.json", "kymograph.npy"):
        assert os.path.exists(os.path.join(out, name)), name

    with open(os.path.join(out, "metrics.csv")) as fh:
        rows = list(csv.DictReader(fh))
    assert [r["epoch"] for r in rows] == ["0", "5", "10", "15", "20"]
    assert set(rows[0]) == set(CSV_FIELDS)
    assert float(rows[0]["entropy_bits"]) > 7.9      # random initial condition
    assert int(rows[0]["runs"]) == 0                 # nothing has run yet
    assert int(rows[1]["runs"]) == 5 * (2048 // 64)

    kymo = np.load(os.path.join(out, "kymograph.npy"))
    assert kymo.shape == (5, 256)

    snaps = sorted(os.listdir(os.path.join(out, "snapshots")))
    assert snaps == ["epoch_00000000.npy", "epoch_00000010.npy",
                     "epoch_00000020.npy"]
    assert np.load(os.path.join(out, "snapshots", snaps[0])).shape == (2048,)

    cfg = json.load(open(os.path.join(out, "config.json")))
    assert cfg["M"] == 2048 and cfg["seed"] == 1
    summary = json.load(open(os.path.join(out, "summary.json")))
    assert summary["epochs"] == 20 and summary["runs"] == 20 * 32


def test_rerunning_the_same_command_gives_the_same_csv(tmp_path):
    args = ["--mode", "bff", "--N", "32", "--k", "256", "--epochs", "10",
            "--snapshot-interval", "5", "--seed", "42", "--quiet", "--out"]
    main(args + [str(tmp_path / "a")])
    main(args + [str(tmp_path / "b")])

    def body(p):
        with open(os.path.join(p, "metrics.csv")) as fh:
            # drop the two timing columns, which are wall-clock
            return [{k: v for k, v in r.items() if k not in ("wall_s", "sim_s")}
                    for r in csv.DictReader(fh)]
    assert body(str(tmp_path / "a")) == body(str(tmp_path / "b"))
    assert (np.load(str(tmp_path / "a" / "kymograph.npy")) ==
            np.load(str(tmp_path / "b" / "kymograph.npy"))).all()
