"""Everything that runs after the plateau continuations finish.

A flat job queue over N workers, in the order the work was prioritised:

  1. the competition assays for every continued run (item 2's evidence)
  2. blocks d=2 and d=8 from random init, cubff head rule  (item 3b)
  3. the same four checkpoints continued under blocks d=2  (item 3a)
  4. ring with a data-driven head rule                     (item 4)

Jobs whose output already exists are skipped, so the script can be
re-run after an interruption.

    python -m experiments.phase2 --jobs 4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from experiments.plateau_suite import jobs_for

PLATEAU = {12: 5500, 13: 5750, 14: 7000, 19: 12750}
ENV = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")


def assay_cmds(out_dir, spacing, n, epochs, reps):
    cmds = []
    for seed in PLATEAU:
        run = "runs/plateau_s%d" % seed
        if not os.path.isdir(os.path.join(run, "snapshots")):
            continue
        for j in jobs_for(run, out_dir, spacing, n, epochs, reps):
            cmds.append((j["out"] + ".csv", [
                sys.executable, "-m", "soup.assay", "--a", j["a"], "--b",
                j["b"], "--out", j["out"], "--n", str(n), "--epochs",
                str(epochs), "--every", "10", "--reps", str(reps), "--quiet"]))
    return cmds


def run_cmds():
    """The simulation runs: (sentinel path, argv) pairs, in priority order."""
    out = []
    # item 3b -- origin rate under locality, from random init
    for d in (2, 8):
        for seed in (1, 2, 3, 4, 5, 6):
            tag = "runs/blocks_ch_d%d_s%d" % (d, seed)
            out.append((os.path.join(tag, "summary.json"), [
                sys.executable, "-m", "soup.run", "--config",
                "configs/blocks_cubffhead.json", "--d", str(d),
                "--seed", str(seed), "--out", tag, "--quiet"]))
    # item 3a -- the plateau checkpoints, continued under locality
    for seed, start in PLATEAU.items():
        ck = "runs/compat_cubff_s%d/snapshots/epoch_%08d.npy" % (seed, start)
        tag = "runs/plateau_blocks_s%d" % seed
        out.append((os.path.join(tag, "summary.json"), [
            sys.executable, "-m", "soup.run", "--config",
            "configs/plateau_blocks.json", "--seed", str(seed),
            "--load", ck, "--start-epoch", str(start), "--out", tag,
            "--quiet"]))
    # item 4 -- boundary-free ring with a data-driven head rule
    for seed in (1, 2, 3):
        tag = "runs/ring_datahead_s%d" % seed
        out.append((os.path.join(tag, "summary.json"), [
            sys.executable, "-m", "soup.run", "--config",
            "configs/ring_datahead.json", "--seed", str(seed),
            "--out", tag, "--quiet"]))
    return out


def execute(job):
    sentinel, cmd = job
    if os.path.exists(sentinel):
        return "cached  %s" % sentinel
    os.makedirs(os.path.dirname(sentinel) or ".", exist_ok=True)
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode:
        return "FAILED  %s: %s" % (sentinel, r.stderr.strip()[-400:])
    return "done    %s  (%.0fs)" % (sentinel, time.time() - t0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out", default="analysis/plateau")
    ap.add_argument("--spacing", type=int, default=5000)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-assays", action="store_true")
    ap.add_argument("--only-assays", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    queue = []
    if not a.skip_assays:
        queue += assay_cmds(a.out, a.spacing, a.n, a.epochs, a.reps)
    if not a.only_assays:
        queue += run_cmds()
    print("%d jobs, %d workers" % (len(queue), a.jobs), flush=True)
    if a.dry_run:
        for sentinel, cmd in queue:
            print("  %-70s %s" % (sentinel, " ".join(cmd[2:])))
        return
    with ThreadPoolExecutor(a.jobs) as pool:
        for i, msg in enumerate(pool.map(execute, queue), 1):
            print("[%3d/%3d] %s" % (i, len(queue), msg), flush=True)


if __name__ == "__main__":
    main()
