"""Drive the plateau continuations to a target epoch, restartable at will.

Background work in this environment does not survive an idle period, so
this is written to be run again and again: each call finds the latest tape
dump of each run, resumes from it, and stops when the target epoch is
reached.  Nothing is recomputed and the logs are continuous, because
soup.run truncates past the resume point and reloads A(t) from the tracker
state saved beside the dump.

    python -m experiments.resume --target 55500 --jobs 4
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

EPOCH_RE = re.compile(r"epoch_(\d+)\.npy$")
ENV = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")

# seed -> the epoch of the checkpoint the takeover was detected at
ORIGIN = {12: 5500, 13: 5750, 14: 7000, 19: 12750}
EXTRA = 50000            # epochs to add past the origin


def latest_dump(run_dir):
    dumps = []
    for p in glob.glob(os.path.join(run_dir, "snapshots", "epoch_*.npy")):
        m = EPOCH_RE.search(p)
        if m:
            dumps.append((int(m.group(1)), p))
    return max(dumps) if dumps else None


def step(spec):
    """One resume of one run; returns a line describing what happened."""
    tag, config, seed, origin_ck, origin_epoch, target, chunk = spec
    run_dir = os.path.join("runs", tag)
    found = latest_dump(run_dir)
    start, ck = found if found else (origin_epoch, origin_ck)
    if start >= target:
        return "%s already at %d" % (tag, start)
    n = min(chunk, target - start) if chunk else target - start
    cmd = [sys.executable, "-m", "soup.run", "--config", config,
           "--seed", str(seed), "--load", ck, "--start-epoch", str(start),
           "--epochs", str(n), "--out", run_dir, "--quiet"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV)
    if r.returncode:
        return "FAILED %s at %d: %s" % (tag, start, r.stderr.strip()[-400:])
    return "%s %d -> %d in %.0fs" % (tag, start, start + n, time.time() - t0)


def specs(seeds, target_extra, config, chunk):
    out = []
    for seed in seeds:
        origin = ORIGIN[seed]
        ck = "runs/compat_cubff_s%d/snapshots/epoch_%08d.npy" % (seed, origin)
        out.append(("plateau_s%d" % seed, config, seed, ck, origin,
                    origin + target_extra, chunk))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seeds", type=int, nargs="+", default=sorted(ORIGIN))
    ap.add_argument("--extra", type=int, default=EXTRA,
                    help="epochs past the takeover checkpoint")
    ap.add_argument("--config", default="configs/plateau.json")
    ap.add_argument("--chunk", type=int, default=0,
                    help="epochs per subprocess (0 = all the way)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=1)
    a = ap.parse_args(argv)
    todo = specs(a.seeds, a.extra, a.config, a.chunk)
    for r in range(a.rounds):
        with ThreadPoolExecutor(a.jobs) as pool:
            for msg in pool.map(step, todo):
                print(msg, flush=True)
        if all(latest_dump(os.path.join("runs", s[0])) and
               latest_dump(os.path.join("runs", s[0]))[0] >= s[5]
               for s in todo):
            print("all runs at target", flush=True)
            break


if __name__ == "__main__":
    main()
