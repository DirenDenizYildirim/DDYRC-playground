"""Run every competition assay a plateau run needs, then draw the figures.

For each continued run this queues, at the requested spacing:

    vs_first_<epoch>   the soup at <epoch> against the first post-takeover
                       soup (the checkpoint the run was resumed from)
    vs_prev_<epoch>    the soup at <epoch> against the soup one step earlier
    control_first      the first soup against itself
    control_last       the last soup against itself

In every pair the *later* soup is side A, so a curve above 0.5 means later
beats earlier.  The two controls bracket the neutral noise band; a curve
that stays inside them has shown nothing.

    python -m experiments.plateau_suite runs/plateau_s12 runs/plateau_s13 \
        --out analysis/plateau --jobs 4
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from soup.motifs import snapshot_paths


def jobs_for(run_dir, out_dir, spacing, n=8192, epochs=300, reps=5, every=10):
    tag = os.path.basename(run_dir.rstrip("/"))
    snaps = snapshot_paths(run_dir)
    if not snaps:
        raise SystemExit("no snapshots in %s" % run_dir)
    first_epoch, first = snaps[0]
    picked = [(e, p) for e, p in snaps[1:] if e % spacing == 0]
    if picked and picked[-1][0] != snaps[-1][0]:
        picked.append(snaps[-1])
    jobs = []

    def add(name, a, b):
        jobs.append({"tag": tag, "name": name, "a": a, "b": b,
                     "out": os.path.join(out_dir, tag, name),
                     "n": n, "epochs": epochs, "reps": reps, "every": every})

    add("control_first", first, first)
    prev = first
    for e, p in picked:
        add("vs_first_%08d" % e, p, first)
        if prev is not first:
            add("vs_prev_%08d" % e, p, prev)
        prev = p
    add("control_last", prev, prev)
    return jobs


def run_job(job):
    out = job["out"]
    if os.path.exists(out + ".csv"):
        return out + ".csv (cached)"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [sys.executable, "-m", "soup.assay", "--a", job["a"], "--b", job["b"],
           "--out", out, "--n", str(job["n"]), "--epochs", str(job["epochs"]),
           "--every", str(job["every"]), "--reps", str(job["reps"]), "--quiet"]
    env = dict(os.environ, OMP_NUM_THREADS="1", NUMBA_NUM_THREADS="1")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        return "FAILED %s: %s" % (job["name"], r.stderr.strip()[-300:])
    return out + ".csv"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="analysis/plateau")
    ap.add_argument("--spacing", type=int, default=5000)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    jobs = []
    for run in a.runs:
        jobs += jobs_for(run, a.out, a.spacing, a.n, a.epochs, a.reps)
    print("%d assays, %d replicates each" % (len(jobs), a.reps), flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "jobs.json"), "w") as fh:
        json.dump(jobs, fh, indent=2)
    if a.dry_run:
        for j in jobs:
            print(" ", j["tag"], j["name"])
        return
    with ThreadPoolExecutor(a.jobs) as pool:
        for i, msg in enumerate(pool.map(run_job, jobs), 1):
            print("[%3d/%3d] %s" % (i, len(jobs), msg), flush=True)


if __name__ == "__main__":
    main()
