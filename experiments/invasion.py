"""Is the ring's evolved state a replicator, or just somewhere the soup gets stuck?

Half the ring is filled with a candidate field, half with fresh uniform random
bytes, mutation is switched off, and we watch whether the field spreads into
the random half.  A field that grows is replicating; a field that sits still is
merely an absorbing state.

The synthetic arms isolate the mechanism: the evolved field is ~83% '<' and
~11% ',', and the controls swap one of those two bytes for something inert.

    python experiments/invasion.py --epochs 400 [--snapshot runs/ring_s1/snapshots/epoch_00050000.npy]
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from soup import core                                        # noqa: E402
from soup.rng import make_state, rand_below                   # noqa: E402

M, R, K = 65536, 128, 8192
EVOLVED_MIX = {"<": 0.83, ",": 0.11}


def synthetic(walker, writer, fractions, rng):
    """A field of `walker` and `writer` bytes in the observed proportions."""
    out = rng.integers(0, 256, M // 2, dtype=np.uint8)
    u = rng.random(M // 2)
    out[u < fractions[0]] = ord(walker)
    out[(u >= fractions[0]) & (u < fractions[0] + fractions[1])] = ord(writer)
    return out


def invade(field, epochs, seed, track):
    rng = np.random.default_rng(seed)
    ring = rng.integers(0, 256, M, dtype=np.uint8)
    ring[M // 2:] = field
    buf = np.zeros(2 * R + 1, np.uint8)
    stats = np.zeros(core.N_STATS, np.int64)
    st = make_state(seed + 1000)
    left = slice(0, M // 2)
    start = float((ring[left] == ord(track)).mean())
    for _ in range(epochs):
        for _t in range(M // 64):
            core.ring_tick_at(ring, buf, rand_below(st, M), R, K, stats)
    end = float((ring[left] == ord(track)).mean())
    held = float((ring[M // 2:] == ord(track)).mean())
    h = np.bincount(ring, minlength=256) / M
    h = h[h > 0]
    return start, end, held, float(-(h * np.log2(h)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--snapshot", default="runs/ring_s1/snapshots/epoch_00050000.npy")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    fr = (EVOLVED_MIX["<"], EVOLVED_MIX[","])
    arms = []
    if os.path.exists(args.snapshot):
        arms.append(("evolved ring field (from a finished run)",
                     np.load(args.snapshot)[M // 2:], "<"))
    arms += [
        ("synthetic  83% '<'  + 11% ','", synthetic("<", ",", fr, rng), "<"),
        ("control    83% 'A'  + 11% ','", synthetic("A", ",", fr, rng), "A"),
        ("control    83% '<'  + 11% 'A'", synthetic("<", "A", fr, rng), "<"),
        ("mirror     83% '>'  + 11% ','", synthetic(">", ",", fr, rng), ">"),
        ("mirror     83% '{'  + 11% '.'", synthetic("{", ".", fr, rng), "{"),
    ]
    print("%d epochs, mutation off, half ring seeded.  "
          "'tracked byte' is the field's majority byte.\n" % args.epochs)
    print("%-38s %10s %10s %10s %8s" % ("field", "random half",
                                        "-> after", "field half", "H bits"))
    for name, field, track in arms:
        s, e, held, h = invade(field, args.epochs, args.seed, track)
        print("%-38s %10.4f %10.4f %10.4f %8.3f%s"
              % (name, s, e, held, h, "   INVADES" if e > 10 * max(s, 1 / 256)
                 else ""))


if __name__ == "__main__":
    main()
