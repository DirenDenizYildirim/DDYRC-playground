"""Measured interpreter throughput.  python -m soup.bench"""
import time

import numpy as np

from . import core
from .config import Config
from .run import Soup


def bench(cfg, epochs, warm=2):
    soup = Soup(cfg)
    soup.advance(warm)                      # pay for JIT compilation first
    soup.stats[:] = 0
    t0 = time.perf_counter()
    soup.advance(epochs)
    dt = time.perf_counter() - t0
    runs = int(soup.stats[core.STAT_RUNS])
    steps = int(soup.stats[core.STAT_STEPS])
    return dict(epochs=epochs, seconds=dt, epochs_per_s=epochs / dt,
                runs_per_s=runs / dt, steps_per_s=steps / dt,
                mean_steps=steps / runs)


def main():
    for name, cfg, n in (
        ("ring M=65536 R=128 k=8192 (random soup)",
         Config(mode="ring", seed=1), 200),
        ("bff  N=1024  k=8192       (random soup)",
         Config(mode="bff", seed=1), 400),
    ):
        r = bench(cfg, n)
        print("%s\n  %7.1f epochs/s  %9.0f runs/s  %11.3g steps/s  "
              "(mean %5.0f steps/run)\n  -> 50k epochs in %.1f min"
              % (name, r["epochs_per_s"], r["runs_per_s"], r["steps_per_s"],
                 r["mean_steps"], 50000 / r["epochs_per_s"] / 60))


if __name__ == "__main__":
    main()
