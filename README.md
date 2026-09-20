# soup v0

A minimal artificial-life simulator, built as a research baseline for
open-ended-evolution experiments.

Memory is bytes. There is **no fitness function** and **no notion of an
organism**. Byte strings are executed as programs, the programs can overwrite
themselves and each other, and whatever persists, persists. The only things
imposed from outside are a uniformly random initial condition, a uniform
background mutation rate, and a step budget.

The point of this repository is honest measurement, not features. If nothing
emerges, the results say so.

---

## Install

```sh
pip install numpy numba matplotlib      # pytest as well, to run the tests
```

## Run

```sh
# the validation mode: a population of 64-byte tapes, paired at random
python -m soup.run --config configs/bff.json  --seed 1 --out runs/bff_s1

# the actual baseline: one ring of memory, random execution sites
python -m soup.run --config configs/ring.json --seed 1 --out runs/ring_s1

# every field of the config is also a CLI flag, and CLI wins
python -m soup.run --config configs/ring.json --seed 2 --epochs 5000 \
                   --mu 0.001 --R 64 --out runs/ring_hot
```

Then:

```sh
python -m soup.plots   runs/ring_s1 runs/ring_s2 runs/ring_s3 --out analysis/ring
python -m soup.analyze runs/ring_s1 runs/ring_s2 runs/ring_s3 --json analysis/ring.json
python -m pytest tests/ -q
python -m soup.bench                      # measured interpreter throughput
```

A run is **exactly reproducible from its seed**. The simulator uses its own
xorshift64\* stream rather than numpy's global RNG, so the trajectory does not
depend on the numpy/numba version, on thread count, or on the snapshot
interval; `tests/test_run.py` asserts all of that.

---

## The model

### Instruction set

Memory is a flat array of bytes. These ten byte values are instructions;
**every other byte value is a no-op** (and still costs one step):

| byte | char | effect |
|---|---|---|
| 60 | `<` | head0 -= 1 |
| 62 | `>` | head0 += 1 |
| 123 | `{` | head1 -= 1 |
| 125 | `}` | head1 += 1 |
| 45 | `-` | `mem[head0] -= 1` (mod 256) |
| 43 | `+` | `mem[head0] += 1` (mod 256) |
| 46 | `.` | `mem[head1] = mem[head0]` |
| 44 | `,` | `mem[head0] = mem[head1]` |
| 91 | `[` | if `mem[head0] == 0`, jump forward to the matching `]` |
| 93 | `]` | if `mem[head0] != 0`, jump back to the matching `[` |

Execution of one *run* ends when, and only when:

1. the step budget `k` is exhausted (default 8192),
2. the instruction pointer leaves the executable region, or
3. a bracket has no match inside the region.

Bracket matching is computed **at execution time** by scanning the region, so a
program that rewrites a bracket changes the control flow it is running under.
Code and data are the same memory throughout.

**Head confinement.** head0 and head1 wrap modulo the region length. A head
move is therefore never a termination condition, which is what keeps the list
above at exactly three items. This is an interpretation of "heads are confined
to within ±R of p"; the alternative readings (clamp at the boundary, or
terminate on out-of-bounds) would give a different interpreter, and it matters
— see *Known limitations*.

### Mode `bff` — validation

A population of `N` tapes of 64 bytes. Each epoch draws a uniformly random
perfect matching of the tapes; each pair is concatenated into a 128-byte
region, run from `ip = head0 = head1 = 0`, and split back into two tapes. This
is the setup of Agüera y Arcas et al. (2024), *Computational Life*, where
self-replicators are reported to emerge and take over. It exists here to
validate the interpreter and the metrics against a known result.

### Mode `ring` — the baseline

One ring of `M` bytes (default 65536) with wraparound. One *tick* picks a
uniformly random position `p` and runs from `ip = head0 = head1 = p`, with the
instruction pointer and both heads confined to the `2R+1` window centred on `p`
(default `R = 128`). One epoch is `M/64` ticks, run sequentially. The window is
copied out of the ring, executed, and written back, so a tick sees every
earlier tick's writes.

### Both modes

Memory is initialised with uniform random bytes. After each epoch every byte is
flipped to a uniformly random byte with probability `mu` (default 0.00024, the
rate used in the reference paper). Sampling is done with geometric gaps rather
than one variate per byte; `tests/test_rng.py` checks the realised rate.

---

## Metrics

Written to `metrics.csv` every `snapshot_interval` epochs, computed over all of
memory.

| column | meaning |
|---|---|
| `entropy_bits` | Shannon entropy of the byte distribution, bits/byte (8.0 = uniform, 0.0 = one value) |
| `zlib_bits` | zlib level-9 compressed size, bits/byte |
| `high_order_entropy` | `entropy_bits - zlib_bits` |
| `mean_steps` | mean steps executed per run, since the previous snapshot |
| `frac_copy_runs` | fraction of runs that executed at least one `.` or `,`, since the previous snapshot |
| `frac_term_*` | how runs ended: budget / off-region / unmatched-bracket |
| `distinct_windows` | number of distinct 8-byte windows currently in memory |
| `n_persistent` | 8-byte windows that are persistent *right now* |
| `A_t` | **A(t)**: distinct 8-byte windows that have *ever* been persistent |
| `runs`, `steps`, `mutations` | raw counters since the previous snapshot |

**High-order entropy** is the headline. For i.i.d. bytes it is ~0: the order-0
entropy and the compressed size agree, because there is no structure beyond the
byte histogram for zlib to find. It rises when the *same substrings* recur, so
a sharp rise is the signature of a replicator takeover. The planted-replicator
control in `runs/bff_plant20_s1` takes it from 0.03 to 5.7 bits/byte in under
30 epochs — that is what a takeover looks like in this metric.

**A(t)** is a cumulative novelty count. At each snapshot every 8-byte window in
memory is counted; a window becomes *persistent* once its count has been at
least `c_min` (default 8) for `tau` (default 5) **consecutive snapshots**. A(t)
is the number of distinct windows that have ever crossed that bar, so it only
grows. A soup that keeps inventing new durable patterns has an A(t) that keeps
climbing; one that has settled has an A(t) that flattens.

**`patterns.log`** records the ten most frequent 16-byte windows at every
snapshot, in printable form *and* in hex, so you can read what evolved without
having to guess whether a `.` is the copy instruction or an unprintable byte.

---

## Outputs

Each run directory contains:

```
config.json        the effective configuration, after CLI overrides
metrics.csv        one row per snapshot
patterns.log       top-10 16-byte windows per snapshot
kymograph.npy      (n_snapshots, kymo_width) uint8 -- a fixed slice of memory over time
snapshots/         raw memory dumps, epoch_%08d.npy
summary.json       wall-clock, throughput, platform
```

`python -m soup.plots` turns those into `high_order_entropy.png`, `a_t.png`,
`mean_steps.png`, `frac_copy_runs.png`, `byte_composition.png` (small multiples
of each instruction byte's abundance) and a kymograph per run — a 4096-byte
slice of memory as pixel rows over time, coloured by instruction family (copy /
loop / head-and-arithmetic) against a recessive grey ramp for no-op bytes, plus
a `_raw` version at exactly one pixel per byte.

Figures are rendered on a light surface; they are ordinary research PNGs, not
theme-aware web charts.

---

## Performance

The interpreter is a numba kernel (`soup/interp.py`); orchestration and all
analysis are plain Python/numpy. Measured numbers are in `RESULTS.md` and in
each run's `summary.json`; `python -m soup.bench` reproduces them. The target
was 50k epochs of the default ring config in under an hour on a laptop — the
measured figure is about two and a half minutes.

---

## Known limitations

- **Head confinement is an interpretation.** Heads wrap modulo the region.
  Under a "terminate on out-of-bounds head" rule the ring's dominant structure
  (see `RESULTS.md`) could not exist in the same form, because it depends on
  head0 walking backwards past the start of the window. This is the single
  assumption most likely to change the results.
- **High-order entropy does not see a monoculture.** `H - zlib` is near zero
  both for i.i.d. random bytes *and* for memory that has collapsed to one
  repeated byte, because in the second case the order-0 entropy has already
  fallen to meet the compressed size. Always read `entropy_bits` next to it.
  This is not hypothetical: it is what the ring runs do.
- **zlib has a 32 KiB window.** For a 64 KiB ring it cannot exploit
  correlations at range beyond half the memory, so `zlib_bits` is an upper
  bound on the compressed size and `high_order_entropy` a lower bound.
- **A(t)'s `tau` is counted in snapshots, not epochs.** Runs with different
  `snapshot_interval` values are not directly comparable on A(t).
- **A(t) counts 8-byte windows, not organisms.** It cannot distinguish "one
  new 40-byte replicator" from "33 new windows"; a single new structure of
  length L adds up to L+7 to A(t) at once.
- **One mutation model.** Point substitution only — no insertion, deletion,
  duplication or recombination, so there is no cheap route to longer patterns.
- **bff and ring are not directly comparable.** They differ in topology,
  execution-site distribution and locality all at once; the default configs
  only match on total memory size (65536 bytes) and on `k` and `mu`.
- **Single-threaded.** Ticks are sequential by definition of the model; runs
  are parallelised across seeds by running separate processes.
