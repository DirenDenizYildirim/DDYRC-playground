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
pip install numpy numba matplotlib brotli zstandard   # pytest to run the tests
```

## Run

```sh
# the validation mode: a population of 64-byte tapes, paired at random
python -m soup.run --config configs/bff.json  --seed 1 --out runs/bff_s1

# the actual baseline: one ring of memory, random execution sites
python -m soup.run --config configs/ring.json --seed 1 --out runs/ring_s1

# bff's pairing with ring locality: blocks pair with near neighbours only
python -m soup.run --config configs/blocks.json --seed 1 --out runs/blocks_s1

# bit-exact cubff, the implementation the reference paper used
python -m soup.run --config configs/bff_compat.json --compat cubff --seed 1 \
                   --out runs/compat_s1

# language variants
python -m soup.run --config configs/ring.json --seed 1 --no-copy 1 --out runs/nocopy_s1
python -m soup.run --config configs/ring.json --seed 1 --indel 1   --out runs/indel_s1

# resume a saved soup and carry provenance labels
python -m soup.run --config configs/plateau.json --seed 12 --labels 1 \
                   --load runs/compat_cubff_s12/snapshots/epoch_00005500.npy \
                   --start-epoch 5500 --out runs/plateau_s12

# cubff's head rule applied to the other modes
python -m soup.run --config configs/blocks_cubffhead.json --d 8 --seed 1 \
                   --out runs/blocks_ch_d8_s1
python -m soup.run --config configs/ring_datahead.json --seed 1 \
                   --out runs/ring_datahead_s1

# every field of the config is also a CLI flag, and CLI wins
python -m soup.run --config configs/ring.json --seed 2 --epochs 5000 \
                   --mu 0.001 --R 64 --out runs/ring_hot
```

Then:

```sh
python -m soup.plots   runs/ring_s1 runs/ring_s2 runs/ring_s3 --out analysis/ring
python -m soup.analyze runs/ring_s1 runs/ring_s2 runs/ring_s3 --json analysis/ring.json
python -m soup.rescore runs/ring_s1       # recompute metrics for an old run
python -m pytest tests/ -q
python -m soup.bench                      # measured interpreter throughput

# post-takeover analysis (see "Does anything keep happening?" below)
python -m soup.assay   --a later.npy --b earlier.npy --out analysis/a_vs_b
python -m soup.motifs  runs/plateau_s12 --out analysis/motifs/s12
python -m soup.posthoc runs/plateau_s12 --out analysis/posthoc/s12
python -m experiments.phase2 --jobs 4     # the whole queue, resumable
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
to within ±R of p"; `--head-bound halt` selects the other one, where a head
move that would leave the region ends the run instead. Both readings share the
same interpreter, and the ring results are the same under either (RESULTS.md).

### Mode `bff` — validation



A population of `N` tapes of 64 bytes. Each epoch draws a uniformly random
perfect matching of the tapes; each pair is concatenated into a 128-byte
region, run from `ip = head0 = head1 = 0`, and split back into two tapes. This
is the setup of Agüera y Arcas et al. (2024), *Computational Life*, where
self-replicators are reported to emerge and take over. It exists here to
validate the interpreter and the metrics against a known result.

`--compat cubff` and `--compat cubff_noheads` replace our RNG, pairing and
mutation with [cubff](https://github.com/paradigms-of-intelligence/cubff)'s,
so a run can be diffed byte-for-byte against a cubff checkpoint — which
`tests/test_compat.py` does. cubff's `bff_noheads` is the spec above;
its `bff` starts with `head0 = tape[0] % 128`, `head1 = tape[1] % 128` and
`pc = 2`, which is a materially different language. RESULTS.md has the full
difference table.

### Mode `ring` — the baseline

One ring of `M` bytes (default 65536) with wraparound. One *tick* picks a
uniformly random position `p` and runs from `ip = head0 = head1 = p`, with the
instruction pointer and both heads confined to the `2R+1` window centred on `p`
(default `R = 128`). One epoch is `M/64` ticks, run sequentially. The window is
copied out of the ring, executed, and written back, so a tick sees every
earlier tick's writes.

### Mode `blocks` — bff's pairing with the ring's locality

A ring of `N` blocks of 64 bytes. A tick picks a random block `i` and a partner
drawn uniformly from the `2d` blocks within `±d` of it (wrapping, `i` itself
excluded), concatenates the two in a random order into a 128-byte region, runs
it bff-style from `ip = head0 = head1 = 0`, and writes both back. One epoch is
`N/2` ticks. `d` interpolates between bff (`d = N/2`, any partner) and a purely
local soup (`d = 1`).

### Language variants

`--no-copy 1` turns `.` and `,` into no-ops: they still cost a step and still
count as instructions, they just move no bytes. `--indel 1` adds insertions and
deletions to the mutation operator at the same *total* rate as substitution
(so half that rate each); both shift bytes only within `R` of the site and both
inject exactly one random byte, so an indel costs the same entropy as a
substitution and only the reading frame differs.

### Both modes

Memory is initialised with uniform random bytes. After each epoch every byte is
flipped to a uniformly random byte with probability `mu` (default 0.00024, the
rate used in the reference paper). Sampling is done with geometric gaps rather
than one variate per byte; `tests/test_rng.py` checks the realised rate.

### Provenance labels

`--labels 1` allocates a second byte array the same shape as memory. `.` and
`,` copy a byte's label along with the byte; every other instruction, and
mutation, leave the destination cell's label untouched. **No instruction can
read a label**, so a labelled run is byte-for-byte the run without labels —
`tests/test_labels.py` asserts that memory, every counter and the whole metric
row are identical with labels on and off. Labels start as one lineage id per
tape (`i % 251`), and are dumped beside every tape dump.

### Resuming

`--load <dump.npy> --start-epoch <n>` starts from a saved soup instead of a
random initial condition. Under `--compat cubff` the RNG streams are keyed on
the epoch, so a resume reproduces an uninterrupted run exactly; the test for
that is in `tests/test_labels.py`.

---

## Does anything keep happening?

Three tools answer that, and all of them work on saved soups rather than on
a live run.

**`soup.assay` — competition.** Build one population from half of soup A and
half of soup B, label the halves, and run the normal dynamics at the normal
mutation rate. The curve is the fraction of memory still carrying A's label.
A against itself is the neutral control, and its spread across replicates is
the noise band any real difference has to leave.

**`soup.motifs` — families and sweeps.** The dominant 16-byte windows of a
taken-over soup are mostly shifts of one string, so counting windows
overstates how many distinct things are present. Windows within edit distance
`MAX_DIST` are one family; a family's *coverage* is the fraction of tapes
carrying it. A **sweep** is a family passing 10% coverage while the family
that was leading falls below 5%. `MAX_DIST = 8` is set from
`motifs.null_distances()`: over 2000 pairs drawn from a soup's own byte
frequencies the smallest distance observed was 12.

**`soup.posthoc` — self-sufficiency and where the variation sits.** A tape
from the soup is paired with a *fresh uniform-random* tape and run once;
*colonisation* is the share of the resulting 128 bytes carrying the soup
tape's label, which is 0.5 when neither side gains. The controls are
soup-vs-soup and random-vs-random. For the second question, tapes are aligned
on the leading family's representative window and per-offset byte entropy is
reported — offsets 0..15 are the seed window, so low entropy there is
circular and is shown only for completeness; the conserved width *outside*
that range is the measurement.

---

## Metrics

Written to `metrics.csv` every `snapshot_interval` epochs, computed over all of
memory.

| column | meaning |
|---|---|
| `entropy_bits` | order-0 Shannon entropy **H**, bits/byte (8.0 = uniform, 0.0 = one value) |
| `comp_bits` | compressed size, bits/byte — brotli quality 6, window 2^24 |
| `high_order_entropy` | `entropy_bits - comp_bits` |
| `brotli2_bits` | brotli quality 2, the setting cubff reports, for comparability |
| `zlib_bits`, `hoe_zlib` | zlib level 9, kept so the earlier runs stay comparable |
| `mean_steps` | mean steps executed per run, since the previous snapshot |
| `frac_steps_in_loop` | fraction of steps executed at an instruction pointer the run had already visited — i.e. time spent re-running code |
| `ops_per_run` | executed instructions per run, excluding no-ops (cubff's "ops") |
| `frac_copy_runs` | fraction of runs that executed at least one `.` or `,` |
| `frac_term_*` | how runs ended: budget / off-region / unmatched-bracket |
| `distinct_windows` | number of distinct 8-byte windows currently in memory |
| `n_persistent`, `A_t` | **A(t)**: windows that are persistent now / have ever been |
| `n_persistent_naive`, `A_t_naive` | the same without the null filter |
| `runs`, `steps`, `mutations` | raw counters since the previous snapshot |

**High-order entropy** is the headline, and it is never reported alone. For
i.i.d. bytes it is ~0: the order-0 entropy and the compressed size agree,
because there is no structure beyond the byte histogram to find. It rises when
the *same substrings* recur, so a sharp rise is the signature of a replicator
takeover — the planted-replicator control in `runs/bff_plant20_s1` takes it
from 0.03 to 5.5 bits/byte in under 30 epochs. But it is *also* ~0 for memory
that has collapsed to one repeated byte, because H has fallen to meet the
compressed size, so **H is always logged beside it**: 8.0 means random, ~1.3
means monoculture, and high-order entropy alone cannot tell them apart.

**A(t)** is a cumulative novelty count. At each snapshot every 8-byte window in
memory is counted. A window *qualifies* when its count is at least `c_min`
(default 8) **and** at least `null_ratio` (default 5) times the count an
i.i.d. model with the soup's *current* byte frequencies would predict. It
becomes *persistent* once it has qualified continuously for `tau_epochs`
(default 250) **epochs** — epochs, not snapshots, so the logging cadence does
not change what A(t) counts. A(t) is the number of distinct windows that have
ever crossed that bar, so it only grows.

Both halves of that definition are load-bearing and both were added after the
first version misled us. Counting `tau` in snapshots made A(t) depend on the
logging interval (the same configuration scored 547 at one cadence and 262 at
another). Omitting the null filter made a soup that is 82% one byte look
endlessly creative, because a two-letter alphabet has thousands of
arrangements that clear `c_min` by chance. `A_t_naive` keeps the unfiltered
count so the two can be compared.

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
                   and labels_%08d.npy when --labels 1
summary.json       wall-clock, throughput, platform
```

`soup.assay` writes `<out>.csv` (one row per replicate and assay epoch) beside
`<out>.json` with the exact inputs. `soup.motifs` writes family coverage per
snapshot plus a `.json` listing the sweeps it found. `soup.posthoc` writes one
row per snapshot plus `_profiles.json` with the per-offset entropy profiles.
`soup.figures` draws all three.

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

- **Head confinement is an interpretation.** Heads wrap modulo the region by
  default. The other reading is available as `--head-bound halt` and was
  measured rather than assumed: the ring runs reach the same state at the same
  time under either rule (`RESULTS.md`), so this assumption turns out not to be
  load-bearing there. It has not been tested at length in bff mode, and the
  reference implementation's rule is not stated in the material available here.
- **High-order entropy does not see a monoculture.** `H - zlib` is near zero
  both for i.i.d. random bytes *and* for memory that has collapsed to one
  repeated byte, because in the second case the order-0 entropy has already
  fallen to meet the compressed size. Always read `entropy_bits` next to it.
  This is not hypothetical: it is what the ring runs do.
- **zlib has a 32 KiB window.** For a 64 KiB ring it cannot exploit
  correlations at range beyond half the memory, so `zlib_bits` is an upper
  bound on the compressed size and `high_order_entropy` a lower bound.
- **A(t) counts 8-byte windows, not organisms.** It cannot distinguish "one
  new 40-byte replicator" from "33 new windows"; a single new structure of
  length L adds up to L+7 to A(t) at once.
- **A(t)'s null filter is i.i.d.** It knows the soup's byte frequencies but
  not its spatial structure, so a field of clustered `,` runs still scores as
  novel — correctly, since the clustering *is* non-i.i.d. structure, but it is
  not a program either.
- **`frac_steps_in_loop` counts re-executed addresses, not syntactic loops.**
  That is deliberate — it catches loops built by self-modification, which a
  bracket-matching definition would miss — but a program that legitimately
  revisits an address without looping would be counted too.
- **Mutation is point substitution by default.** `--indel 1` adds insertions
  and deletions; there is still no duplication or recombination, so there is
  no cheap route to *longer* patterns.
- **bff and ring are not directly comparable.** They differ in topology,
  execution-site distribution and locality all at once; the default configs
  only match on total memory size (65536 bytes) and on `k` and `mu`.
  `--mode blocks` exists to separate those factors: it shares bff's pairing
  and differs only in locality.
- **Single-threaded.** Ticks are sequential by definition of the model; runs
  are parallelised across seeds by running separate processes.
