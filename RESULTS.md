# Results

Runs: the two specified experiments — `ring` for 50k epochs at three seeds and
`bff` at three seeds — plus a planted-replicator control, a head-confinement
sensitivity set, and four further `bff` configurations run to chase down the
`bff` result. Every number below comes from a committed run directory;
`python -m soup.analyze runs/...` regenerates the tables and
`python -m soup.plots runs/...` the figures.

---

## The four questions, answered first

**Did a replicator transition occur?**

* **ring — yes, in all three seeds, and it is trivial.** Memory converges to a
  field built from two byte values: `<` at ~82% and `,` at ~11% (seed 2 found
  the mirror-image pair `{` and `.` instead, and ended up with both). Order-0
  entropy falls from 8.0 to ~1.3 bits/byte. A separate invasion experiment
  confirms this field is genuinely self-propagating and not merely an
  absorbing state: seeded into half a ring of fresh random bytes with mutation
  off, it spreads.
* **bff — no, in any of ten runs, including three at a matched interaction
  budget.** Nothing took over at N=1024 (3 seeds × 20k epochs), N=8192
  (3 seeds × 20k), N=1024 × 200k epochs, or N=32768 × 64k epochs
  (3 seeds, 1.05e9 pairwise interactions each — the budget at which the
  reference paper reports a 40% transition rate). The interpreter and the
  metrics are separately verified to detect a takeover when one happens, so
  this is a real negative, but at 3 seeds against a 40% rate it is also
  **not a failed validation**: three misses have probability 0.6³ ≈ 0.22 even
  if the process here matched the reference exactly. The honest verdict is
  that the published emergence result was **not reproduced, and not
  contradicted either**. See below for what would settle it.

**At what epoch?** In the ring, high-order entropy crosses 0.5 bits/byte at
epoch 250–300 and 1.0 at epoch 850–1450; order-0 entropy has fallen below 4.0
bits/byte by epoch 2100–2700. The whole transition takes about 2000 of 50000
epochs, i.e. the first 4% of the run.

**What do the dominant patterns look like?** Runs of `<` interrupted by an
occasional `,`. The top ten 16-byte windows at the end of `ring_s1` are
`<<<<<<<<<<<<<<<<` (20630 occurrences), then `,,,,,,,,,,,,,,,,` (539), then
single-`,` substitutions of the all-`<` window at every offset (490–526 each).
That is the whole vocabulary.

**Does A(t) plateau afterwards?** **No — and that is a problem with A(t), not a
sign of ongoing novelty.** A(t) is still climbing at epoch 50000 — decelerating
(72 new windows per 1000 epochs over the first half, 16 over the second) but
nowhere near flat, long after the soup has stopped doing anything new. Of the
448 windows abundant (count >= `c_min`) at the final snapshot of `ring_s1`,
57% are built *only* from the four bytes `< , { .`; most of the rest add one
byte adjacent in value to `<` (`;` = 0x3b, `=` = 0x3d). A(t) is counting the
combinatorics of a two-letter alphabet, not new structure. Details below.

---

## What was validated, and how

Nothing here rests on the interpreter being right by assertion.

| check | result |
|---|---|
| 164 unit tests | every instruction, bracket depth-matching in both directions, all three termination conditions, head confinement, no-op behaviour for all 246 non-instruction byte values, self-modification |
| compiled kernel vs. a pure-Python reference | identical `(steps, copies, termination)` and identical final memory on 80 random programs, under both head-confinement rules |
| hand-written replicator (`soup/replicator.py`) | copies its own 25 bytes **exactly**, counter cell included, in a bare region, in bff mode, in ring mode, and across the ring's wraparound seam; the copy replicates in turn |
| reproducibility | same seed → identical memory; independent of how epochs are chunked between snapshots |
| mutation operator | realised rate matches `mu` within 5σ; positions spread across the whole array |

**Do the metrics detect a takeover?** `runs/bff_plant20_s1` plants 20 copies of
the hand-written replicator in an otherwise standard N=1024 bff soup. Within 30
epochs:

| | epoch 0 | epoch 30 | epoch 400 |
|---|---|---|---|
| order-0 entropy | 7.995 | 6.157 | 5.883 |
| zlib bits/byte | 7.969 | 0.630 | 0.335 |
| **high-order entropy** | **0.025** | **5.527** | **5.548** |
| mean steps per run | — | 4815 | 8072 (budget is 8192) |
| fraction of runs copying | — | 0.853 | 0.991 |

and `patterns.log` fills with the replicator's own bytes
(`}}}}]++++++++[.>`, 1007 occurrences in a 1024-tape soup). That is what a
takeover looks like in every metric at once. Nothing resembling it appears in
any unplanted run.

A single planted founder usually dies: across 8 seeds, 5 went extinct within
two epochs and 1 took over. The growth factor is only ~1.5 per epoch (the
program copies to +64, so it replicates only when the shuffle puts it in the
first half of the pair), so founder stochasticity dominates. This matters for
reading the bff result.

---

## cubff: every semantic difference, and the one that matters

`--compat cubff` and `--compat cubff_noheads` reproduce
[cubff](https://github.com/paradigms-of-intelligence/cubff) (commit f212e849,
built `make CUDA=0`) **bit-exactly**: same SplitMix64 streams for
initialisation, pairing and mutation, same order of operations. After 129
epochs on 256 tapes our soup equals a cubff checkpoint byte for byte, for both
languages and two seeds; `tests/test_compat.py` asserts it against checkpoints
cubff wrote, committed under `tests/data/`. Longer manual checks match at 513
epochs on 1024 tapes.

cubff ships two BFF languages. The spec this repository implements — "run from
IP=0 with head0=head1=0" — is `bff_noheads`. The one cubff *names* `bff`, built
with `BFF_HEADS` defined, is a different language.

| aspect | this repo's spec | cubff `bff` | cubff `bff_noheads` |
|---|---|---|---|
| **initial head0, head1** | **0, 0** | **`tape[0] % 128`, `tape[1] % 128`** | 0, 0 |
| **initial pc** | **0** | **2** | 0 |
| head leaving the tape | wraps modulo the region | wraps modulo 128 (mask applied at the top of each step) | same |
| `[` condition / target | `tape[head0] == 0`; jump to the matching `]`, then +1 | identical | identical |
| `]` condition / target | `tape[head0] != 0`; jump to the matching `[`, then +1 | identical | identical |
| unmatched `[` | run ends, the `[` costs one step | pc set past the end, ends the same way, same step count | same |
| unmatched `]` | run ends, the `]` costs one step | pc set to −1, ends the same way, same step count | same |
| step budget | `k`, counts no-ops (default 8192) | 8192, counts no-ops | same |
| "ops" it reports | steps including no-ops (we now also log `ops_per_run`, excluding them) | `i − nskip`: excludes no-ops | same |
| tape length | 64 | 64 | 64 |
| default soup size | `N`, our default 1024 | 131072 = 2^17 | same |
| pairing | Fisher-Yates over all N, adjacent pairs | identity, then a descending Fisher-Yates seeded `SplitMix64(seed(epoch·N+i))`, adjacent pairs | same |
| mutation timing | after each epoch, over the whole soup | on each pair's 128-byte tape immediately before it runs | same |
| mutation rate | `mu` per byte per epoch, default 0.00024 | `1<<18` over a `1<<30` denominator = 2^−12 = 0.000244140625 | same |
| initial bytes | our xorshift64\* stream | `SplitMix64(64·N·seed + 64·i + j) % 256` | same |
| compression metric | zlib level 9 (now brotli q6, window 2^24) | brotli quality 2, window 2^24 | same |

Everything below the first two rows is either identical or scaffolding that
does not change the language — different RNG streams, a mutation applied a
few hundred microseconds earlier, a counter that reports a different thing.
Our `mu = 0.00024` is cubff's own displayed default, which is 2^−12 rounded;
the compat configs use the exact value.

**The one behavioural difference is the initial machine state**, and it is
large. In `bff`, the first two bytes of the concatenated pair *are* the head
positions, and execution starts at byte 2. A program therefore gets an
arbitrary head0/head1 offset for free. That offset is precisely what a
replicator needs and precisely what is expensive to compute: the hand-written
replicator in this repository spends 20 of its 25 bytes on a counter loop and
a counter-restoring `++++++++` run whose only job is to put head1 64 cells
away from head0. Under `bff` that entire apparatus is two data bytes.

Two immediate consequences, both visible in the runs: `bff` executes more —
1120 mean steps per run against 640 for our spec at the same soup size — and
its runs write more often.

---

## bff: no spontaneous takeover in ten runs

| run | N | epochs | interactions | HOE final | HOE max | H final | A(t) | takeover |
|---|---|---|---|---|---|---|---|---|
| bff_s1 | 1024 | 20000 | 1.0e7 | 0.214 | 0.529 | 7.517 | 547 | no |
| bff_s2 | 1024 | 20000 | 1.0e7 | 0.226 | 0.600 | 7.397 | 499 | no |
| bff_s3 | 1024 | 20000 | 1.0e7 | 0.242 | 0.575 | 7.392 | 535 | no |
| bff_n8192_s1 | 8192 | 20000 | 8.2e7 | 0.341 | 0.592 | 7.509 | 2229 | no |
| bff_n8192_s2 | 8192 | 20000 | 8.2e7 | 0.320 | 0.576 | 7.514 | 2230 | no |
| bff_n8192_s3 | 8192 | 20000 | 8.2e7 | 0.320 | 0.597 | 7.527 | 2410 | no |
| bff_long_s1 | 1024 | 200000 | 1.0e8 | 0.466 | 0.694 | 7.490 | 262 | no |
| bff_n32768_s1 | 32768 | 64000 | **1.05e9** | 0.292 | 0.589 | 7.487 | 569 | no |
| bff_n32768_s2 | 32768 | 64000 | **1.05e9** | 0.335 | 0.564 | 7.510 | 560 | no |
| bff_n32768_s3 | 32768 | 64000 | **1.05e9** | 0.290 | 0.586 | 7.526 | 556 | no |

The reference result (Agüera y Arcas et al. 2024) uses a soup of **2^17 = 131072
tapes** and reports that **40% of runs show a state transition within 16k
epochs** — i.e. even at full scale, most runs do not transition in that window.
16k epochs at 2^17 tapes is about **1.1e9 pairwise interactions**. The first
seven rows of the table fall 11× to 110× short of that budget, so finding no
transition in them is the outcome the published numbers predict rather than
evidence of a broken interpreter.

The last three rows close that gap: N=32768 for 64000 epochs is 1.05e9
interactions, matching the reference budget to within 5%. All three seeds ran
to completion (44 minutes each) with no transition — peak high-order entropy
0.56–0.59 against the 5.5 a verified takeover produces, order-0 entropy still
7.49–7.53, and the ten most frequent 16-byte windows at the end are
constant-byte runs (`>>>>>>>>>>>>>>>>`, 591 occurrences in a 2 MiB soup, then
`;;;;;;;;;;;;;;;;`, `::::::::::::::::` — the ±1 neighbours of `<`), not program
text.

Three seeds against a reported 40% per-run rate is weak evidence either way:
P(0 of 3) = 0.22 under the reference's own numbers. So this does **not**
establish a discrepancy, and it does not establish agreement. What it does
establish is that the substrate is not inert — a planted replicator fixates
here in ~20 epochs and every metric screams — so the open question is about
the *rate of spontaneous origination*, not about whether replication works.

Two caveats on the matching, stated because they are not controlled for:

* The budget was matched on **interactions**, not on soup size. The reference
  uses 2^17 tapes for 16k epochs; these runs use 2^15 tapes for 64k epochs.
  If the hazard of a replicator arising scales with interactions, the two are
  equivalent; if it scales with the number of independent *sites*, these runs
  have 4× fewer chances. (Each tape here accumulates 4× more personal history,
  which should if anything help, but that is an argument, not a measurement.)
* The reference's rule for a head moving out of bounds is not stated in the
  material available here, and bff was not tested under both readings at
  length. In ring mode the two readings agree; in bff they might not.

Settling it properly needs either a soup of 2^17 tapes — about 3.5 hours per
seed at this throughput once the pre-transition phase alone is counted, and far
longer if a transition occurs and every run starts using the full 8192-step
budget — or ~10 seeds at the present scale to put a useful confidence interval
on a 40% rate. Neither fits the time available here, and neither should be
skipped before claiming the reference result reproduces.

What the bff soups *do* show is the same chemistry the ring runs push to
completion, just never reaching fixation: `<` is enriched to ~7% of memory
against a random-soup level of 1/256 = 0.39% (18×) and `,` to ~2.5% (6×), while
`+`, `-` and `]` stay at the random level. See
`analysis/bff/byte_composition.png`. This is a stationary state, not a slow
climb — across all 33 memory dumps from epoch 40000 onwards `<` stays within
0.039–0.079 and `,` within 0.011–0.050, with no trend (every 40000th dump
shown):

| epoch | 0 | 40000 | 80000 | 120000 | 160000 | 200000 |
|---|---|---|---|---|---|---|
| `<` | 0.0035 | 0.0718 | 0.0698 | 0.0679 | 0.0509 | 0.0581 |
| `,` | 0.0039 | 0.0328 | 0.0222 | 0.0468 | 0.0329 | 0.0307 |

Why it stalls in bff and fixes in the ring was not established. Two
differences are candidates and neither was tested: the ring's copied value
`memory[p]` is a uniform sample of all memory, whereas bff's is always byte 0
of whichever tape the shuffle put first; and bff re-pairs every tape every
epoch, so no two tapes stay adjacent.

---

## ring: a real transition, to something trivial

| seed | HOE > 0.5 | HOE ≥ 1.0 | H ≤ 4.0 | H final | zlib final | HOE final | mean steps | copy frac | A(t) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 300 | 1450 | 2500 | 1.279 | 1.245 | 0.035 | 128.2 | 0.817 | 2189 |
| 2 | 250 | 850 | 2700 | 2.120 | 1.127 | 0.994 | 128.8 | 0.855 | 3077 |
| 3 | 250 | 850 | 2100 | 1.280 | 1.235 | 0.045 | 128.8 | 0.839 | 2233 |

Final composition of memory (fraction of all 65536 bytes):

| seed | `<` | `{` | `,` | `.` | everything else |
|---|---|---|---|---|---|
| 1 | 0.821 | — | 0.109 | — | below 1/256 |
| 2 | 0.348 | 0.480 | 0.050 | 0.063 | below 1/256 |
| 3 | 0.814 | — | 0.118 | — | below 1/256 |

`>`, `}`, `-`, `+`, `[` and `]` all end up **below** the 1/256 level they
started at: they are actively purged. Mean steps per run settles at 128.2–128.8
against a budget of 8192, and 98.7% of runs end by the instruction pointer
running off the window (1.3% on an unmatched bracket, 0.0% on the step budget).
129 steps is exactly a straight walk from `p` to the far edge: with the
brackets gone there are no loops left to run, and the soup has become quiet.

It is a patchwork rather than a uniform field. In final `ring_s1` memory the
`<` runs average 8.7 bytes (longest 261, wider than the 257-byte window itself)
and the `,` runs average 1.9 bytes (longest 89) — the solid-`,` blocks are what
put `,,,,,,,,,,,,,,,,` second in the pattern log. That granularity is what the
copy-fraction metric is measuring: 18.0% of 129-byte windows in the final
memory contain no `,` at all, against a measured 18.3% of runs that executed no
copy. The two numbers agreeing is a consistency check on both.

**Is it a replicator?** `experiments/invasion.py` seeds half a ring with a
candidate field and half with fresh uniform random bytes, switches mutation
off, and runs 400 epochs. Tracked value is the field's majority byte, measured
in the half that started random:

| field | random half, before | after | field half | verdict |
|---|---|---|---|---|
| evolved `ring_s1` memory | 0.0038 | 0.0489 | 0.851 | spreads |
| synthetic 83% `<` + 11% `,` | 0.0038 | 0.0631 | 0.828 | spreads |
| 83% `A` (no-op) + 11% `,` | 0.0041 | 0.0042 | 0.492 | does not spread, and decays |
| 83% `<` + 11% `A` (no-op) | 0.0038 | 0.0216 | 0.798 | spreads ~3× slower |
| 83% `>` + 11% `,` | 0.0042 | 0.0089 | 0.927 | barely spreads (13× slower) |
| 83% `{` + 11% `.` | 0.0037 | 0.0487 | 0.857 | spreads |

So the state is genuinely self-propagating, two byte values are enough to
reproduce the entire phenomenon, and both ingredients are needed: replacing the
copier with a no-op kills it outright, and replacing the head-mover with a
no-op leaves only the residual copiers already present in the random half.

**The mechanism, as far as the data supports it.** A tick starts with
`head1 = p` and never moves it, because `{` and `}` have been purged. So every
`,` executed writes `memory[p]` — a uniform sample of memory, hence almost
always `<` — into `memory[head0]`, and every `<` walks `head0` one cell
*backwards*, behind the instruction pointer. The field therefore paints copies
of its own majority byte over whatever is behind it. The `>` arm of the
experiment shows why the direction matters: `>` walks `head0` forward, into the
code the instruction pointer is about to execute, and destroys the very run
doing the copying — its growth above the starting level is 13× smaller. `{`/`.` is the exact mirror (`.` writes to
`head1`, `{` walks `head1` backwards) and works exactly as well, which is why
seed 2 found it. This is the simplest possible self-replicating structure the
instruction set admits, and it is what an unstructured soup finds first.

Seed 2 is the only run with a sustained high-order entropy (~1.0 at the end):
it ended up with two competing domains, `<`/`,` and `{`/`.`, and the domain
structure is compressible beyond what the byte histogram explains. Seeds 1 and
3 are single-domain, so all their structure is already in the histogram and
high-order entropy falls back to ~0.04. See *What high-order entropy misses*.

`analysis/ring/kymograph_seed*.png` shows the transition as a thin grey band at
the top (random bytes) giving way to a solid field within the first few hundred
epochs. For contrast, `analysis/bff_plant20/kymograph_seed1.png` is what a
*non*-trivial takeover looks like in the same rendering: 25 epochs of grey
noise, then vertical stripes as one 25-byte program tiles the entire soup at a
64-byte pitch.

---

## Sensitivity: does the head-confinement rule cause this?

The spec says heads are "confined to within ±R of p" without saying what
happens at the edge. The default here wraps them. Because the winning field
depends on `head0` walking backwards out of the window, that choice is the one
most likely to be load-bearing, so it was tested rather than argued about:
`--head-bound halt` ends a run when a head would leave the region instead.

The answer is that it is not load-bearing. All three seeds reach the same
state, by the same route, at almost the same epoch, under either rule:

| seed | rule | HOE ≥ 1.0 | H ≤ 4.0 | H final | `<` | `,` |
|---|---|---|---|---|---|---|
| 1 | wrap | 1450 | 2500 | 1.279 | 0.821 | 0.109 |
| 1 | halt | 1400 | 2700 | 1.165 | 0.835 | 0.105 |
| 2 | wrap | 850 | 2700 | 2.120 | 0.348 † | 0.050 † |
| 2 | halt | 1000 | 2800 | 1.245 | 0.814 | 0.125 |
| 3 | wrap | 850 | 2100 | 1.280 | 0.814 | 0.118 |
| 3 | halt | 850 | 2200 | 1.309 | 0.798 | 0.129 |

† wrap seed 2 is the run that also grew a `{`/`.` domain (`{` 0.480,
`.` 0.063); under the halt rule the same seed settled on `<`/`,` alone, so
which of the two mirror pairs wins is a coin flip, not a consequence of the
confinement rule.

`analysis/ring_headbound/high_order_entropy.png` overlays the two: identical
shape, with halt decaying slightly more slowly.

In hindsight the reason is simple, and it is worth stating because the README
originally guessed the opposite. The copying happens *inside* the window —
`head0` has 128 cells of room behind `p` before it reaches an edge — so the
mechanism never depends on a head leaving the region. The halt rule only ends
some runs a little earlier. Heads can outrun the instruction pointer at all
only inside a loop, since otherwise both advance one cell per step from the
same starting point, and loops are exactly what this soup purges.

---

## What the metrics missed, and what changed

Three of the four metrics in the first version of this baseline were
misleading, in ways the runs made obvious. All four were replaced; the old
columns are kept beside the new ones so the change can be audited.

### 1. High-order entropy is blind to a monoculture — so H is always beside it

`H - C` is near zero for i.i.d. random bytes **and** near zero for memory that
has collapsed to one repeated byte, because in the second case the order-0
entropy has already fallen to meet the compressed size.

| state | H | compressed | HOE |
|---|---|---|---|
| random bytes | 7.997 | 8.000 | −0.003 |
| ring crystal (seed 1, epoch 50000) | 1.279 | 1.273 | 0.006 |
| planted replicator at fixation | 5.883 | 0.341 | 5.542 |
| emergent cubff replicator at fixation | 6.578 | 1.523 | 5.749 |

The first two rows are indistinguishable on HOE and could not be more
different. `entropy_bits` is now a required companion column, and the
end-state classifier below uses both.

### 2. zlib's 32 KiB window is too small — now brotli with a 16 MiB window

zlib cannot see a repeat spanning more than half of a 64 KiB ring, so it
over-estimates the code length for exactly the soups that matter.
`comp_bits` is now brotli quality 6 with `lgwin=24`; `brotli2_bits` (cubff's
own setting) and `zlib_bits` are logged alongside.

Quality matters more than expected. At quality 2 — what cubff reports — brotli
compresses the ring crystal to 1.390 bits/byte, *worse* than its 1.279 bits of
order-0 entropy, which would make high-order entropy negative. Quality 6 gives
1.273. Quality 11 gives 1.151 but costs 7.6 s on a 2 MiB soup against 55 ms,
so it is available and not the default.

### 3. A(t)'s `tau` counted snapshots — now epochs

The same bff configuration logged every 20 epochs scored A(t) = 547 and logged
every 200 epochs scored 262. That factor of two came from the logging cadence
and nothing else. `tau_epochs` (default 250) replaces it, and
`tests/test_metrics.py` asserts that the same history sampled at two cadences
gives the same A(t).

### 4. A(t) counted combinatorics — now filtered against an i.i.d. null

A window counts only if its multiplicity is at least `c_min` **and** at least
5× what an i.i.d. model with the soup's *current* byte frequencies predicts.
In a soup that is 82% `<`, a window of eight `<` occurs about 13400 times by
chance; the old A(t) counted every such arrangement as a discovery.

Re-scored with both rules, on each run's stored memory dumps
(`python -m soup.rescore`, `tau_epochs = 1000`, `null_ratio = 5`):

| run | H | brotli6 | HOE | A(t) filtered | A(t) naive | filter cuts |
|---|---|---|---|---|---|---|
| ring_s1 | 1.279 | 1.273 | 0.006 | 220 | 741 | 70% |
| ring_s2 | 2.120 | 1.156 | 0.965 | 740 | 788 | 6% |
| ring_s3 | 1.280 | 1.265 | 0.015 | 250 | 765 | 67% |
| ring_halt_s1 | 1.165 | 1.168 | −0.003 | 300 | 896 | 67% |
| ring_halt_s2 | 1.245 | 1.219 | 0.027 | 491 | 845 | 42% |
| ring_halt_s3 | 1.308 | 1.262 | 0.047 | 224 | 773 | 71% |
| bff_s1 | 7.517 | 7.394 | 0.123 | 84 | 84 | 0% |
| bff_s2 | 7.397 | 7.223 | 0.174 | 98 | 98 | 0% |
| bff_s3 | 7.392 | 7.220 | 0.172 | 84 | 84 | 0% |
| bff_n8192_s1–3 | 7.51–7.53 | 7.22–7.26 | 0.27–0.29 | 272–277 | 272–277 | 0% |
| bff_n32768_s1–3 | 7.49–7.53 | 7.23–7.29 | 0.24–0.28 | 367–374 | 367–374 | 0% |
| bff_long_s1 | 7.490 | 7.088 | 0.402 | 121 | 121 | 0% |

The filter removes 42–71% of the crystal runs' A(t) and **nothing at all** from
the bff runs, whose byte distributions are still near-uniform so the null model
predicts essentially zero for every window. That is the intended behaviour: it
is a filter against a *biased alphabet*, not against structure.

Two honest caveats. The rescored numbers use the tape-dump cadence, which is
20× coarser than the metric cadence, so they are comparable with each other and
not with the live `A_t` column. And the null model is i.i.d., so it knows the
soup's letter frequencies but not its spatial structure: a field of clustered
`,` runs still scores as novel. That is arguably correct — the clustering *is*
non-i.i.d. structure — but it is not a program either, which is what the
classifier is for.

### 5. New: what the machine actually did

`frac_steps_in_loop` is the fraction of executed steps at an instruction
pointer the run had already visited — time spent re-running code. It is
defined on addresses rather than brackets so that loops built by
self-modification count too. `ops_per_run` counts executed instructions
excluding no-ops, which is the quantity cubff reports as "ops".

These separate two states that the entropy metrics conflate. A ring crystal
and a random soup both sit near HOE = 0, but a random soup spends 80–90% of
its steps re-running code (random memory is full of brackets) while a crystal
spends ~1% (the brackets have been purged and every run is a straight walk).

### The end-state classifier

Every finished run is labelled from the median of its last five snapshots,
with these thresholds applied identically to all of them:

| class | rule |
|---|---|
| **program** | HOE ≥ 1.0 **and** mean steps ≥ 3 × walk length **and** the most frequent 16-byte window contains a bracket |
| **crystal** | H < 3.0 **and** in-loop fraction < 0.10 **and** mean steps ≤ 1.5 × walk length |
| **random** | H ≥ 7.0 **and** HOE < 0.5 |
| **mixed** | none of the above |

"Walk length" is the number of steps a run takes if it executes only no-ops:
`R+1` for ring, 128 for bff and blocks, 126 for cubff compat (its pc starts at
2). Mean steps well above it is the signature of loops actually running. The
bracket requirement is what separates a program from a crystal that happens to
be compressible: a dominant pattern with no control flow in it is not a
program, however abundant.

---

## Things that look like bugs rather than findings

Stated plainly, including the ones that turned out to be neither.

1. **The bff non-result is not a bug, but it is not a clean validation
   either.** It was chased down rather than reported: the interpreter supports
   exact self-replication (verified), the dynamics amplify a planted replicator
   to fixation in ~20 epochs (verified), the metrics register that takeover
   unmistakably (verified). Ten runs, three of them at the reference's own
   interaction budget, produced no spontaneous transition. Against a reported
   40% per-run rate that is an unremarkable outcome (p ≈ 0.22 for three
   misses), so the correct statement is that bff mode has **not yet validated
   the interpreter against the published result** — it has only shown that
   nothing in the substrate prevents the result.
2. **A single planted replicator usually goes extinct** (5 of 8 seeds). This
   looked like a failure of the bff driver at first. It is founder
   stochasticity: the hand-written replicator only copies when the random
   pairing puts it in the first half, so its per-epoch growth factor is ~1.5,
   and extinction from one founder is likely. With 20 founders, takeover is
   reliable across seeds.
3. **Mean steps per run falling to 128.5 in the ring** looked like the
   interpreter failing to execute anything. It is real and explainable: 129
   steps is exactly a straight walk from `p` to the far edge of the window, and
   the soup has purged the brackets, so there are no loops left to run.
4. **A(t) rising forever in a dead soup** is a property of the metric as
   specified, quantified above, not a counting error — the persistence logic is
   unit-tested against hand-built cases.
5. **`high_order_entropy` reading ≈ 0 for a fully structured soup** is likewise
   the metric behaving as defined, not a compression bug; `zlib_bits` for
   `ring_s1` is 1.245 bits/byte against 8.003 for random memory.
6. **Not resolved:** whether the reference implementation of bff wraps heads,
   clamps them, or terminates on out-of-bounds. The published description
   available here does not say. Both readings implemented here produce the same
   ring outcome (above), but the two are not guaranteed to agree in bff, and a
   faithful reproduction of the published result would want this pinned down.

---

## Performance

Measured on 4 cores of an Intel Xeon @ 2.80 GHz (Linux 6.18, Python 3.11.15,
numpy 2.4.6, numba 0.67.0). Seeds are run as separate processes, so the
wall-clock figures below include contention between them; `sim` is time inside
the compiled kernel.

**The target was 50k epochs of the default ring config in under an hour. It
takes 142–149 seconds — about two and a half minutes, with three seeds running
concurrently on four cores — so roughly 25× inside the budget.**

| run | epochs | wall | sim | epochs/s |
|---|---|---|---|---|
| ring (M=65536, R=128), 3 seeds in parallel | 50000 | 142–149 s | 85–87 s | 336–352 |
| ring, head-bound=halt, 3 seeds | 50000 | 125–126 s | 73–74 s | ~399 |
| bff N=1024, 3 seeds in parallel | 20000 | 57–64 s | 30–36 s | 315–348 |
| bff N=8192, 3 seeds in parallel | 20000 | 359 s | 181 s | 56 |
| bff N=1024, single process | 200000 | 248 s | 228 s | 805 |
| bff N=32768, 3 seeds in parallel | 64000 | 2659–2668 s | 2306–2326 s | ~24 |

`python -m soup.bench`, single process, measured on the *random* soup — the
slowest phase for the ring, because random memory is full of brackets and runs
average 1565 steps instead of the 129 they average once the soup has settled:

```
ring M=65536 R=128 k=8192     260 epochs/s    266k runs/s   4.2e8 steps/s
bff  N=1024  k=8192          1264 epochs/s    647k runs/s   3.5e8 steps/s
```

Roughly 4e8 interpreted byte-machine steps per second per core, including the
bracket scan, which is re-done on every executed bracket because self-modifying
code means a precomputed jump table would be wrong. About 40% of the wall-clock
in a ring run is metrics, not simulation: zlib level 9 plus two `np.unique`
passes over ~65k windows at every snapshot.

---

## Reproducing

```sh
python -m pytest tests/ -q
for s in 1 2 3; do
  python -m soup.run --config configs/bff.json  --seed $s --out runs/bff_s$s
  python -m soup.run --config configs/ring.json --seed $s --out runs/ring_s$s
done
python -m soup.run --config configs/bff.json --seed 1 --epochs 400 --plant 20 \
                   --snapshot-interval 10 --out runs/bff_plant20_s1
# the matched-budget bff runs (about 45 minutes each)
for s in 1 2 3; do
  python -m soup.run --config configs/bff_scale.json --seed $s --out runs/bff_n32768_s$s
done
# the head-confinement sensitivity set
for s in 1 2 3; do
  python -m soup.run --config configs/ring.json --seed $s --head-bound halt \
                     --out runs/ring_halt_s$s
done
python experiments/invasion.py --epochs 400
python -m soup.plots   runs/ring_s1 runs/ring_s2 runs/ring_s3 --out analysis/ring
python -m soup.analyze runs/ring_s1 runs/ring_s2 runs/ring_s3
```
