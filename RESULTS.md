# Results

Runs: `bff` at three seeds and `ring` at three seeds, plus controls and
sensitivity runs. Every number below comes from a committed run directory;
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
* **bff — no.** Nothing took over in any run: 3 seeds × 20k epochs at N=1024,
  3 seeds × 20k epochs at N=8192, and one run of 200k epochs at N=1024. This
  is expected rather than alarming, for a reason given below, and the
  interpreter and the metrics are separately verified to detect a takeover
  when one happens.

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

## bff: no spontaneous takeover, and why that is the expected outcome

| run | N | epochs | interactions | HOE final | HOE max | H final | A(t) | takeover |
|---|---|---|---|---|---|---|---|---|
| bff_s1 | 1024 | 20000 | 1.0e7 | 0.214 | 0.529 | 7.517 | 547 | no |
| bff_s2 | 1024 | 20000 | 1.0e7 | 0.226 | 0.600 | 7.397 | 499 | no |
| bff_s3 | 1024 | 20000 | 1.0e7 | 0.242 | 0.575 | 7.392 | 535 | no |
| bff_n8192_s1 | 8192 | 20000 | 8.2e7 | 0.341 | 0.592 | 7.509 | 2229 | no |
| bff_n8192_s2 | 8192 | 20000 | 8.2e7 | 0.320 | 0.576 | 7.514 | 2230 | no |
| bff_n8192_s3 | 8192 | 20000 | 8.2e7 | 0.320 | 0.597 | 7.527 | 2410 | no |
| bff_long_s1 | 1024 | 200000 | 1.0e8 | 0.466 | 0.694 | 7.490 | 262 | no |

The reference result (Agüera y Arcas et al. 2024) uses a soup of **2^17 = 131072
tapes** and reports that **40% of runs show a state transition within 16k
epochs** — i.e. even at full scale, most runs do not transition in that window.
16k epochs at 2^17 tapes is about **1.1e9 pairwise interactions**. The runs in
the table above are between 10 and 100 times short of that budget, so finding
no transition in them is the outcome the published numbers predict, not
evidence of a broken interpreter.

<!--SCALE-->

What the bff soups *do* show is the same chemistry the ring runs push to
completion, just never reaching fixation: `<` is enriched to ~7% of memory
against a random-soup level of 1/256 = 0.39% (18×) and `,` to ~2.5% (6×), while
`+`, `-` and `]` stay at the random level. See
`analysis/bff/byte_composition.png`. This is a stationary state, not a slow
climb — over the 200k-epoch run `<` sits between 0.051 and 0.072 and `,`
between 0.022 and 0.047 from epoch 40000 onwards, with no trend:

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
against a maximum of 8192 — a run now consists of walking forward through the
window and falling off the far end, with no loops. The soup has become quiet.

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
| 83% `>` + 11% `,` | 0.0042 | 0.0089 | 0.927 | barely spreads |
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
doing the copying — 7× weaker. `{`/`.` is the exact mirror (`.` writes to
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

## What high-order entropy misses

`H - zlib` is near zero for i.i.d. random bytes **and** near zero for memory
that has collapsed to one repeated byte, because in the second case the order-0
entropy has already fallen to meet the compressed size. In the ring runs it
spikes to ~1.0–1.4 bits/byte *during* the transition, when structured and
unstructured memory coexist, and then falls back to 0.035 in seed 1 — a value
indistinguishable from the random initial condition, describing a soup that
could not be more different from it.

So the metric the brief nominates as the takeover signal is, in this soup, a
detector of the *transition* and not of the *state*. `entropy_bits` has to be
read next to it: 8.0 means random, ~1.3 means monoculture, and high-order
entropy alone cannot tell them apart. The planted-replicator control shows the
case high-order entropy is genuinely good at — many copies of a long,
byte-diverse pattern, which keeps `H` at 5.9 while `zlib` collapses to 0.33.

## What A(t) misses

A(t) rises monotonically in every run and never flattens: in the ring it adds
~72 new persistent windows per 1000 epochs over the first half and still ~16
per 1000 over the second half, at epoch 50000, in a soup made of two byte
values. Two reasons, both of which are measurement artifacts:

1. **Combinatorics of a small alphabet.** With `<` at 0.82 and `,` at 0.11,
   the expected count of an 8-byte window containing *j* `,` bytes is
   65536 × 0.82^(8−j) × 0.11^j, which stays above `c_min` = 8 up to j = 3 —
   93 windows before anything else is counted. Fluctuation and the ±1
   neighbours that `+`/`-` and mutation produce (`;`, `=`) supply the rest.
   57% (seed 1) to 74% (seed 2) of the windows persistent at the end are built
   only from `< , { .`.
2. **`tau` is counted in snapshots, not epochs.** The same bff configuration
   logged every 20 epochs reaches A(t) = 547, and logged every 200 epochs
   reaches A(t) = 262 — a factor of two from the logging cadence alone
   (`bff_s1` vs `bff_long_s1`, which is also 10× longer).

A(t) as specified answers "how many distinct 8-byte windows have ever been
abundant for a while", and that question has a large answer in a soup with a
two-letter alphabet. It is not a novelty measure in this regime.

---

## Things that look like bugs rather than findings

Stated plainly, including the ones that turned out to be neither.

1. **The bff non-result is not a bug.** It was chased down rather than
   reported: the interpreter supports exact self-replication (verified), the
   dynamics amplify a planted replicator to fixation in ~20 epochs (verified),
   the metrics register that takeover unmistakably (verified), and the
   published reference reports only a 40% transition rate at a soup size 128×
   larger than the primary configuration here. The runs are simply 10–100×
   short of the interaction budget where transitions are reported.
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

<!--PERF-->

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
python experiments/invasion.py --epochs 400
python -m soup.plots   runs/ring_s1 runs/ring_s2 runs/ring_s3 --out analysis/ring
python -m soup.analyze runs/ring_s1 runs/ring_s2 runs/ring_s3
```
