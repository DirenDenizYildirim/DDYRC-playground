# Results

**Leading question, answered first: after a takeover, does evolution measurably
continue, and for how long?** It continues in **one of four seeds**, as a
**single discrete step** about twenty thousand epochs after the takeover, after
which it **stops**. In the other three seeds nothing measurable happens in
fifty thousand epochs. The evidence is a competition assay with a neutral
control, corrected for multiple testing, and is set out immediately below.

Everything else — the validation against cubff, the metric rebuild, the
locality sweep and the boundary-free ring variants — follows after.

---

## 1. Does evolution continue after takeover?

### What was run

Four seeds whose bff/`--compat cubff` runs at N = 32768 had already produced a
program-class takeover were resumed from the checkpoint at which the takeover
was detected and run a further **50 000 epochs each**, well mixed, at the
normal mutation rate, with tape dumps every 500 epochs.

| run | resumed from epoch | ran to | tape dumps |
|---|---|---|---|
| `plateau_s12` | 5 500 | 55 500 | 101 |
| `plateau_s13` | 5 750 | 55 750 | 102 |
| `plateau_s14` | 7 000 | 57 000 | 101 |
| `plateau_s19` | 12 750 | 62 750 | 102 |

### The assay

Saved soups were compared against each other by putting them in the same soup:
half the tapes from A, half from B, labelled, run 300 epochs at the normal
mutation rate, five replicates with different shuffles. The label rides along
with `.` and `,` and **no instruction can read it**, so the measurement does
not perturb what it measures (`tests/test_labels.py` asserts a labelled run is
byte-identical to an unlabelled one). In every pair the *later* soup is side A,
so a value above 0.5 means later beat earlier.

**The control is the same soup against itself.** It matters more than it looks:
lineage drift at N = 8192 over 300 epochs is large, and a self-control lands
anywhere in 0.40–0.64. Reading the raw fractions without it would produce
"findings" at every second comparison.

| seed | self-control, 10 replicates | sd |
|---|---|---|
| 12 | mean 0.5187, range 0.4558–0.5714 | 0.042 |
| 13 | mean 0.5075, range 0.3800–0.5951 | 0.070 |
| 14 | mean 0.5031, range 0.4012–0.6448 | 0.072 |
| 19 | mean 0.4794, range 0.4039–0.5404 | 0.051 |

Each comparison is tested against its seed's pooled self-controls with a Welch
*t*-test. That gives **84 tests across four seeds**, where roughly four hits at
*p* < 0.05 are expected by chance — and exactly two isolated ones appeared
(seed 12 at epoch 20 000, seed 19 at 62 750). **Holm–Bonferroni over all 84
leaves eight, every one of them seed 14 at epoch 30 000 or later.** The two
isolated hits do not survive, which is what a false positive should do.

### The result

**Seed 14** — the soup at each age against the first post-takeover soup:

| epoch | vs first soup | vs previous soup |
|---|---|---|
| 10 000 – 25 000 | 0.49–0.52, no difference (p ≥ 0.62) | no difference |
| **30 000** | **0.868 ± 0.034, p = 3 × 10⁻⁵** | **0.914 ± 0.022, p ≈ 0** |
| 35 000 | 0.853 ± 0.017, p ≈ 0 | 0.667, p = 0.005 |
| 40 000 | 0.898 ± 0.020, p ≈ 0 | 0.703, p = 0.001 |
| 45 000 | 0.868 ± 0.027, p ≈ 0 | 0.549, no difference |
| 50 000 | 0.867 ± 0.020, p ≈ 0 | 0.466, no difference |
| 57 000 | 0.872 ± 0.026, p ≈ 0 | 0.529, no difference |

**Seeds 12, 13 and 19**: every comparison is inside the neutral band, at every
age, against both the first soup and the previous one. Fifty thousand epochs,
nothing.

So, to the question as posed — *does later beat earlier, and does the advantage
keep growing, stop, or cycle?* — **later beats earlier in one seed of four, by
a large margin, and then the advantage stops growing.** It does not cycle: once
the step is taken, consecutive soups stop beating each other while the margin
over the *first* soup holds flat at about 0.87 for the remaining 27 000 epochs.
The step is a one-off, not a rate.

The independent motif tracking agrees on timing. Across the run, leading
families come and go with coverage 0.10–0.17; at **epoch 29 500 a family sweeps
to 0.443 coverage**, two-and-a-half times any other leader in that run. Nothing
comparable happens in the other three seeds, whose largest family coverage over
the whole 50 000 epochs is 0.15–0.26.

### What did not change when it happened

The step is not a gain in self-sufficiency. A tape drawn from the soup and
paired with a *fresh uniform-random* tape colonises the same share of the pair
before and after:

| seed | self-sufficiency, first → last | soup-vs-soup control | distinct tapes, first → last | sweeps |
|---|---|---|---|---|
| 12 | 0.789 → 0.771 | 0.4962 ± 0.0182 | 2 970 → 2 766 | 7 |
| 13 | 0.771 → 0.772 | 0.4970 ± 0.0185 | 2 899 → 2 867 | 2 |
| **14** | 0.688 → 0.718 | 0.4996 ± 0.0113 | **15 766 → 28 313** | **26** |
| 19 | 0.774 → 0.765 | 0.4979 ± 0.0170 | 2 910 → 4 481 | 4 |

Seed 14's self-sufficiency is 0.7011 ± 0.0067 across its whole continuation and
does not move at epoch 30 000. Whatever the step bought, it was an advantage
*against the other soup*, not a better machine in isolation.

One correlation worth recording and not over-reading: **seed 14 is also the
only one of the four whose population was diverse to begin with** — 15 766
distinct tapes at the resume point against about 2 900 for the others, rising
to 28 313 of 32 768. The seed that kept evolving is the seed that had standing
variation. With n = 1 that is a hypothesis, not a finding.

### A(t) does not see any of this

The cumulative count of novel persistent 8-byte windows, filtered against the
i.i.d. null, grows **linearly and at the same rate in all four seeds**:

| seed | HOE, start → end | A(t) at end | A(t) slope, 2nd half | standing stock |
|---|---|---|---|---|
| 12 | 6.35 → 7.00 | 81 349 | 1.58 /epoch | 1 090 |
| 13 | 6.22 → 7.13 | 85 775 | 1.72 /epoch | 1 142 |
| 14 | 5.62 → 6.54 | 72 637 | **1.45 /epoch** | 997 |
| 19 | 5.82 → 7.10 | 86 138 | 1.73 /epoch | 1 150 |

**A(t) does not flatten.** It also does not distinguish the one seed where
evolution demonstrably continued from the three where it demonstrably did not —
and seed 14's slope is the *lowest* of the four. Meanwhile the *standing stock*
of currently-persistent windows is flat at about a thousand throughout. A(t) is
measuring turnover: windows are discovered and lost at matched rates, and the
cumulative counter integrates the discovery rate. On this evidence A(t) is not
a measure of continuing evolution, and the competition assay is. That is the
single most useful thing these runs produced.

A caveat on the filter: post-takeover, `A_t` and `A_t_naive` are **identical**.
The soup's byte frequencies are skewed enough that every window abundant enough
to qualify also clears five times its i.i.d. expectation, so the null filter
removes nothing at all once a takeover has happened. It still does its job
before one.

### Figures

`analysis/plateau_report/` — `assay_vs_first.png` (the competition curves with
the control band), `advantage_vs_first.png` and `advantage_vs_prev.png` (final
advantage against soup age), `families.png` (motif family coverage with sweeps
marked), `selfsuff.png`, `profiles_seed*.png` (per-offset entropy around the
leading motif). `analysis/plateau_report/summary.md` carries every number and
`summary.json` the machine-readable form.

---

65 runs across ten configurations: the ring baseline and its
boundary-free variants, bff at four scales, bff reproduced bit-exactly against
[cubff](https://github.com/paradigms-of-intelligence/cubff) in both of its
languages, the blocks locality sweep, a head-confinement sensitivity set and a
planted-replicator control. Every number comes from a committed run directory;
`python -m soup.analyze --rates runs/...` regenerates the tables and
`python -m soup.plots runs/...` the figures.

---

## The instrumentation, and the three ways it was wrong first

Everything in section 1 rests on four new measurements. Each one had a flaw
that would have produced a confident wrong answer, and each flaw was found by
running a control that had to come out a particular way.

**Provenance labels** (`soup/interp.py`). A second byte array the same shape as
memory. `.` and `,` copy a byte's label with the byte; every other instruction,
and mutation, leave the destination cell's label alone. No instruction can read
a label. The control is that a labelled run must be byte-identical to an
unlabelled one — memory, every counter, and every column of the metric row —
and `tests/test_labels.py` asserts it at three seeds. A second test relabels a
population and checks the trajectory is unchanged and only the bookkeeping
mirrors.

**The competition assay** (`soup/assay.py`). Half the tapes from soup A, half
from B, labelled, 300 epochs, five replicates. Two controls: a dead soup of
no-ops must never gain ground, and — the one that mattered — **A against
itself**. The self-control lands anywhere in 0.40–0.64. Without it the raw
fractions look like signal at every second comparison; with it, 76 of 84
comparisons are noise. The range test I wrote first was the opposite error,
so conservative it would have missed seed 14 entirely; the Welch test against
pooled self-controls, Holm-corrected, is what the numbers above use.

**Motif families** (`soup/motifs.py`). The dominant 16-byte windows of a
taken-over soup are mostly shifts of one string, so counting windows overstates
how much is there. Windows within edit distance 8 are one family, and a
family's *coverage* is the fraction of tapes carrying it — a frequency with a
denominator worth quoting. The threshold is not a guess: `null_distances()`
draws 2000 pairs from a soup's own byte-frequency null, and the smallest
distance any unrelated pair reached was 12. My first attempt used 4, which
split one obvious motif into four families.

Sweep counting is reported but should be read carefully. The detector fires 26
times in seed 14 and 2–7 times in the others, and most of those are the
representative window of a quasispecies drifting, not a selective sweep. The
one that coincides with the assay's step reaches 0.443 coverage where every
other leader in that run sits at 0.10–0.17. **Coverage magnitude separates the
two; the sweep count does not.**

**Per-snapshot post-takeover measurements** (`soup/posthoc.py`). Self-sufficiency
pairs a soup tape with a fresh uniform-random tape; *colonisation* is the share
of the resulting 128 bytes carrying the soup tape's label, 0.5 when neither
side gains. Controls are soup-vs-soup and random-vs-random. Two things went
wrong here:

- The soup-vs-soup control read 0.4728 and looked like a real asymmetry. It is
  not: a single pair run can hand over most of its 128 bytes, so the per-pair
  spread is wide, and quoting the standard error puts the control at 0.5.
- Worse, every snapshot was analysed with the same seed, so every one reused
  the same draw pattern. That pinned the control at 0.522–0.524 across all
  thirty snapshots of a run — stable enough to look like precision — when five
  different seeds on a *single* soup give 0.4985, 0.5061, 0.5070, 0.5092,
  0.5225. Seeding per snapshot put the control back to 0.4996 ± 0.0113 and made
  the across-snapshot spread an honest error bar.

For *where the variation sits*, tapes are aligned on the leading family's
representative window and per-offset byte entropy is reported. Offsets 0–15 are
the seed window, so low entropy there is circular and is shown only for
completeness; the conserved width outside that range is the measurement, and it
runs to 33–53 bytes — well past the 16-byte seed. That column also needed a
floor: before it required 256 tapes per offset, a *pre-takeover* soup whose
leading family covers 0.1% of tapes reported a 23-byte conserved region built
from 40 nearly identical tapes. It now reports nothing there, which is correct.

**Resuming.** The runs in section 1 could not have been completed without it:
background work in this environment does not survive an idle period, and the
first attempt at these continuations died 2 500 epochs in. A run now resumes
from its last tape dump and carries on the same `metrics.csv`, `patterns.log`
and kymograph — rows past the resume point dropped, the row at it kept for the
counters it carries, and A(t) reloaded from tracker state saved beside every
dump. The test tears a run at epoch 20, resumes it, and asserts the metric
rows, the kymograph and the pattern log are identical to an uninterrupted run.
The first set of dumps was discarded rather than resumed from, because it
carried no tracker state and resuming would have reset A(t) mid-series.

---

## 2. Which configurations produce a takeover at all

**One family of configurations produces a program-class takeover: bff pairing
with cubff's initial machine state.** Nothing else tested does.

| configuration | n | program-class | rate (95% CI) |
|---|---|---|---|
| **bff, `--compat cubff`** (head0/head1 read from `tape[0..1]`, pc = 2) | 17 | **11** | **0.65 (0.41–0.83)** |
| bff, `--compat cubff_noheads` (our spec: heads at 0, pc = 0) | 7 | 1 | 0.14 (0.03–0.51) |
| bff, our scaffolding, N = 1024 … 32768 | 10 | 0 | 0.00 (0.00–0.28) |
| blocks, d = 1 / 2 / 8, N = 8192 | 11 | 0 | 0.00 (0.00–0.26) |
| ring, control | 3 | 0 → *crystal* | 0.00 (0.00–0.56) |
| ring, no copy primitive | 3 | 0 → *random* | 0.00 (0.00–0.56) |
| ring, with insertions and deletions | 3 | 0 → *crystal* | 0.00 (0.00–0.56) |

Counts are of distinct experiments, not run directories: `ringv_ctrl_*` and
`bffnew_*` re-run the original ring and bff configurations under the new
metrics at the same seeds, and are not counted twice. The full inventory is at
the end.

**`--compat cubff` at N = 32768 is therefore the baseline to build on.** It is
the only configuration here where something that computes takes over, it does
so within 2000–11250 epochs, and it is reproducible bit-for-bit against the
reference implementation.

### The four questions

**Did a replicator transition occur?** Yes, in two distinct senses that should
not be confused.

* **A program takeover**, in cubff-semantics bff: high-order entropy 5.1–6.4,
  runs saturating the 8192-step budget, 99.4–99.7% of steps re-running code,
  and dominant 16-byte windows that are code containing brackets.
* **A crystal takeover**, in every ring configuration that has a copy
  primitive: memory collapses to a two-byte field (`<` ~82%, `,` ~11%, or the
  `{`/`.` mirror) that genuinely self-propagates but executes nothing. Loops
  are purged; 0.00 of steps re-run an instruction.

The classifier separates them, and the distinction is the main thing the first
version of this baseline got wrong by reporting only high-order entropy.

**At what epoch?** Program takeovers: 2000–11250 (median 8000), i.e. 3.3e7 to
1.8e8 pairwise interactions. Crystal takeovers: order-0 entropy below 4.0 by
epoch 2100–2800, in the first 5% of the run.

**What do the dominant patterns look like?** Programs: a *quasispecies*, not a
clone — 16636 distinct tapes in a 32768-tape soup, no tape with more than 6
copies, but a shared 16-byte motif in 20.5% of them. Crystals: runs of `<`
interrupted by an occasional `,`, and nothing else — the top ten 16-byte
windows at the end of a ring run are `<<<<<<<<<<<<<<<<` (20630), then
`,,,,,,,,,,,,,,,,` (539), then single-`,` substitutions.

**Does A(t) plateau afterwards?** With the null filter and epoch-based `tau`,
A(t) behaves far better than it did: it is flat at exactly **0** for the whole
50000 epochs of the no-copy control, where nothing whatever happens, which the
old A(t) could not have shown. In the crystal runs it still climbs, at 3–5×
lower values than before (362 against 1967 for ring seed 1). In the program
runs it jumps by an order of magnitude at the transition — 29 to 1415 in one
snapshot — and then stops climbing steeply. So: it plateaus where the soup is
dead, it jumps where a program takes over, and it still drifts upward in a
crystal, which remains the metric's weakest case.

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

## bff, settled: the transition is real, and the head rule sets its rate

Running cubff's own language bit-exactly answers the question the earlier runs
could not. **Program-class takeovers happen, they happen quickly, and the
initial machine state is what governs how often.**

Budget: N = 32768 tapes, 16000 epochs — the window the reference paper states
its own rate over. A seed counts as a transition if the detector fires at or
before epoch 16000 and the end state classifies as *program*.

| configuration | n | program | rate | 95% CI (Wilson) | takeover epochs |
|---|---|---|---|---|---|
| `--compat cubff` (heads from tape, pc=2) | 17 | 11 | **0.65** | 0.41–0.83 | 2000, 2750, 4000, 4250, 5500, 8000, 8750, 9500, 9500, 10250, 11250 |
| `--compat cubff_noheads` (our spec) | 7 | 1 | **0.14** | 0.03–0.51 | 2750 |
| our own bff scaffolding, N=32768, 64000 epochs | 3 | 0 | 0.00 | 0.00–0.56 | — |

The 0.65 rate is statistically consistent with the 40%-within-16k-epochs the
paper reports, and the takeovers are not marginal: median epoch 8000, i.e.
1.3e8 pairwise interactions, well inside the budget. **The published emergence
result reproduces.**

The head rule is the difference, but the evidence is suggestive rather than
conclusive at this sample size. Comparing the two compat languages directly,
11/17 against 1/7 gives Fisher p = 0.069. Pooling the three earlier N=32768
runs — the same language, our scaffolding, a *four times longer* budget, no
transition — gives 11/17 against 1/10 (rate 0.10, CI 0.02–0.40) and p = 0.014.
Pooling is defensible here because `--compat cubff_noheads` is provably the
same language as our bff (byte-exact against cubff, `tests/test_compat.py`),
but it mixes two RNG streams and two epoch budgets, so it is stated separately
rather than quietly merged.

**The earlier write-up said bff_noheads never transitions. That was wrong** —
it was an artefact of ten runs, not a property of the language. Seed 103
transitions at epoch 2750 with HOE 3.45 and dominant windows that are plainly
code (`Qa.{.H......[...`, 1169 occurrences). The honest statement is that both
languages transition and the heads variant transitions several times more
often.

### What a takeover looks like

Every transitioned run moves together on every metric, within one snapshot:

| | before | after |
|---|---|---|
| order-0 entropy H | 7.75–7.92 | 5.80–6.86 |
| high-order entropy | 0.12–0.17 | **5.10–6.35** |
| mean steps per run | 830–1130 | 7720–8174 (budget 8192) |
| fraction of steps re-running code | 0.89–0.93 | 0.994–0.997 |
| fraction of runs copying | 0.74 | 0.999 |

### The end state is a quasispecies, not a clone

This is where the emergent replicator differs sharply from the planted one.
Tracking seed 11 across its memory dumps:

| epoch | distinct tapes (of 32768) | copies of the modal tape | occurrences of the dominant motif |
|---|---|---|---|
| 0–9000 | 32766–32768 | 1–3 | 0 |
| 10000 | 16636 | 6 | 6708 |

The transition is **abrupt** — it completes inside a single 1000-epoch window,
which is what makes "phase transition" the right word — but what takes over is
a *family*. There is no dominant tape: the most common exact 64-byte tape
appears 6 times out of 32768, and 8005 distinct tapes are needed to cover half
the soup. What is shared is a motif: `.{W.E..<..@]>...` occurs 6708 times, one
per carrier, in 20.5% of tapes.

The planted control is the opposite: 1006 identical copies of one 25-byte
program in a 1024-tape soup. Emergent replication here is a cloud of variants
around a motif; designed replication is a clone.

The motif is also not self-sufficient. Pairing a carrier with a *fresh random*
tape, the motif count grows in 26% of trials and the mean change is −0.24; two
random tapes never produce it (0 occurrences in 300 trials). At its 20%
equilibrium abundance, carrier × non-carrier pairings gain +0.07 motifs on
average, with gains and losses nearly balanced — the population sits at a fixed
point rather than mid-sweep. So the replicators here depend on an environment
that is already mostly relatives, which is a property worth knowing before
building on this baseline.

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

## blocks: locality, with bff's pairing held fixed

`--mode blocks` was built to separate the two things that differ between bff
and ring at once — pairing and locality. It keeps bff's pairing exactly (two
64-byte tapes concatenated, run from ip = head0 = head1 = 0) and varies only
how far apart the partners may be: a block pairs with one of the `2d` blocks
within `±d`. N = 8192, 20000 epochs.

| d | seeds | class | H | HOE | mean steps | in-loop | copy frac | A(t) |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | random | 7.979–7.981 | −0.019 – 0.038 | 581–618 | 0.897–0.903 | 0.397–0.403 | 252–268 |
| 2 | 5 | random | 7.969–7.971 | 0.075–0.092 | 562–589 | 0.892–0.899 | 0.385–0.403 | 276–297 |
| 8 | 3 | random | 7.845–7.876 | 0.298–0.343 | 581–631 | 0.866–0.878 | 0.448–0.485 | 515–563 |

**No configuration produced a program-class takeover, and none came close.**
What the sweep does show is a clean monotone effect of mixing: high-order
entropy rises 0.02 → 0.08 → 0.32 with `d`, A(t) roughly doubles from `d = 1`
to `d = 8`, the copy fraction rises, and the in-loop fraction falls. More
mixing, more structure — the opposite of the intuition that locality helps
replicators by keeping relatives together.

Two honest qualifications. 20000 epochs at N = 8192 is 8.2e7 pairwise
interactions, and blocks mode runs **our spec's** initial machine state, which
is the one that produced no transition in ten bff runs even at 1.05e9
interactions. So this sweep measures how locality behaves *inside the
non-transitioning regime*; it does not test whether locality would help a soup
that transitions. The obvious follow-up is blocks with cubff's head rule, which
is now a one-flag change and was not run here.

---

## 3. Locality: does the takeover still originate when pairing is local?

**No, in six runs out of six.** Blocks pairing with the cubff head rule, from
random initial conditions, N = 32768, 16 000 epochs, three seeds at each
neighbourhood radius:

| configuration | n | program-class | rate (95% CI) | H | HOE | in-loop | copy frac | A(t) |
|---|---|---|---|---|---|---|---|---|
| **bff, well mixed** (`--compat cubff`) | 17 | **11** | **0.65 (0.41–0.83)** | — | — | — | — | — |
| blocks, d = 2, cubff head rule | 3 | 0 | 0.00 (0.00–0.56) | 7.977–7.979 | 0.056–0.061 | 0.905–0.907 | 0.53–0.54 | 257–333 |
| blocks, d = 8, cubff head rule | 3 | 0 | 0.00 (0.00–0.56) | 7.936–7.941 | 0.160–0.174 | 0.899–0.903 | 0.58–0.58 | 256–259 |
| blocks, pooled | 6 | 0 | 0.00 (0.00–0.39) | — | — | — | — | — |

Every run is classified *random*: order-0 entropy within 0.07 of the 8.0
ceiling, high-order entropy two orders below the 1.0 a program-class takeover
reaches, and A(t) in the hundreds against tens of thousands. Against the
well-mixed rate at the same head rule and the same N, Fisher's exact test gives
**p = 0.0137** pooled (p = 0.074 for d = 2 alone, which three seeds cannot do
better than).

There is a hint of dose-response in the right direction: d = 8 ends with
high-order entropy of 0.160–0.174 against 0.056–0.061 for d = 2, and a higher
copy fraction, so a wider neighbourhood is slightly closer to the well-mixed
case. It is a difference between two flat lines, not a transition.

**What was not run, and why.** The specification asked for six seeds at each
radius; three were run. It also asked for the four takeover checkpoints to be
continued under blocks d = 2 with the same measurements, which was not run at
all. Both were cut for compute: the continuations in section 1 took six hours
of the budget on their own, and an hour of that had already been lost to the
environment killing background work before the runs were made resumable. The
d = 2 result on three seeds cannot reach p < 0.05 by itself even if the true
rate is zero, so the pooled test is the one to quote. Nothing was tuned or
re-run to get these numbers; they are the first six runs of that configuration.

---

## Boundary-free ring variants

Three seeds each, 50000 epochs, M = 65536, R = 128.

| variant | class | H | HOE | mean steps | in-loop | copy frac | A(t) | A(t) naive |
|---|---|---|---|---|---|---|---|---|
| **a. control** | crystal ×3 | 1.279, 2.120, 1.280 | 0.006, 0.965, 0.015 | 128.2–128.8 | 0.00–0.09 | 0.82–0.85 | 362–2214 | 1967–2493 |
| **b. no copy** | random ×3 | 7.975–7.977 | −0.025 – −0.023 | 1864–2083 | 0.946–0.952 | **0.000** | **0** | **0** |
| **c. indel** | crystal ×2, mixed ×1 | 2.483, 2.567, 2.853 | 0.636, 0.912, 1.000 | 128.6–179.8 | 0.00–0.32 | 0.79–0.85 | 760–2163 | 2067–2340 |
| **d. data-driven heads** | mixed ×3 | 6.383, 6.493, 6.467 | 0.235, 0.211, 0.228 | 551–747 | 0.79–0.85 | 0.28–0.31 | **53–61** | 53–61 |

**(a) Control** reproduces the earlier ring result under the new metrics, and
the new in-loop column makes the crystal diagnosis direct rather than inferred:
**0.00** of steps are spent re-running an instruction, against 0.95 in a random
soup. The brackets really are gone and every run really is a straight walk.

**(d) The data-driven head rule stops the crystal without producing a
program.** This is the one variant where cubff's own convention is carried over
to the boundary-free geometry: `head0` and `head1` are read from `mem[p]` and
`mem[p + 1]` and execution starts at `p + 2` — exactly the rule that decides
the transition rate in bff. Three seeds, 50 000 epochs each.

The end state is neither of the two the ring reached before. It is **not** the
crystal: order-0 entropy ends at 6.38–6.49 against 1.28 for the control ring,
and 0.79–0.85 of steps are spent re-running an instruction against 0.00 for a
crystal, so the brackets are still there and runs still loop. It is **not**
random either: entropy is a bit and a half below the 7.98 of the no-copy ring,
and 28–31% of runs still copy. And it is emphatically not a program:
high-order entropy is 0.21–0.24 where a program-class takeover reaches 1.0 and
above, and **A(t) ends at 53–61** against 72 637–86 138 for the bff
continuations. The classifier calls all three *mixed*.

So the head rule does change the boundary-free ring's fate — away from
crystallisation — but it does not supply what bff has. Whatever separates the
two, it is not the initial machine state alone.

*A correction to my own first reading of this run.* I initially called it a
crystal on the strength of the top-10 window log, whose first two entries are
`01 01 01 …` and `00 00 00 …` in every seed. Those entries have counts of 1905
and 1001 out of 65 536 windows — 2.9% and 1.5%. They are the commonest 16-mers
in a largely disordered soup, not a dominant motif. The pattern log has no
denominator in it and should not be read as though it did; `frac_steps_in_loop`
and order-0 entropy, which do, say the opposite.

**(b) Removing the copy primitive stops everything.** With `.` and `,` turned
into no-ops, memory after 50000 epochs is statistically indistinguishable from
its initial condition: H = 7.977 against 7.997 at epoch 0, high-order entropy
slightly *negative*, and **A(t) = 0 — not one 8-byte window ever became
persistent, in any seed**. Meanwhile the machine is working hard: 1950 steps
per run against the control's 128, and 95% of those steps re-running code. The
soup is computing furiously and producing nothing, because the only write left
is `+`/`-` on the single cell under head0, which cannot move information.

This is the cleanest control in the repository. It says the `<`/`,` crystal is
not an artefact of the ring's topology or of mutation pressure — it needs a
primitive that copies a byte from one place to another, and without one the
ring has no fixed point to fall into.

**(c) Insertions and deletions do not prevent crystallisation — they enrich
it.** All three seeds still collapse (mean steps 128.6–179.8, in-loop 0.00–0.32),
but to a *richer* state than the control: H rises from 1.28 to 2.48–2.85 and
high-order entropy sits at 0.64–1.00 instead of ~0.01. The reason is visible in
the composition: with indels, both mirror pairs coexist rather than one winning.
Seed 1 ends at `{` 0.42, `<` 0.35, `.` 0.08, `,` 0.06 — a two-domain state that
the control reached in only one seed of three.

The reading frame shifting appears to keep domain boundaries alive: an
insertion inside a `<` field displaces everything after it by one, which a
substitution cannot do, and that is enough to stop a single domain from
sweeping cleanly. It is still a crystal, though. Indels bought variety, not
computation.

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

Stated plainly, including the ones that were bugs and the one claim that was
simply wrong.

**Real bugs, found and fixed:**

1. **numba typed a SplitMix64 seed as int64 and sign-extended its shifts.**
   The compat stream then matched cubff for seed 1 and diverged for seed 7 —
   the worst kind of bug, silent and value-dependent. Caught only because the
   byte-exact test covered more than one seed. Fixed by casting at entry to
   `splitmix64`; `tests/test_compat.py` now runs two seeds per language.
2. **Our pairing reshuffled the previous epoch's permutation** where cubff
   rebuilds the identity permutation and shuffles that. Epoch 0 matched, epoch
   1 onward did not. Fixed, and the shuffle is reset explicitly with a comment
   saying why.
3. **The blocks-mode partner offset was wrong**: `d = 2` drew offsets
   {−2, −1, **+3, +4**} instead of {−2, −1, +1, +2}. Caught by a test written
   after the first blocks runs had started; those runs were discarded and
   redone. No blocks number in this document comes from the buggy version.
4. **`plot_kymograph` crashed on a run stopped by hand**, which has no final
   kymograph. Now skips with a message.

**A claim that was wrong:**

5. **"bff_noheads never transitions" was an artefact of ten runs.** It is in
   the git history of this file. Seed 103 transitions at epoch 2750. The
   corrected statement is that both languages transition and the heads variant
   transitions several times more often (p = 0.069 comparing the compat
   languages directly, 0.014 pooling the earlier runs).
6. **"Head confinement is load-bearing for the ring result" was also wrong**,
   and was corrected earlier in the same way: by implementing the alternative
   rule and measuring it rather than arguing about it.

**Things that looked like bugs and were not:**

7. **A single planted replicator usually goes extinct** (5 of 8 seeds). Founder
   stochasticity: it only copies when the pairing puts it in the first half, so
   its growth factor is ~1.5 per epoch. With 20 founders, takeover is reliable.
8. **Mean steps per run falling to 128.5 in the ring.** Real and explainable:
   129 steps is exactly a straight walk from `p` to the far edge, and the
   brackets have been purged. The new in-loop column makes this direct — 0.00
   of steps re-run an instruction.
9. **A(t) rising forever in a crystal** is the metric as specified, quantified
   above. The null filter reduces it 3–5× but does not eliminate it, because a
   clustered `,` field genuinely is non-i.i.d.
10. **High-order entropy ≈ 0 for a fully structured soup** is the metric
    behaving as defined. It is why H is now a required companion column.
11. **Negative high-order entropy** (−0.02 in the no-copy runs, −0.003 in one
    halt run) is not a compression failure: it is brotli spending a few bytes
    of header and failing to beat the order-0 model on incompressible data.

**Still open:**

12. **Whether locality would help a soup that can transition.** The blocks
    sweep ran our spec's head rule — the low-rate one — so it measures locality
    inside the non-transitioning regime. Blocks with `--compat cubff` head
    semantics is a one-flag change and was not run.
13. **Why the emergent quasispecies is not self-sufficient** against fresh
    random partners (mean motif change −0.24). It survives because its
    neighbours are relatives; whether that is a transient of the measurement
    epoch or a stable property was not established.

---

## Performance

Measured on 4 cores of an Intel Xeon @ 2.80 GHz (Linux 6.18, Python 3.11.15,
numpy 2.4.6, numba 0.67.0). Seeds are run as separate processes, so the
wall-clock figures below include contention between them, which was heavy for
most of this run set.

**The target was 50k epochs of the default ring config in under an hour. It
takes 125–284 seconds depending on how many other runs share the machine — so
between 13× and 29× inside the budget.**

| run | epochs | wall | epochs/s |
|---|---|---|---|
| ring (M=65536, R=128) | 50000 | 125–284 s | 176–399 |
| ring, no copy primitive | 50000 | 371–528 s | 95–135 |
| ring, with indels | 50000 | 146–195 s | 256–342 |
| blocks N=8192, d = 1 / 2 / 8 | 20000 | 264–485 s | 41–76 |
| bff N=1024 | 20000 | 57–64 s | 315–348 |
| bff N=8192 | 20000 | 351–364 s | 55–57 |
| bff N=32768 | 64000 | 2659–2668 s | 24 |
| bff `--compat cubff` N=32768 | 16000 | 833–1493 s | 6.4–14.1 |
| bff `--compat cubff_noheads` N=32768 | 16000 | 382–1092 s | 11.1–18.0 |

Ranges are wide because up to seven runs shared four cores, and because a run
that transitions becomes roughly twenty times more expensive per epoch — every
run then uses the full 8192-step budget instead of ~1000 steps. The no-copy
ring is slower than the control for the same reason in reverse: with no
crystal to fall into, it keeps executing ~1950 steps a run for all 50000
epochs.

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

## Run inventory

Every run directory holds `config.json`, `metrics.csv` (one row per snapshot),
`patterns.log` (top-10 16-byte windows per snapshot, printable and hex),
`summary.json` and raw `.npy` memory dumps. Runs stopped by hand carry
`stopped_early.json` instead of a summary, saying why and at which epoch.
Runs made before the metric changes also carry `rescored.csv`.

| directories | mode | N or M | variant | seeds | epochs |
|---|---|---|---|---|---|
| `bff_s*` | bff | 1024 | — | 3 | 20000 |
| `bff_long_s*` | bff | 1024 | — | 1 | 200000 |
| `bff_n32768_s*` | bff | 32768 | — | 3 | 64000 |
| `bff_n8192_s*` | bff | 8192 | — | 3 | 20000 |
| `bff_plant_s*` | bff | 1024 | 1 planted | 1 | 2000 |
| `bff_plant20_s*` | bff | 1024 | 20 planted | 1 | 400 |
| `bffnew_s*` | bff | 1024 | — | 3 | 20000 |
| `blocks_d1_s*` | blocks | 8192 | d=1 | 3 | 20000 |
| `blocks_d2_s*` | blocks | 8192 | d=2 | 5 | 20000 |
| `blocks_d8_s*` | blocks | 8192 | d=8 | 3 | 20000 |
| `compat_cubff_s*` | bff | 32768 | cubff | 17 | 64000 |
| `compat_noheads_s*` | bff | 32768 | cubff_noheads | 7 | 16000 |
| `ring_s*` | ring | 65536 | — | 3 | 50000 |
| `ring_halt_s*` | ring | 65536 | head-bound=halt | 3 | 50000 |
| `ringv_ctrl_s*` | ring | 65536 | — | 3 | 50000 |
| `ringv_indel_s*` | ring | 65536 | indel | 3 | 50000 |
| `ringv_nocopy_s*` | ring | 65536 | no-copy | 3 | 50000 |

The `compat_cubff_*` row shows the `epochs` field of its config, which was
64000 for the first five seeds before the budget was fixed at 16000. **Every
compat seed is scored over the same 16000-epoch window**: the later ones stop
there by configuration, the earlier ones were stopped there by the supervisor
in `stopped_early.json`, and no seed transitioned between 16000 and wherever
it was stopped. `bff_n32768_*` ran the full 64000 epochs and is reported as
its own row in the rate table, not pooled into the 16000-epoch comparison
except where the text says so explicitly.

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
# bit-exact cubff, the configuration that produces program-class takeovers
for s in 1 2 3 4 5; do
  python -m soup.run --config configs/bff_compat.json --compat cubff --seed $s \
                     --out runs/compat_cubff_s$s
done
# blocks locality sweep and the boundary-free ring variants
for s in 1 2 3; do
  python -m soup.run --config configs/blocks.json --d 2 --seed $s --out runs/blocks_d2_s$s
  python -m soup.run --config configs/ring.json --seed $s --no-copy 1 --out runs/ringv_nocopy_s$s
  python -m soup.run --config configs/ring.json --seed $s --indel 1   --out runs/ringv_indel_s$s
done
python -m soup.analyze --rates --finished-only runs/compat_*/
python -m soup.plots   runs/ring_s1 runs/ring_s2 runs/ring_s3 --out analysis/ring
python -m soup.analyze runs/ring_s1 runs/ring_s2 runs/ring_s3
```
