# Drift-Sense

SEMICON India Hackathon 2026, Applied Materials Problem Statement 2.

Given a close-up reference image of a spot on a wafer and a wider search
image of the area around it, find where the reference sits in the search
image and return its centre coordinates. DRAM and FinFET chip layouts are
made of repeating patterns, so a plain template match often finds several
places that all look right. Most of the actual work here is telling the
correct one apart from its look-alikes.

## Results

30 pairs per seed, generated with the Applied Materials starter resource
(see Dataset below). The localizer only ever sees the two image files -- no
answers are read from the data. Headline numbers are over **four
independent seeds, 120 pairs total**, since tuning on one seed and then
reporting on that same seed doesn't really prove anything.

| Metric | 120 pairs, 4 seeds | Single seed (42) |
|---|---|---|
| within 5 px | 94.2% | 100% |
| within 4 px | 94.2% | 100% |
| within 2 px | 94.2% | 100% |
| within 1 px | 90.0% | 100% |
| within 0.5 px | 80.0% | 86.7% |
| median error | 0.17 px | 0.11 px |
| mean runtime | 1.41 s/pair | 1.39 s/pair |

(Runtime went up from an earlier 0.56s/pair -- see "Robustness" below for
why; accuracy on this exact-scale-10, no-rotation data is unaffected,
re-verified on all four seeds after the change.)

Every pair is scored under one of six scan conditions (default, clean, low
dose, heavy drift, speckle/salt-pepper, charging), cycling through the
generator's own presets. On seed 42, all six conditions land 100% within
1 px. `results/metrics.json` has the full breakdown by condition,
architecture, dose, drift amount, and target position.

The misses are real and worth being upfront about: 7 of 120 pairs miss by
more than 5px, and 6 of those are FinFET. They also miss with high
confidence -- the match scores were 0.86 to 0.98, not weak or borderline.
A FinFET area is just parallel fin lines broken up by gate bars, so there
are simply more spots that look genuinely identical than in DRAM's 2D grid
of dots. Checked one directly: the wrong spot the tool picked scored
0.8657, and the correct spot scored 0.8193. That's not a search problem --
the image itself doesn't have enough unique detail there to tell them apart.

## How it works

The short version: we looked at exactly how the test images are built,
and used that instead of guessing. The generator draws a huge, very
detailed canvas, crops a small 1000x1000 patch out of it as the reference,
and shrinks the *same* canvas down by exactly 10x to make the search
image. So the scale is always exactly 10, and there's no rotation, in
*this specific generator's output* -- both of which we used to assume
we'd have to search for. The four steps below are the core method, built
around that exact case; "Robustness" further down covers what we added
on top once we checked what happens when that assumption doesn't hold.

Four things make up the core method:

**1. Getting the pixel grids to line up.** When you shrink an image down
by 10x, you're averaging each 10x10 block of pixels into one. If the
reference is cropped starting at some random pixel, the "10x10 blocks"
used to shrink it won't necessarily line up with the blocks the search
image was built from -- so a plain shrink-and-compare gets a slightly
blurred, misaligned version and does worse than it should, especially on
fine detail. Fix: try shrinking the reference starting from all 100
possible pixel offsets, and see which one matches best. This one change
took a plain match from finding 21 of 30 pairs correctly to 30 of 30.

Whichever starting offset wins also tells us the exact position, for
free, with no extra step -- it falls straight out of the arithmetic of
which offset was used. That's why the vertical position came out exactly
right, down to 0.000px, on every one of the first 30 test pairs.

**2. Correcting for scan drift.** The search image's rows can be nudged
sideways a little more as you go down the image, the way a scan can drift
slightly over time. That's too small to notice by looking at just the
small reference patch, so instead we look at the strip regions between
memory blocks -- the only feature that runs the entire height of the
image -- and measure how much they visually lean. Fixing this cut the
average error from 0.80px down to 0.18px.

**3. Blurring to match scan shakiness.** On top of steady drift, a scan
can also wobble row to row at random. That blurs the image sideways, and
badly enough that it can throw the match off if left uncorrected. So we
measure how shaky the scan looks (by comparing each row to the next one)
and blur our comparison template by roughly that much before matching --
matching blur to blur gives a fairer comparison than comparing a sharp
template to a shaky image.

**4. Breaking ties by picking the more central match.** The problem
statement's own rule: if a few spots score almost the same, go with
whichever is closest to the centre of the search image. We only treat
scores as "tied" if they're extremely close (within 0.2%), since a wrong
spot in a repeating pattern usually scores a real 1-3% lower than the
right one -- treating that as a tie and picking on position alone would
throw away information the score already has. Loosening this to a 1% tie
band drops accuracy from 94.2% to 73.3%, which is why it's kept tight.

One honest note: this rule actually costs about 0.8% accuracy on our own
test data, because our generator places targets pretty much anywhere,
not mostly near the centre. We kept it anyway because the problem
statement requires it, and because it makes sense for what the tool is
actually meant to do -- recover from a small drift, where the real answer
usually is near where it landed.

## Robustness: what happens off the exact case

Everything above assumes the search image is *exactly* a 10x shrink of
the reference with *zero* rotation, because that's genuinely how this
particular generator builds every pair. But the problem statement itself
says real testing may use scale anywhere from 9:1 to 11:1 and rotation of
1-2 degrees -- and the generator we're using doesn't actually have a way
to produce that kind of data to test against directly, so we checked it
the honest way: took the generator's real output images and applied a
known, controlled scale change or rotation to them ourselves, then
checked how far off our answer landed from the (correspondingly shifted)
true position. That's an approximation of what a genuinely different
scale or rotation would look like, not a perfect stand-in for it, but
it's far better than assuming the method would just work.

**It didn't, at first.** Just 0.5 degrees of rotation -- well inside the
stated range -- dropped accuracy from 100% to 60% at 5px. A 3% scale
error did something similar. That's a real, serious gap: a method that's
essentially perfect in one exact condition and falls apart just outside
it isn't something you'd actually want running a real tool.

So we added two things on top of the exact method above, not instead of
it:

- **A small rotation check, always run.** Before doing the precise phase
  search, we quickly try a handful of small rotation angles (-2 to +2
  degrees, half-degree steps) and use whichever one matches best, then
  run the exact method as before using that corrected reference. We
  checked whether this could instead be triggered only when needed (by a
  low match score, the way the scale check below works) and it doesn't
  work: a mismatched rotation doesn't reliably score any lower than a
  correct one, so skipping this when "it looks fine" would have missed
  real errors silently.
- **A scale check, also always run.** We try a spread of scale ratios (9
  to 11) and keep whichever answer -- the exact method's, or this
  fallback's -- scores better. This one's slower (about 0.7 extra seconds
  per pair) and, unlike the rotation check, mostly does help when
  triggered by a weak score, but we measured that relying on the score
  alone still missed real mismatches sometimes, so it just always runs
  instead of waiting to be asked.

The first version of the scale check just kept its single best match --
and immediately lost to periodic decoys, the exact same mistake this
project already made and fixed once before with the main method. Fixed
the same way here too: keep several candidates, not one, and use the
same tight-tolerance closest-to-centre rule to choose between them.

**Where this leaves us, honestly, on the (approximated) 9:1-11:1 / 1-2
degree range:**

| Condition | within 5px | within 1px |
|---|---|---|
| exact (baseline) | 100% | 100% |
| scale off by 5% | 63-70% | 53-57% |
| rotation off by 1-2 deg | 27-40% | 0-3% |

Scale recovers reasonably well. Rotation still doesn't, not fully --
correcting *to* the right discretized angle candidate helps a lot (was
13-40%, now 27-40%), but rotating an image at all means resampling it,
which blurs it slightly and quietly breaks the exact-arithmetic
guarantee the main method depends on, even once the right angle is
found. Getting rotation genuinely solved would need a different
sub-pixel step after the rotation correction, not just picking a better
discrete angle -- flagged honestly as unfinished rather than papered
over. See "Known limitations" for the rest of this.

## Dataset

Built using the Applied Materials starter resource for this exact problem
statement (huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data),
which the official problem statement links as their recommended tool --
so we use it directly instead of writing our own generator from scratch.
`src/pipeline.py`, `src/presets.py`, `src/sem_imaging.py`,
`src/structural_defects.py` and `src/patterns/` are theirs.
`generate_dataset.py` is a thin wrapper around it that saves the output in
the flat file layout our other scripts expect, and adds a
`--vary-conditions` option so one run can cover several scan conditions
instead of just one. See `references/citations.md` for the published
sources behind the structures and the noise model.

## Layout

```
  references/citations.md    sources for the structures and noise model
  src/pipeline.py            dataset generation (starter resource)
  src/presets.py             DRAM/FinFET structure presets (starter resource)
  src/sem_imaging.py         SEM noise and optics (starter resource)
  src/structural_defects.py  pattern-collapse defects (starter resource)
  src/patterns/              DRAM/FinFET/zone rendering (starter resource)
  src/localizer.py           the localization method (ours)
  src/metrics.py             accuracy and runtime summaries
  src/visualize.py           overlay, error distribution, failure figures
  generate_dataset.py        CLI: build a dataset
  localize.py                CLI: the graded entry point
  evaluate.py                CLI: score predictions against a manifest
  tests/                     35 tests
```

## Setup

```
pip install -r requirements.txt
```

## Run

Single pair -- this is the graded interface, two file paths in, one line out:

```
python localize.py --reference path/to/reference.png --search path/to/search.png
```

Prints `x.xx,y.yy`. Coordinates are in search-image pixels, top-left is
(0, 0), x grows to the right and y grows downward.

To reproduce the single-seed numbers above:

```
python generate_dataset.py --num-samples 30 --output-dir results/dataset --seed 42 --vary-conditions
python localize.py --manifest results/dataset/manifest.csv --out results/predictions.csv
python evaluate.py --manifest results/dataset/manifest.csv --predictions results/predictions.csv --out-dir results
```

For the four-seed numbers, repeat with `--seed 777`, `--seed 20260101` and
`--seed 5` into separate output folders.

Tests:

```
python -m pytest tests/ -q
```

## Known limitations

- About 6% of pairs miss by more than 5px, almost all FinFET, and they miss
  with high confidence. The wrong spot genuinely scores better than the
  right one there, so this comes from the image itself, not from something
  a smarter search would fix. See `results/failure_worst_pair.png` for an
  example.
- The drift correction needs the strip boundaries between memory blocks to
  actually be visible, since that's what makes a sideways lean measurable
  in the first place. On a search image that was one uniform pattern edge
  to edge, it would do worse -- but it fails safe rather than making things
  worse: it reports close to zero drift instead of guessing, and there's a
  test that checks this.
- Scale and rotation robustness (see "Robustness" above) is real but
  partial. Scale mismatches recover to roughly 63-70% at 5px; rotation
  mismatches only reach 27-40%, because rotating an image at all
  resamples it and quietly costs some of the exact-arithmetic precision
  the main method relies on, even once the right angle is identified.
  This was checked with controlled perturbations of the generator's real
  output, not a generator that can natively produce scale/rotation
  variation, so treat it as a reasonable estimate, not a guarantee.
- The always-on rotation and scale checks add real runtime: about
  1.4s/pair now, up from 0.56s/pair when the method only handled the
  exact case. Worth it since a gated version was silently trusting wrong
  answers, but it's a real cost, not free.
- All numbers above are on synthetic data from the generator linked above,
  not the organizers' own held-out test set, which they've said will be
  noisier.
