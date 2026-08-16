# Citations

References backing the design choices in this repo.

## Localization (src/localizer.py)

- Lewis, J.P. (1995), "Fast Normalized Cross-Correlation," Vision Interface.
  The normalized cross correlation used for matching, and the running-sum /
  integral-image formulation behind building all 100 phase templates in one
  pass instead of resizing the reference 100 times.
- Tian, Q., Huhns, M.N. (1986), "Algorithms for subpixel registration,"
  Computer Vision, Graphics, and Image Processing, 35(2):220-233. Basis for
  the parabolic fit on the correlation peak used by both distortion
  estimators.
- Guizar-Sicairos, M., Thurman, S.T., Fienup, J.R. (2008), "Efficient
  subpixel image registration algorithms," Optics Letters, 33(2):156-158.
  Standard reference for sub-pixel registration by upsampled correlation
  (skimage.registration.phase_cross_correlation implements this). Not
  needed for the main exact-scale-10 method, which recovers the sub-pixel
  offset exactly from the downsample geometry instead -- but it is used
  in the scale-robustness fallback (src/localizer.py _scale_fallback),
  where scale isn't fixed at exactly 10 and there's no equivalent exact
  arithmetic available.
- Turin, G.L. (1960), "An introduction to matched filters," IRE Transactions
  on Information Theory, 6(3):311-329. The matched-filter argument behind
  setting the horizontal match blur from the measured row jitter: scan
  jitter smears the search image in x only, so the template is smeared to
  match rather than left sharp.

## Dataset generation

Dataset generation (src/pipeline.py, src/presets.py, src/sem_imaging.py,
src/structural_defects.py, src/patterns/) is adapted from the Applied
Materials Drift-Sense starter resource, not written from scratch --
see README.md for why. The citations below back the physical choices it
makes, whether we wrote that code or adapted it.

- aayushraina21, "Drift-Sense Synthetic Dataset Generator,"
  huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data. Listed
  on the SEMICON India Hackathon 2026 page (i4c.in/hackathon-2026) as the
  Applied Materials starter resource for this problem statement. The
  source of src/pipeline.py, src/presets.py, src/sem_imaging.py,
  src/structural_defects.py and src/patterns/, and of the reference/search
  scale convention (1000x1000px, nominal 10:1 magnification) and
  --reference/--search CLI shape used throughout this repo.

**Sensor noise.** src/sem_imaging.py models shot noise (Poisson) plus
detector noise (Gaussian) as separate terms:
- Foi, A., Trimeche, M., Katkovnik, V., Egiazarian, K. (2008), "Practical
  Poissonian-Gaussian Noise Modeling and Fitting for Single-Image
  Raw-Data," IEEE Trans. Image Processing, 17(10):1737-1754. Electron
  detectors are shot-noise dominated, not uniform-Gaussian like a
  consumer camera sensor, which is why the two noise terms are modeled
  and summed separately rather than as one flat noise level.

**Beam optics and astigmatism.** src/sem_imaging.py's beam-spot blur and
per-axis astigmatism:
- Reimer, L., "Scanning Electron Microscopy: Physics of Image Formation
  and Microanalysis," Springer. Finite probe diameter and uncorrected
  stigmator behavior (different focus diameter along orthogonal axes)
  are standard SEM imaging effects, not an invented degradation.

**DRAM structure.** Word-line pitch and bit-line pitch are modeled as
distinct (2F and 3F), not one shared pitch:
- US Patent 7,349,232 B2, "6F2 DRAM cell design with 3F-pitch folded
  digitline sense amplifier" -- states the cell as a 3F bitline pitch by
  2F wordline pitch rectangle, area 3F x 2F = 6F^2.
- Kim, K., Hwang, C.G., Lee, J.G. (1998), "DRAM technology perspective
  for gigabit era," IEEE Trans. Electron Devices, 45(3):598-608. Same
  6F^2 folded-bitline cell, corroborating the patent above.

**FinFET structure.** Fin pitch and contacted gate pitch are modeled as
distinct values, not one shared pitch:
- Auth, C. et al. (2012), "A 22nm high performance and low-power CMOS
  technology featuring fully-depleted tri-gate transistors...," Symp.
  VLSI Technology.
- Natarajan, S. et al. (2014), "A 14nm logic technology featuring
  2nd-generation FinFET transistors...," IEDM Tech. Dig.
- Auth, C. et al. (2017), "A 10nm high performance and low-power CMOS
  technology featuring 3rd generation FinFET transistors...," IEDM Tech.
  Dig.

These three establish that fin pitch and contacted gate pitch are
genuinely different values at real nodes, and give the real scaling
trend. The starter resource's own preset values are deliberately
coarsened for on-screen visibility (its own source comments say so, and
its FINFET_10NM preset does not use the literal 34nm fin pitch from the
2017 paper) -- these papers back the structural relationship the presets
follow, not an exact match to any one preset's numbers.

## Problem statement source

- Applied Materials, "Drift-Sense: AI-Powered Navigation-Error Recovery
  for Wafer Inspection Tools," problem statement document, SEMICON India
  Hackathon 2026 (i4c.in/hackathon-2026). Source for the closest-to-centre
  selection rule, the required accuracy thresholds, and the required
  results breakdowns, all referenced in README.md and in the code.
