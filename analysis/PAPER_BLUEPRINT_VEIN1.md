# Paper Blueprint (IEEE format) — Vein 1

**Working title:** "Bunching at the Ban: Regulatory Threshold Effects in England &
Wales Non-Domestic Energy Performance Certificates"

**Target length:** 8-10 pages IEEE two-column. Course-compatible; journal variant
(Energy Policy / Energy Economics) would expand Sections V-VII.

---

- **Abstract** (150-200 words): policy notch → preliminary magnitude (placebo ~1.5x
  vs 16.6x at E/F) → estimator → year-by-year finding → what it implies for
  EPC-based ML and for MEES enforcement design.

- **I. Introduction.** MEES makes an F rating a letting ban — a *notch*, not a kink.
  Owners of marginal buildings gain enormously from a rating of 125 vs 126. Question:
  does the certificate distribution show the signature of threshold targeting, and
  did it intensify with the 2018 (new lets) and 2023 (all lets) enforcement steps?
  Contributions: (1) first counterfactual-density bunching analysis of the E&W
  **non-domestic** register we are aware of; (2) exploitation of two enforcement
  steps + pre-2015 announcement period as within-register natural experiments;
  (3) implications for ML systems trained on rating bands (label distortion is
  concentrated exactly at the policy boundary); (4) a registered, falsifiable
  forecast: bunching at B/C for >1,000 m² stock as the announced 2031 standard nears.

- **II. Related Work.** Residential EPC bunching (Collins & Curtis 2018 — Ireland,
  RDD + counterfactual, post-retrofit bunching only); E&W EPC data quality (Hardy &
  Glew 2019: 27% flagged, ~4-point typical error; Crawley et al. 2019 measurement
  error); bunching methodology canon (McCrary 2008; Chetty et al. 2011; Kleven &
  Waseem 2013; Kleven 2016; Cattaneo et al. 2020). Gap paragraph per
  RELATED_WORK_VEIN1.md — no non-domestic/MEES bunching study surfaced.

- **III. Policy Background.** Non-domestic EPC scale and bands (A+…G; lower =
  better; E ends at 125, F begins at 126); SBEM assessment; MEES timeline: 2015
  regulations, 1 Apr 2018 new lets, 1 Apr 2023 all lets; exemptions register; the
  June 2026 announcement (≥B by 2031 for >1,000 m²; E retained below 1,000 m²).

- **IV. Data.** E&W non-domestic EPC register 2012-2026 (~1.4M certificates;
  ~485k in our ≤500 m² small-business subset), integer asset ratings; descriptive
  density plots; the three boundaries studied; disclosure of register error rates
  as measurement context.

- **V. Methodology.** The estimator exactly as implemented in
  `analysis/vein1_bunching.py`:
  binned counts, degree-4 polynomial in centered rating, excluded window
  Z = [121,130], heaping dummies (multiples of 5/10 outside Z), bin dummies inside
  Z; counterfactual = polynomial + heaping; excess mass B (bins 121-125), missing
  mass M (126-130), normalized b = B / mean counterfactual (Chetty); parametric
  residual bootstrap (500 reps, percentile CIs). Notch logic (Kleven-Waseem): expect
  B > 0 *and* M > 0 with M/B near 1 under relabeling. Placebos: identical spec at
  75 and 50. Robustness: polynomial degree {3,4,5}, Z width {±3,±5,±7}, local
  polynomial density (Cattaneo et al.) as alternative estimator.

- **VI. Results.**
  (a) Pooled MEES-era estimate with observed-vs-counterfactual figure;
  (b) year-by-year b with CIs, MEES vs placebo boundaries (the core figure);
  (c) event alignment: change at 2018 and 2023 steps; pre-2015 baseline;
  (d) M/B mass accounting — how much of the excess is drawn from just-F;
  (e) small-business (≤500 m²) vs full-register comparison.

- **VII. Mechanisms & Threats to Validity.** Manipulation vs threshold-targeted
  retrofit vs selective re-assessment (only-retest-if-F). Partial separation via:
  UPRN repeat-certificate analysis (short-interval F→E re-certifications), input-
  parameter discontinuities (Collins & Curtis approach), heaping/SBEM lumpiness
  placebo logic. State plainly which stories the data cannot separate.

- **VIII. Implications.**
  (1) Policy: excess mass ≈ buildings whose true standing is F presented as E;
  enforcement/audit targeting just-above-threshold certificates; forecast for the
  2031 B-threshold.
  (2) ML: band labels near the boundary are endogenous to the policy — models
  (including ours) inherit concentrated label distortion; propose threshold-aware
  label handling.

- **IX. Conclusion.** One page: finding, scope limits (E&W only; observational;
  cannot fully separate compliance from gaming), registered forecast.

- **Reproducibility note.** Code + register download instructions public in the
  GreenLedger repo; every figure regenerable from `vein1_bunching.py`.
