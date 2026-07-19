# GreenLedger — Research Roadmap

Where this project could go beyond the course: three research veins that would each
anchor a serious thesis chapter, ordered by how much evidence we already have in hand.
Everything below is grounded in checks actually run against the data on 2026-07-19 —
preliminary numbers are labeled as such, and every claim of novelty still requires a
proper related-work pass before it is asserted anywhere public.

---

## Vein 1 — Bunching at the MEES threshold: are ratings being gamed?

**The policy mechanism.** Since April 2018 (new lets) and April 2023 (all continuing
lets), the Minimum Energy Efficiency Standards (MEES) make it unlawful to let
commercial property in England & Wales rated **F or G**. That gives an owner — and the
assessor they hire — an enormous financial incentive for a building to score *just
inside E* (rating ≤ 125) rather than just inside F (≥ 126).

**Preliminary evidence, from our own 485k records (2023 file):**

| Boundary | Just below (5-pt window) | Just above | Ratio |
|---|---|---|---|
| C/D at 75 — no regulatory consequence | 9,393 | 6,120 | 1.5x |
| D/E at 100 — no regulatory consequence | 4,270 | 2,631 | 1.6x |
| **E/F at 125 — MEES letting ban** | 2,193 | 132 | **16.6x** |

The placebo boundaries behave like a smoothly declining density. The regulated boundary
does not — by an order of magnitude. This is the classic signature of **bunching at a
regulatory threshold** (the same phenomenon economists study at tax kinks and emissions
standards). Two readings, both policy-relevant: (a) assessors/owners nudge marginal
buildings across the line (measurement manipulation), or (b) owners make genuinely
threshold-targeted retrofits ("do exactly enough to reach E"). Distinguishing them is
the research.

**What a rigorous version needs:** McCrary-style counterfactual density estimation
(fit the smooth density excluding a window around 125, measure excess mass), the same
test year-by-year (does bunching intensify after the 2023 all-lets deadline?), placebo
years/boundaries formalized, and — if assessor IDs can be recovered from register
data — an assessor-level analysis (do some assessors bunch far more than others?).
Prior literature exists on *domestic* EPC bunching in the UK; the non-domestic /
small-business register is much less studied. **Run the related-work search before
claiming novelty.**

**Why it feeds back into the ML work:** labels near the threshold are partially
manipulated → any model trained on bands inherits gamed labels near E/F. That is a
label-noise story with a known, localizable cause — itself a publishable methods note.

## Vein 2 — What actually improves a rating? A repeat-certificate panel

**The design.** Certificates carry a `uprn` (unique property reference number). Checked
against just 5 of the 15 available year files: **29,154 buildings already have 2+
certificates; 6,057 have 3+.** Across all years, this yields a panel of tens of
thousands of *within-building* rating changes.

**The question prediction can't answer but a panel can:** when the same building is
re-certified, what changed — and which changes move the rating? Heating-fuel switches
(gas → grid electricity/heat pump) are directly observable between certificates of the
same building. Event-study / within-building fixed-effects design: rating change
regressed on observed changes (fuel, floor area, building environment), with
certificate-year effects absorbing the secular drift we already documented.

**Why this is the strongest vein scientifically:** it moves the project from
correlation to (cautious) causal inference on real policy-relevant actions, using only
data already on disk. It also grounds the app's recommendation engine in *our own
evidence* ("buildings like yours that switched from X to Y improved by Z bands, n=…")
instead of only citing Carbon Trust ranges.

**Honest limits:** re-certification is not random (buildings get re-certified when
sold/re-let — selection), and observed changes are coarse. Fixed effects and event-time
plots mitigate, not eliminate. Say so.

## Vein 3 — Screening under drift: deployment as a research problem

Two findings we already published in-repo become a third chapter:

1. **Secular drift** — mean rating improved 85.6 → 59.4 (2018→2026); a static model
   trained once will silently mis-calibrate within a few years.
2. **The accuracy trap** — model selection by accuracy picked a majority-class
   collapser (4% High-risk recall); selection by recall picked a 44x larger model.

**The research framing:**
- **Conformal prediction under drift.** Replace softmax confidence with conformal
  prediction sets (distribution-free coverage guarantees), then measure how coverage
  *decays* year-over-year as exchangeability breaks — and whether adaptive/weighted
  conformal repairs it. Current, publishable ML methodology with a rare asset: an
  8-year real-world drift testbed of 485k records.
- **Value-of-information for audits.** Formalize proxy-vs-audit as a decision problem:
  given audit cost £c and misclassification costs, which buildings should be sent for
  a real audit? Produces a triage curve (send top-k% most-uncertain to audit) — turns
  "cheap data is almost as good" into an operational screening policy.
- **The green-computing tradeoff, quantified:** recall-per-megabyte across the model
  family, honestly extending the tension we already disclosed (RF: best recall, worst
  footprint).

## What falls out for the app (practical layer)

- **"Pathway to E" planner** — for a building predicted High risk near the threshold,
  show which observable change moved *real comparable buildings* across (from Vein 2's
  panel), with n and observed rating deltas. Evidence, not advice.
- **Conformal risk sets in the UI** — "Low-or-Medium (90% coverage)" instead of a raw
  confidence bar (Vein 3).
- **Drift monitor tab** — model performance by certificate year, live; the model's own
  expiry date made visible.
- **Deprivation overlay** — join the Index of Multiple Deprivation (public,
  postcode-level) to the map: do deprived areas carry systematically worse
  small-business stock? (Equity/energy-poverty angle; also a possible fourth vein.)
- **Scotland replication** — Scotland runs a separate EPC register; replicating Veins
  1-2 there is an external-validity chapter and extends coverage the current app
  honestly lacks.

## What "doctorate level" actually requires (no shortcuts)

- **A real related-work pass per vein** (the SemanticPQC discipline): domestic-EPC
  bunching literature, building-retrofit evaluation literature, conformal-under-shift
  literature. Novelty is claimed only for the specific unclaimed intersection.
- **Statistical rigor:** counterfactual densities and standard errors (Vein 1),
  fixed-effects with clustered SEs and event-study plots (Vein 2), multi-seed +
  coverage guarantees (Vein 3). No single-run headline numbers.
- **A written paper per vein.** Venues to aim at: *Energy Policy* / *Energy Economics*
  (Vein 1), *Energy and Buildings* / *Applied Energy* (Vein 2), an ML venue's
  datasets/distribution-shift track (Vein 3).
- **Honest scope:** executed at course scale, Veins 1+3 make this a master's-thesis-
  grade project. All three, with the related-work discipline, replication, and real
  reviewer scrutiny, is the shape of a doctoral program — not a semester.

## Recommended order

1. **Vein 1 first** — the preliminary signal is already strong, the McCrary test is a
   week of careful work on data in hand, and the result is striking either way.
2. **Vein 2 second** — build the UPRN panel (all 15 year-files), event-study on fuel
   switches; feeds the app's "Pathway to E."
3. **Vein 3 third** — conformal + drift analysis; upgrades the app's uncertainty UI.
4. Related-work pass runs alongside each, *before* any external claim.
