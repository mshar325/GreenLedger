# Vein 1 — Related Work: EPC Threshold Bunching (verified 2026-07-19)

Every citation below was verified by search against live sources on the date above.
One citation supplied to us — "McCrone et al., 2018" — **could not be verified and
appears not to exist**; searches for it resolve to Collins & Curtis (2018), which is
the actual residential anchor. Do not cite McCrone anywhere.

## A. Direct anchors: EPC bunching (residential only)

1. **Collins, M. & Curtis, J. (2018).** "Bunching of residential building energy
   performance certificates at threshold values." *Applied Energy* 211, 662-676.
   - Irish domestic BER/EPC register; 15 grades = 14 notches on a continuous scale.
   - Method: regression discontinuity design + estimated counterfactual distribution.
   - Finding: bunching on the favourable side of grade boundaries in **post-retrofit**
     certificates but **not pre-retrofit** ones; "widespread but not systemic";
     discontinuities in specific assessment input parameters.
   - Relevance: the closest methodological template to ours; establishes that EPC
     bunching is measurable with counterfactual-density methods, in a residential
     register, with retrofit-incentive (grant) motives rather than a letting ban.

2. **Hardy, A. & Glew, D. (2019).** "An analysis of errors in the Energy Performance
   certificate database." *Energy Policy* 129, 1168-1178.
   - England & Wales **domestic** register: 27% of EPCs carry ≥1 error flag; estimated
     true error rate 36-62%; ratings typically shift ~4 points under error correction.
   - Relevance: measurement-noise context. A 4-point typical error is material when
     the E/F boundary decision sits within ±5 points — errors and gaming are
     observationally adjacent, and our design must discuss both.

3. **Crawley, J. et al. (2019).** "Quantifying the Measurement Error on England and
   Wales EPC Ratings." *Energies* 12(18), 3523.
   - Quantifies EPC rating measurement error for England & Wales (domestic).
   - Relevance: supports modelling the rating as score = signal + noise; genuine
     random error alone produces *smooth* density, not one-sided excess mass — which
     is why the placebo-boundary contrast carries the identification.

4. Industry/policy commentary (Energy UK consultation responses, 2024-2026) explicitly
   flags "gaming risk" around PRS/MEES compliance — concern exists in the policy
   conversation, but we found **no academic study quantifying it for non-domestic
   stock**.

## B. Methodological canon (bunching / density discontinuity)

5. **McCrary, J. (2008).** "Manipulation of the running variable in the regression
   discontinuity design: A density test." *Journal of Econometrics* 142(2), 698-714.
   — The original density-discontinuity test; our robustness check.
6. **Chetty, R., Friedman, J., Olsen, T. & Pistaferri, L. (2011).** "Adjustment costs,
   firm responses, and micro vs. macro labor supply elasticities." *QJE* 126(2).
   — The polynomial-counterfactual bunching estimator (fit binned counts excluding a
   window around the kink; excess mass vs. counterfactual; bootstrap SEs). Our primary
   estimator follows this design.
7. **Kleven, H. & Waseem, M. (2013).** "Using notches to uncover optimization frictions
   and structural elasticities." *QJE* 128(2).
   — Notches (discrete jumps in incentives) vs. kinks: MEES is a **notch** — an F
   rating makes letting unlawful outright — so we expect excess mass just inside E
   *and a hole* just inside F, exactly the Kleven-Waseem signature.
8. **Kleven, H. (2016).** "Bunching." *Annual Review of Economics* 8, 435-464.
   — Survey; standard reference for estimator variants and pitfalls (round-number
   heaping, excluded-window selection, integration constraints).
9. **Cattaneo, M., Jansson, M. & Ma, X. (2020).** "Simple local polynomial density
   estimators." *JASA* 115(531). — `rddensity`-style local-polynomial density test;
   planned second robustness check.

## C. Policy facts (verified against current legal commentary)

- **MEES, non-domestic England & Wales:** unlawful to grant a **new** lease on an
  EPC F/G building from **1 April 2018**; extended to **all continuing leases from
  1 April 2023** (exemptions registrable). The compliance target is E (asset rating
  ≤ 125 on the non-domestic scale; F begins at 126).
- **June 2026 government announcement:** from **2031**, commercial buildings
  **>1,000 m² must reach EPC B**; buildings **below 1,000 m² stay at minimum E**.
  Two implications for us: (a) our ≤500 m² small-business population remains governed
  by the E/F threshold — the bunching margin we study stays policy-live; (b) a sharp
  out-of-sample prediction: bunching should emerge at the **B/C boundary for large
  buildings** as 2031 approaches — a falsifiable forecast the paper can register now.

## D. The gap (stated honestly)

Across all searches: EPC bunching has been quantified for **residential** registers
(Ireland: Collins & Curtis 2018; England & Wales domestic data-quality: Hardy & Glew
2019, Crawley et al. 2019), and MEES gaming risk is discussed in industry commentary —
but **no study we could find applies a counterfactual-density bunching design to the
England & Wales non-domestic register, none ties bunching to the MEES E/F letting
ban, and none exploits the two enforcement steps (April 2018 new lets; April 2023 all
lets) as within-register natural experiments.** Our preliminary evidence (placebo
boundaries ~1.5-1.6x vs 16.6x at E/F, 2023 file) sits squarely in that gap.

Scope discipline: claim "no study surfaced in our searches," not "no study exists,"
until a full systematic pass (Scopus/WoS) is run — same rule as SemanticPQC.

## E. Identification threats to carry into the paper

1. **Error vs. gaming:** random measurement error (Hardy & Glew) smooths density;
   it cannot generate one-sided excess mass at exactly one policy boundary while
   placebo boundaries stay smooth. But *asymmetric* error-correction behaviour
   (owners only re-commission an assessment when the first result is F) is a form of
   selective re-testing that produces bunching without input manipulation — this is
   partially distinguishable via repeat-certificate (UPRN) analysis: look for F→E
   re-certifications at short intervals.
2. **Genuine threshold-targeted retrofit** ("do exactly enough to reach E") is
   rational compliance, not fraud; both produce excess mass. The year-over-year
   design (2018 step vs 2023 step), time-to-relet analysis, and input-parameter
   discontinuities (Collins & Curtis's approach) help separate stories.
3. **Software discreteness:** SBEM (the non-domestic calculation engine) may produce
   lumpy ratings; heaping dummies + placebo boundaries address generic lumpiness.
