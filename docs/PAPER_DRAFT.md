# Bunching at the Ban: Regulatory Threshold Distortion in England & Wales Non-Domestic Energy Performance Certificates, and Implications for Machine-Learning Risk Screening

*Draft — IEEE format. Author(s): [names]. Affiliation: [institution]. SDG GreenTech 2026.*

---

## Abstract

Minimum Energy Efficiency Standards (MEES) make it unlawful to let a commercial building
in England & Wales rated F or G. Using the full public non-domestic Energy Performance
Certificate (EPC) register (≈1.4M certificates, 2012–2026), we show that the certificate
distribution exhibits strong bunching precisely at the legal E/F boundary (asset rating
125/126). A counterfactual-density (bunching) estimator yields normalized excess mass
b ≈ 0.02 before the policy, rising through the announcement window and plateauing at
b ≈ 3.1–3.9 under enforcement, while placebo boundaries remain flat; the excess mass at
"just-E" is drawn from "just-F" (missing/excess mass ratio ≈ 0.8), the notch-relabeling
signature. A repeat-certificate panel (213k buildings) shows failing buildings escape to
compliance at 96–99% post-enforcement, most via substantial genuine improvement but a
small fast-returning segment via no observable change. We then show this distortion
matters for the machine-learning tools — commercial and academic — trained on this
register: a neural network attains the highest accuracy while catching 4.4% of genuinely
high-risk buildings, so we select models on high-risk recall (Random Forest, 0.699
[0.685, 0.713]) and build an uncertainty-ranked audit-triage tool whose reliability we
track with conformal prediction under drift. Contributions: (1) the first
counterfactual-density bunching analysis, to our knowledge, of the E&W non-domestic
register tied to the MEES notch; (2) a fitness-for-purpose model-selection argument
(recall over accuracy) with confidence intervals; (3) a reproducible, fully open
alternative to opaque commercial EPC estimators.

**Index terms** — energy performance certificates, MEES, bunching estimator, regulatory
manipulation, machine learning, model selection, conformal prediction, algorithmic
fairness.

---

## I. Introduction

The EPC asset rating is the standard building-level proxy for energy/carbon performance in
UK policy, and MEES turns it into a legal notch: from April 2018 (new lets) and April 2023
(all continuing lets), letting an F/G-rated (rating ≥ 126) non-domestic building is
unlawful. Owners of marginal buildings therefore have a large incentive for a rating just
inside E (≤ 125) rather than just inside F.

We ask two linked questions. **(Q1)** Does the certificate register show the distributional
signature of a behavioral response at the E/F boundary, and did it track the enforcement
timeline? **(Q2)** Given that machine-learning tools — commercial estimators and academic
classifiers — are trained on this same register, what does that distortion, and the class
imbalance it interacts with, imply for how such models should be built and selected?

Contributions:
1. A counterfactual-density bunching analysis of the E&W **non-domestic** register tied to
   MEES, with a robustness battery (specification sweep, boundary permutation,
   local-linear estimator, pre-policy temporal placebo) — a combination we did not find in
   prior work.
2. A repeat-certificate panel and building-fixed-effects analysis separating genuine
   improvement from the manipulation-signature minority.
3. A machine-learning contribution: fitness-for-purpose model selection (high-risk recall,
   with bootstrap confidence intervals), an uncertainty-ranked triage tool, conformal
   coverage-decay diagnostics under drift, and a fairness audit — all reproducible and
   open, unlike the opaque commercial tools in this space.

## II. Related Work

**EPC bunching.** Collins & Curtis [1] document bunching in the Irish *residential* BER
register using regression discontinuity and counterfactual densities, finding threshold
effects in post-retrofit certificates. EPC data-quality/error studies for E&W domestic
stock [2], [3] establish material measurement error (≈27% flagged; ~4-rating-point typical
error). We are not aware of a counterfactual-density bunching analysis of the E&W
*non-domestic* register tied to the MEES notch.

**Bunching methodology.** We follow the density-manipulation and bunching canon [4]–[7]:
polynomial counterfactuals with excluded windows [5], the notch (vs kink) distinction [6]
predicting excess mass with a matching hole, and density-discontinuity tests [4], [8].

**ML for EPC prediction.** Commercial estimators (e.g. GPT-based London commercial tools)
and academic MLP classifiers on the Irish register [9] predict ratings from building
inputs; the latter explicitly notes register class imbalance. None, to our knowledge,
combine prediction with register-distortion evidence, adopt a recall-first selection
criterion, or release reproducible metrics.

## III. Policy & Data

MEES timeline and bands (A+…G; E ends at 125, F begins at 126); the June 2026 announcement
of a ≥ B standard for > 1,000 m² buildings by 2031 with E retained below. Data: the public
non-domestic EPC register (epc.opendatacommunities.org), 2012–2026; filtered to
small-business activity types (retail/financial/professional, restaurant/cafe,
office/workshop) under 500 m² for the ML tasks (485k buildings), full register for the
bunching analysis.

## IV. Methods

**A. Bunching estimator.** Binned certificate counts on [101,150]; degree-4 polynomial in
centered rating with an excluded window [121,130] and round-number heaping dummies;
counterfactual = polynomial + heaping; excess mass B over 121–125, missing mass M over
126–130, normalized b = B / mean counterfactual; residual bootstrap CIs. Placebo
boundaries at 75 and 50. Robustness: degree {3,4,5} × window {±3,±5,±7}; a boundary
permutation over all integer thresholds; a local-linear counterfactual.

**B. Panel.** UPRN-linked consecutive certificate pairs; escape = F/G → ≤125; landing
distribution, timing, observable-change channels; building fixed-effects OLS (within
transformation, year FE, clustered SEs).

**C. ML.** Proxy vs proxy+audit feature sets; temporal holdout (train 2018–2024, test
2025); class-balanced Logistic Regression / Random Forest / XGBoost / MLP; selection by
high-risk recall; bootstrap + multi-seed CIs; predictive-entropy triage; split-conformal
coverage across years; group fairness audit.

## V. Results

**A. Bunching (Q1).** Normalized excess mass at the E/F boundary: ~0.02 (2012), rising
through 2015–2017 to plateau at 3.1–3.9 under enforcement; placebo boundaries flat
(median boundary b ≈ 0.04; 125 exceeds ~98% of placebo boundaries). M/B ≈ 0.7–0.9 — the
relabeling signature. Robust across all nine specifications (b ∈ [2.12, 3.50]) and under a
local-linear estimator (b ∈ [2.30, 2.70]). Pre-policy temporal absence rules out SBEM
lumpiness / round-number heaping.

**B. Mechanism (panel).** Post-enforcement escape rate 96–99%; 75% of escapes reach D or
better (genuine improvement); escapes ~2.2× over-represented at just-E; a fast-return
segment (≤90 days) shows 40% no-observable-change and ~4.4× just-E landing (the
manipulation-signature minority). Building fixed effects: heating fuel materially moves the
rating within-building (oil +46.5 [32.8, 60.1] vs the low-carbon baseline).

**C. ML (Q2).** Accuracy-optimal model (MLP, 0.681) attains 0.044 high-risk recall;
recall-optimal Random Forest attains 0.699 [0.685, 0.713], a CI disjoint from the MLP's
[0.038, 0.050] and XGBoost's [0.569, 0.599]. Audit triage: top-5% uncertainty finds
high-risk at 13.8% vs 7.4% random (1.86× lift). Conformal coverage decays 90.0% → 85.5%
across 2022 → 2026, an explicit recalibration signal. Fairness: Office/Workshop over-flagged
by +39.4 points and 2.83× over-represented in the queue — disclosed and mitigated by
stratified budgeting.

## VI. Discussion

The register that every downstream EPC-prediction tool trains on is measurably distorted at
exactly the regulated boundary. Most of the response is genuine improvement — MEES appears
partly to work — but a small, identifiable segment is threshold targeting, and the most
common observable channel (assessed floor-area revision) requires no physical work. For ML
practice, the imbalance-driven accuracy trap means accuracy-optimal EPC classifiers may
systematically miss the high-risk buildings that matter; recall-first selection and
uncertainty-aware triage are the corrective. **Registered forecast:** as the 2031 ≥ B
standard for large buildings approaches, bunching should emerge at the B/C boundary for
> 1,000 m² stock — a falsifiable prediction.

## VII. Limitations

England & Wales only; observational (bunching ≠ proof of fraud; the cross-section shows a
behavioral response, manipulation is claimed only for the panel's fast-return segment);
postcode-district location precision; fabric changes invisible in the register; escape
rates conditional on re-certification; prior-art claim pending a systematic database
search; conformal set sizes are loose. Full list in the repository.

## VIII. Conclusion

MEES induces a strong, sharply-timed, specification-robust behavioral response at the E/F
boundary of the non-domestic EPC register, with a minority carrying a manipulation
signature. This distortion, and the class imbalance around it, has direct consequences for
the machine-learning tools trained on the register, which we address with recall-first
selection, uncertainty-ranked triage, drift-aware conformal diagnostics, and an open,
reproducible implementation.

## References (to be completed)
[1] M. Collins, J. Curtis, "Bunching of residential building energy performance
certificates at threshold values," *Applied Energy*, vol. 211, pp. 662–676, 2018.
[2] A. Hardy, D. Glew, "An analysis of errors in the EPC database," *Energy Policy*, 2019.
[3] J. Crawley et al., "Quantifying measurement error on E&W EPC ratings," *Energies*, 2019.
[4] J. McCrary, "Manipulation of the running variable in RDD: a density test," *J.
Econometrics*, 2008.
[5] R. Chetty et al., "Adjustment costs, firm responses...," *QJE*, 2011.
[6] H. Kleven, M. Waseem, "Using notches to uncover optimization frictions...," *QJE*, 2013.
[7] H. Kleven, "Bunching," *Annual Review of Economics*, 2016.
[8] M. Cattaneo, M. Jansson, X. Ma, "Simple local polynomial density estimators," *JASA*, 2020.
[9] LuminLab, "An AI-powered building retrofit and energy modelling platform,"
arXiv:2404.16057, 2024.

*Status: complete structural draft with real results; needs reference completion, figure
insertion (all figures exist in `analysis/results/`), systematic prior-art pass, and prose
polishing before submission.*
