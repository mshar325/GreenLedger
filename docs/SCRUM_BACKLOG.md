# GreenLedger — Product Backlog & Sprint Log

*SDG GreenTech 2026 · CIA-7 Scrum artifact. An honest record — the technical core was
built in an intensive early prototyping phase; this log narrates what happened and what
remains, rather than back-dating a synthetic weekly drip.*

---

## Product vision
A cloud-deployed tool that (1) screens the MEES/energy-rating risk of small UK commercial
buildings from cheap self-reportable inputs, and (2) provides research-grade evidence on
whether the MEES compliance register is distorted by the regulation — combining a
practical product with a defensible empirical contribution.

## Product backlog (prioritized)

| ID | Epic | Item | Priority | Status |
|---|---|---|---|---|
| B1 | Data | Source a real, large building-energy dataset (≥30k rows) | Must | ✅ Done (UK EPC, 485k) |
| B2 | Data | Robust small-business filter surviving label changes | Must | ✅ Done |
| B3 | ML | Proxy-vs-audit feature comparison, temporal holdout | Must | ✅ Done |
| B4 | ML | Fitness-for-purpose model selection (recall, not accuracy) | Must | ✅ Done |
| B5 | ML | Confidence intervals on headline metrics | Should | ✅ Done (multiseed_ci) |
| B6 | App | Deployed Streamlit app, risk assessment + SHAP | Must | ✅ Done |
| B7 | App | Dashboard (KPIs, region map, trends) | Should | ✅ Done |
| B8 | App | Pathway-to-E simulator wired to real panel evidence | Should | ✅ Done |
| B9 | App | Audit-triage tool (uncertainty-ranked queue) | Could | ✅ Done |
| B10 | Research | Vein 1 — MEES threshold bunching + robustness | Should | ✅ Done |
| B11 | Research | Vein 2 — repeat-certificate mechanism + fixed effects | Should | ✅ Done |
| B12 | Research | Vein 3 — conformal under drift + value-of-information | Could | ✅ Done |
| B13 | Ethics | Fairness/bias audit of the triage tool | Should | ✅ Done |
| B14 | Quality | Automated test suite | Should | ✅ Done (23 tests) |
| B15 | Docs | SDG mapping, user manual, sustainability report | Must | ✅ Done |
| B16 | Research | IEEE paper draft | Should | 🟡 Draft (docs/PAPER_DRAFT.md) |
| B17 | Research | Systematic (Scopus/WoS) prior-art search | Should | 🔲 To do |
| B18 | Research | Scotland-register replication (external validity) | Could | 🔲 To do (needs separate data) |
| B19 | ML | Distill the 60 MB RF toward XGBoost footprint | Could | 🔲 To do |
| B20 | Research | Adaptive/weighted conformal to repair coverage decay | Could | 🔲 To do |

## Sprint log

**Sprint 0 — Concept & scope.** Rejected a generic "AI business dashboard" (feature-stack,
no research question); reframed around one falsifiable question and a novel dataset.
Migrated from a first CBECS (US) prototype to the UK non-domestic EPC register after
confirming the scale requirement. *Outcome: B1, B2.*

**Sprint 1 — Core ML + app.** Built the shared pipeline, proxy/audit feature split,
temporal-holdout training; caught the accuracy-trap (ANN wins accuracy, 4% recall) and
switched to recall-based selection; deployed the Streamlit app. *Outcome: B3, B4, B6, B7.*
*Bugs caught: missing deps on deploy, stale-scaler prediction corruption, map colour
normalization.*

**Sprint 2 — Research veins.** Executed Vein 1 (bunching, with a fake-citation catch and a
full robustness battery), Vein 2 (panel mechanism + building fixed effects), Vein 3
(conformal + triage curve). Red-teamed Vein 1 and imposed language discipline
("behavioural response," not "gamed"). *Outcome: B10, B11, B12.*

**Sprint 3 — Rigor, ethics, product polish.** Confidence intervals, fairness audit,
automated tests, simulator wired to real panel evidence, dashboard redesign, mandatory
documents. *Outcome: B5, B8, B9, B13, B14, B15.*

## Definition of Done (applied throughout)
A backlog item is done only when: the code runs end-to-end and is verified (not just
written); every number shown is a real output or a cited source (no fabricated figures);
limitations are documented; and it's committed to the public repo.

## Retrospective — what went well / what to improve
- **Well:** verify-don't-assume discipline caught multiple real bugs (label drift, stale
  scaler, CI class-weight inconsistency, uncertainty-metric bug) before they shipped;
  honest negative/awkward findings (the green-computing tension, the fairness over-flag)
  were surfaced rather than hidden.
- **Improve:** more of the work should be committed incrementally as it happens (going
  forward); the systematic prior-art search and the paper's prose remain the main open
  items.
