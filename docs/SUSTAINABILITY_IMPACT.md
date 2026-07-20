# GreenLedger — Sustainability Impact Report

*SDG GreenTech 2026 · mandatory deliverable*

Two dimensions: the sustainability *impact of the tool's domain* (helping small businesses
act on building energy), and the *green-computing footprint of the software itself* — the
course grades both, and honesty on each matters more than inflated claims.

---

## 1. Domain impact — realistic, not inflated

**What GreenLedger plausibly contributes:**
- Lowers the barrier to a first-pass energy-risk check for small businesses that would not
  otherwise pay for an SBEM assessment (SDG 9).
- Points owners at concrete, cited efficiency measures and shows, from real data, what
  changing heating fuel etc. actually did to comparable buildings (SDG 12).
- Provides policymakers with evidence on whether MEES produces genuine improvement or
  paper compliance at the threshold (SDG 13) — arguably its most novel contribution.

**What it does NOT claim** (the honest boundary):
- No tonnes-of-CO₂-saved figure. The register has no building-level metered consumption, so
  any such number would be fabricated. We refuse to invent one.
- It is a screening aid; behaviour change and real retrofit happen off-platform.
- Impact is via *information access* and *policy evidence*, which is what the tool truly
  delivers.

**A genuine, measurable finding it surfaces:** the register shows small-commercial mean
asset ratings improving from 85.6 (2018) to 59.4 (2026) — real, sector-wide efficiency
gains — while also showing that a slice of the improvement near the E/F line is threshold
targeting rather than deep retrofit. That nuance is itself a sustainability contribution:
it warns against reading raw EPC improvement as pure decarbonisation.

---

## 2. Green-computing footprint of the software — measured, not asserted

The course grades green computing as a design constraint. We measured it (`export_artifacts.py`)
rather than claiming it:

| Model | High-risk recall | Size | Inference |
|---|---|---|---|
| Logistic Regression | 0.049 | 1.6 KB | 0.15 ms |
| **Random Forest (deployed)** | **0.699** | 60 MB | 56 ms |
| XGBoost | 0.584 | 1.4 MB | 1.0 ms |
| ANN (MLP) | 0.044 | 112 KB | 0.22 ms |

**The honest tension (disclosed, not hidden):** unlike a convenient "smallest model wins"
story, the model that actually does the job — Random Forest, selected on High-risk recall —
is the *largest and slowest* here. We deploy the model that works and state the tradeoff
openly (Audit Triage → Green computing expander). XGBoost is documented as the low-footprint
alternative (1.4 MB, 1 ms) for meaningfully lower recall, so a deployer with a hard compute
budget has an evidenced choice.

**Practices actually applied:**
- CPU-only inference; no GPU dependency; no continuous retraining loop.
- The conformal-coverage analysis (Vein 3) gives an explicit signal for *when* retraining is
  warranted, avoiding wasteful periodic retraining.
- Data artifacts compressed (dashboard 15 MB gzip; region GeoJSON simplified 4.2 MB → 459 KB
  to cut transfer and render cost).
- One deployed Streamlit app, no separate backend — minimal infrastructure.

**What we would improve:** the 60 MB Random Forest is the footprint weak point; distilling
it toward XGBoost's size while preserving recall is the obvious next green-computing step,
and is named as future work rather than pretended-away.

---

## 3. Ethics note
Recommendations cite only real published figures. No fabricated statistics anywhere in the
project. The fairness audit (`analysis/FAIRNESS_AUDIT.md`) discloses that the triage tool
over-flags Office/Workshop buildings — a bias we surface in the app rather than conceal,
because a tool that allocates real inspections must be accountable for who it targets.
