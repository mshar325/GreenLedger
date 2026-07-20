# GreenLedger — SDG Mapping Document

*SDG GreenTech 2026 · mandatory deliverable*

GreenLedger maps to three UN Sustainable Development Goals. Each mapping below states the
specific target, the concrete mechanism in the project, and the evidence in the repo —
not a generic gesture at the goal.

---

## SDG 9 — Industry, Innovation & Infrastructure
**Target 9.3** — increase access of small-scale enterprises to services; **9.4** —
upgrade infrastructure and retrofit for resource-use efficiency.

**Mechanism.** A formal non-domestic EPC assessment costs money and requires booking an
accredited SBEM assessor. GreenLedger lets a small business estimate its energy-rating
risk tier from a handful of self-reportable details, for free, in seconds — giving
businesses without technical or ESG infrastructure a first-pass triage of whether they
need a paid assessment, and a "Pathway to E" simulator showing which changes have moved
comparable real buildings.

**Evidence.** The deployed app (`app.py`), trained on 430,942 real buildings; the
simulator wired to real repeat-certificate outcomes (Vein 2). The audit-triage tool
extends this to portfolio holders allocating scarce inspections.

**Honest scope.** It is a screening aid, not a certificate — stated in-app and in the User
Manual. It does not replace a statutory EPC.

---

## SDG 12 — Responsible Consumption & Production
**Target 12.6** — encourage enterprises to adopt sustainable practices; **12.8** — ensure
people have relevant information for sustainable development.

**Mechanism.** The recommendation engine surfaces concrete, source-cited operational
changes (LED lighting, thermostat setback, insulation) with real published savings ranges
from the Carbon Trust and GOV.UK — never invented figures. The dashboard and MEES tab make
the state of the small-commercial building stock legible.

**Evidence.** `greenledger/recommendations.py` (every figure cited); the register
dashboard; the year-over-year efficiency-improvement trend (mean rating 85.6 → 59.4,
2018-2026) shown as real data.

---

## SDG 13 — Climate Action
**Target 13.2** — integrate climate measures into policies and planning; **13.3** —
improve institutional capacity on climate mitigation.

**Mechanism.** The EPC asset rating is the standard UK building-level proxy for carbon
intensity, and MEES is a live climate-policy instrument. GreenLedger (a) targets that
rating directly, and (b) — uniquely — provides evidence on whether the compliance
*register itself* is being distorted by the policy (Vein 1 bunching, Vein 2 mechanism),
which is directly relevant to how effectively MEES actually reduces emissions versus
producing paper compliance.

**Evidence.** The MEES Distortion tab; `analysis/vein1_bunching.py`,
`analysis/vein2_panel.py`; the registered forecast that bunching will emerge at the B/C
boundary as the announced 2031 standard for >1,000 m² buildings approaches.

---

## Summary table

| SDG | Target | GreenLedger mechanism | In-repo evidence |
|---|---|---|---|
| 9 | 9.3 / 9.4 | Free risk-screening + retrofit pathway for SMEs without ESG infrastructure | app.py, simulator, triage |
| 12 | 12.6 / 12.8 | Cited operational recommendations; legible stock data | recommendations.py, dashboard |
| 13 | 13.2 / 13.3 | Targets the carbon-proxy rating; evidences policy distortion of the register | MEES tab, Veins 1-2 |

## What this project deliberately does **not** claim
- It does not quantify tonnes of CO₂ saved (no metered consumption data at building level).
- It does not certify any building.
- SDG relevance is via *information access* and *policy-effectiveness evidence*, which is
  what the tool actually delivers — not an overstated direct-emissions-reduction claim.
