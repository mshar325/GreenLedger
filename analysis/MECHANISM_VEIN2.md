# Vein 2 — The Repeat-Certificate Mechanism Test (executed 2026-07-19)

Panel built from every certificate with a UPRN, 2012-2026: **1,176,148 certificates,
867,996 unique buildings, 212,908 buildings with 2+ certificates, 250,659 consecutive
same-building pairs.** "Escape" = previous certificate F/G (rating ≥126), next
certificate E or better (≤125). n = 40,369 F/G-first pairs, of which 37,767 escape.
All numbers below are from `analysis/vein2_panel.py` output (CSVs in
`analysis/results/`).

## Finding 1 — Escape rates track the policy, exactly

Among buildings whose previous certificate was F/G, the share whose next certificate
escapes to E-or-better, by re-certification year:

| Era | Escape rate |
|---|---|
| 2012 (pre-policy) | **56%** |
| 2015-2017 (announcement/anticipation) | 74% → 88% |
| 2018+ (enforcement) | **96-99%, every year** |

After enforcement, a failing building that gets re-certified almost never stays
failing. (Selection caveat below: this is conditional on choosing to re-certify.)

## Finding 2 — Most escapes look genuine; a marked minority looks targeted

- **75.4% of escapes land at D or better** (≤100) — far past the threshold,
  consistent with substantial real improvement, not minimal compliance.
- **5.7% land at 121-125 ("just barely E") vs a 2.6% baseline** (share of all
  E-or-better certificates at 121-125) — escapes are **~2.2x over-represented** in
  the just-compliant window.
- Echoes Collins & Curtis (2018) on residential: threshold effects are *widespread
  but not systemic*.

## Finding 3 — The speed test inverts the assessor-shopping prior

- Escapes are **slow**: median gap 2,044 days (~5.6 years); only 6.7% within 90
  days. Bulk timing is consistent with retrofit/re-letting/10-year renewal cycles,
  **not** rapid re-assessment.
- Non-escapes are **fast**: median 354 days; **25.6% within 30 days**. Rapid
  re-certification of failing buildings mostly *fails again* — consistent with
  error correction, duplicate lodgements, or evidence-gathering for exemption
  registration rather than successful shopping. (n = 2,602, small.)
- **But the fast-escape segment (≤90 days, n = 2,535) is distinctive:**
  - **39.7% show no observable input change** (vs 23.9% of slow escapes);
  - **11.4% land at just-barely-E** — double the all-escape rate, ~4.4x baseline.
  - This is the subset carrying the manipulation signature; it is small.

## Finding 4 — What changes across escapes (the channels)

| Channel | Fast (≤90d) | Slow (>90d) | All escapes |
|---|---|---|---|
| Heating fuel changed | 29.2% | 23.8% | 24.1% |
| Building environment changed | 20.2% | 24.1% | 23.9% |
| Air-con presence changed | 11.2% | 18.3% | 17.8% |
| **Assessed floor area changed >5%** | 43.6% | **60.6%** | **59.5%** |
| **Nothing observable changed** | **39.7%** | 23.9% | 24.9% |

- **Assessed floor area changes in ~60% of escapes** — the single most common
  channel. Floor area enters the SBEM calculation directly, so re-measurement (or
  re-scoping what counts as conditioned area) can move a rating without any
  physical work. This deserves its own paragraph in the paper: it is either a lot
  of genuine reconfiguration, or a soft lever.
- Fuel switches are roughly **symmetric** (electricity→gas 2,878 vs gas→electricity
  2,748). Plausibly era-dependent — SBEM carbon factors historically penalized
  grid electricity, then swung the other way as the grid decarbonized — flagged
  as a hypothesis to test by splitting switches by year, not asserted.

## Honest limitations

1. **Selection into re-certification:** escape rates are conditional on a new
   certificate existing. Failing buildings that never re-certify (sold, exempted,
   withdrawn from letting) are invisible here — this measures escape *among
   returners*, not population improvement.
2. **Fabric is invisible:** the public non-domestic register carries no glazing,
   insulation, or wall/roof fields. "Nothing observable changed" means none of
   {fuel, environment, air-con, floor area} changed — real fabric retrofits would
   look like "nothing" here. The 39.7% fast/no-change figure is therefore an
   *upper bound* on pure reassessment gaming in that segment, not a measurement
   of it.
3. **Consecutive-pair design:** intermediate certificates (3+ chains) are treated
   as separate transitions; no within-chain modelling yet.
4. UPRN coverage is imperfect (~1.18M of ~1.4M certificates carry one); missing
   UPRNs are unlikely to be random.

## How Veins 1 + 2 fit together

Cross-section (Vein 1): excess mass at 121-125 appears with MEES and scales with
enforcement. Panel (Vein 2): when failing buildings return, they overwhelmingly
escape; most go deep (real improvement), but a small fast-return segment lands
disproportionately at just-barely-E with nothing observable changed — and the most
common observable channel overall is assessed-floor-area revision, a lever that
requires no physical work. Together: the bunching is real, mostly built out of
substantive responses, with a thin, identifiable layer that looks like measurement
gaming — and the floor-area channel is the specific audit-worthy margin.
