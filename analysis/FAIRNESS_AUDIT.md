# Fairness / Bias Audit (executed 2026-07-20)

A tool that flags "High risk" and allocates scarce inspections must be checked for *who*
it flags — not just whether it's accurate overall. Run on the 2025 out-of-time
predictions. Code: `analysis/fairness_audit.py`.

## By business type — the headline problem

| Business type | n | Base rate (actual High) | Flag rate (pred High) | Over-flag | Precision | Queue share vs pop |
|---|---|---|---|---|---|---|
| **Office/Workshop** | 19,132 | 14.5% | **53.9%** | **+39.4 pts** | 23.0% | **2.83×** |
| Retail/Financial/Professional | 25,364 | 4.6% | 8.9% | +4.4 pts | 19.2% | 0.01× |
| Restaurant/Cafe | 9,964 | 1.2% | 1.1% | −0.0 pts | 20.7% | 0.00× |

**The finding, stated plainly:** the model massively over-flags **Office/Workshop**
buildings — it predicts High for 54% of them when only 14.5% are actually High — and as a
result they make up **99.5% of the top-5% uncertainty audit queue** (2.83× their share of
the portfolio). A naive deployment would send almost every inspection to one building
type. This is a real, disclosable bias, not a hypothetical.

## By region

Flag rates run well above base rates in every region (the model is broadly
over-cautious), but the gap is largest in **Wales (+22.9 pts)** and **South West England
(+22.0 pts)**, and smallest in **London (+3.8 pts)**. Precision is relatively flat
(~19-27%) across regions, so the model is not wildly *mis-calibrated* by region — but its
*recall* varies sharply: it catches 72-80% of actual High-risk buildings in most regions
but only **38.9% in London**, so London's High-risk stock is systematically under-served
by the flagger even as the queue over-serves offices.

## What this means for deployment (the honest recommendation)

1. **Stratify the audit budget by building type.** Letting a single uncertainty ranking
   drive the queue hands ~all inspections to offices. A real tool should allocate a budget
   per type (or per type × region) and rank within strata.
2. **Report per-group precision to users**, so an owner in an over-flagged group knows the
   flag is less reliable for them (23% precision means most office "High" flags are not
   actually High).
3. **The over-flagging is downstream of the recall-first design choice.** Selecting for
   High-risk recall deliberately trades precision for catching more true positives — which
   inflates flag rates most in the group with the most borderline cases (offices). This is
   the *cost* of the recall-first decision, made visible. It's defensible, but only if
   disclosed, which is why it now lives in the app's Audit Triage tab.

## Limitations
- Groups here are business type and region; a fuller audit would cross them and add
  building size bands.
- "Fairness" is assessed as calibration + selection-rate parity against the true base
  rate; other fairness definitions (equalized odds, etc.) would give a fuller picture.
- The 2025 ground truth is used to compute base rates — available here only because it's a
  labeled test year.
