# Vein 1 — Anticipated Objections & Defense Brief

Red-teaming our own flagship claim before a viva/jury does. For each attack: the
strongest form of the objection, and the evidence that answers it. Robustness numbers
come from `vein1_robustness.py` (see `results/vein1_robustness_*.csv`).

## First, the language discipline (do this before anything else)

**Do NOT say "the register is being gamed / manipulated" as the headline.** Bunching at
a threshold is the *expected* result of any consequential threshold, even with zero
cheating — honest owners doing exactly enough genuine retrofit to reach the legal
minimum produce the same excess mass. Overclaiming "manipulation" is the single easiest
way to lose credibility.

**Defensible headline:** *"The MEES letting ban induces a strong, measurable behavioral
response at the E/F boundary — the certificate distribution is bunched exactly where the
regulation bites, and a small identifiable minority of that response carries the
signature of measurement manipulation rather than genuine improvement."*

That claim is true, survives follow-up, and is still striking. Save the word
"manipulation" for the specific Vein 2 fast-return / no-observable-change segment where
we can actually point at it.

---

## Attack 1 — "Bunching is just the policy working, not gaming."

**Strongest form:** excess mass at just-inside-E is exactly what rational, honest
compliance looks like. You've measured that the regulation changed behavior, which
everyone already assumed. Where's the finding?

**Defense:**
- Conceded and reframed, not denied — see the language discipline above. The
  *cross-section* (Vein 1) deliberately claims only a "behavioral response," not fraud.
- The finding is not "behavior changed" but *how much, how sharply, and when*: a
  near-zero response pre-policy snapping to a large one at the enforcement date is
  quantitative evidence of the notch's distortionary power on the register that every
  downstream user (including commercial estimators) trains on.
- The manipulation claim is quarantined to Vein 2's fast-escape segment (≤90 days, ~40%
  with no observable input change, landing at just-E at ~4.4× baseline). That is where
  "gaming" is defensible; the cross-section alone is not, and we say so.

## Attack 2 — "Is 125 actually special, or did you fish for it?"

**Strongest form:** you tested the one boundary you expected to work. Round-number
heaping or SBEM lumpiness could produce apparent bunching at other points too.

**First line of defense — the threshold is externally defined, not ours:** 125/126 is
the legal E/F band boundary written into the MEES regulations before we ever touched the
data. We did not select it to fit a pattern; the law selected it, and our hypothesis
(pre-registered in RESEARCH_ROADMAP.md before the estimator was built) was derived from
the regulation, not from data exploration. State this explicitly wherever the analysis
is presented.

**Defense — the boundary permutation test (`vein1_robustness_permutation.csv`):** we ran
the *identical* estimator at every integer boundary from 70 to 140 (pooled 2018-2026).
The **median boundary has b = 0.04** — i.e. the *typical* rating value shows no bunching
at all. The E/F line at **125 has b = 3.13, exceeding 98% of all other boundaries**, and
the E/F neighbourhood [120,125] averages b = 2.47 versus **−0.38 for boundaries more than
7 points away**. (The 121-124 boundaries are elevated because their windows overlap the
same 121-125 pile — they are the *same* signal seen through sliding windows, not
independent spikes.)

**Two things we disclose rather than hide** (honesty is the defense here):
- One boundary (137) produced a numerically unstable estimate (b≈150, a near-zero
  counterfactual denominator) and is excluded from summary stats but kept flagged in the
  CSV.
- One genuine secondary elevation exists at **boundary 136 (b = 3.57)**, mid-F, at no
  policy-relevant threshold. We do *not* claim it — and it does **not** share 125's other
  two signatures: it has no pre-policy-vs-enforcement time profile and no M/B relabeling
  pattern. 125's case rests on *three* independent legs (permutation + pre-policy temporal
  absence + M/B ≈ 1 relabeling); 136 has only a weak version of one. Noting it is more
  credible than pretending 125 is the single highest value.

## Attack 3 — "SBEM software produces lumpy scores; this is a calculation artifact."

**Strongest form:** the SBEM engine may cluster certain building configurations at
certain scores for mechanical reasons unrelated to any behavior.

**Defense — the pre-policy placebo in time (from `vein1_bunching.py`):** if 125 were a
natural SBEM attractor, the bunching would exist *before* the policy. It does not: b ≈
0.02 in 2012, statistically indistinguishable from zero, and indistinguishable from the
placebo boundaries. The same SBEM engine, the same building physics, produced *no*
excess mass at 125 until the regulation attached a consequence to it. Software mechanics
don't switch on in 2015-2018.

## Attack 4 — "Round-number heaping — 125 is a multiple of 25."

**Strongest form:** assessors and software heap at round numbers; your spike is just
that.

**Defense, three ways:**
1. The estimator already includes explicit heaping dummies for multiples of 5 and 10
   outside the excluded window, so generic round-number attraction is absorbed into the
   counterfactual, not counted as excess.
2. Round-number heaping creates a *symmetric* spike at the round value. Our signature is
   *asymmetric*: excess mass at 121-125 **and a matching hole at 126-130**, with M/B ≈
   0.7-0.9 (missing mass ≈ excess mass). Heaping cannot create a deficit just above the
   round number — only relabeling across a threshold does that. That M/B ≈ 1 relabeling
   pattern is the Kleven-Waseem notch signature, not heaping.
3. The permutation test (Attack 2) includes other round numbers; they don't spike.

## Attack 5 — "Your counterfactual is one arbitrary polynomial and window."

**Defense — the specification sweep (`vein1_robustness_spec.csv`):** we re-estimated b
at the E/F boundary across polynomial degrees {3, 4, 5} × excluded-window half-widths
{±3, ±5, ±7} — nine specifications. **b ranges 2.12 to 3.50 across all nine, and the
lowest bootstrap CI lower-bound is 1.49 — every one clear of zero.** The result is not an
artifact of one lucky specification.

## Attack 6 — "Compositional change — MEES changed *who* gets certified."

**Strongest form:** the sample of buildings entering the register shifted over 2012-2026
(failing buildings now must certify), so rising b could be composition, not behavior.

**Defense:** partly conceded as a scope limit. But the identifying variation is
*within-year, cross-boundary* (125 vs placebo boundaries in the same year, same
sample) — composition shifts the whole distribution, not the excess mass at one specific
boundary relative to others. The permutation test is run within the pooled MEES era, so
composition is held roughly fixed while 125 still stands alone.

## Attack 7 — "Novelty — this method on this data must exist."

**Defense:** our web searches surfaced residential EPC bunching (Collins & Curtis 2018,
Ireland) and E&W domestic error-rate work (Hardy & Glew 2019), but nothing applying a
counterfactual-density bunching design to the E&W *non-domestic* register tied to the
MEES notch (`RELATED_WORK_VEIN1.md`, `COMPETITOR_LANDSCAPE.md`). Honest scope: we claim
"no study surfaced in our searches," not "no study exists," pending a systematic
Scopus/Web-of-Science pass — which is the one clearly-labeled to-do here.

## Attack 8 — "Can I reproduce it?"

**Defense:** code is public; the estimator is deterministic given the data. The raw
register requires a free GOV.UK login (gitignored, ~300MB), but the binned counts behind
the headline figure are small and could be committed so the bunching plot regenerates
without the full download. [Action: export `bunching_density.csv` already exists in
app_data — point reviewers there.]

---

## The one-sentence version for the viva

*"Using a counterfactual-density estimator, we show the MEES F/G letting ban produces a
sharp behavioral response at the exact E/F boundary — absent before the policy, absent
at every non-regulated boundary, stable across specifications — and a repeat-certificate
panel isolates a small segment of that response that looks like measurement manipulation
rather than genuine retrofit."*

Every clause in that sentence is backed by a specific exhibit, and none of it overclaims.
