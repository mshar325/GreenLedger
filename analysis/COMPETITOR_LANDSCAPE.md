# Competitor & Prior-Art Landscape — EPC Prediction (verified 2026-07-19)

Written so we never walk into a viva claiming "nobody does this." Predicting EPC
ratings from limited building inputs **does** exist, commercially and academically.
Every entry below was verified by search on the date above. The honest framing: the
*core prediction* is crowded; our defensible contributions are elsewhere (see bottom).

## A. Commercial tools (all opaque on data and accuracy)

1. **Haptic — "world's first AI-powered commercial EPC calculator"**
   ([hapticepc.com](https://www.hapticepc.com/post/commercial-epc-calculator)).
   - Closest direct competitor: UK **commercial** EPC estimation, GPT-based.
   - Verified specifics: built by feeding their own portfolio of completed **London**
     commercial assessments into a GPT; claims it "learned to estimate ratings
     correctly 95% of the time."
   - Why it doesn't threaten our claim: proprietary data (undisclosed size,
     London-only), and the 95% figure has **no stated methodology** — no train/test
     split, no per-class breakdown, no definition of "correct" (exact band? within
     one?). Their own page recommends a certified physical survey for a real rating.

2. **"Commercial EPC Calculator" custom GPTs** (yeschat.ai, genai.works) — general
   LLMs with a prompt wrapper, not trained classifiers validated against held-out
   data. No dataset, no metrics.

3. **TEAM Energy calculator** — described as built by in-house energy consultants
   using simplified assumptions; explicitly an "indicative / educational planning
   aid," not a validated model. Closer to a rules-of-thumb tool than ML.

## B. Academic prior art (real datasets — but not ours, not our geography)

4. **LuminLab** (arXiv [2404.16057](https://arxiv.org/pdf/2404.16057), Maynooth /
   Trinity College Dublin). AI retrofit platform; an **MLP classifier** predicts the
   BER rating from building features, trained on the **Irish** BER dataset (1M+
   entries, 15 classes A1-G). **Residential.** Notably, the paper flags the
   *imbalanced-dataset* problem explicitly — the same issue that produced our ANN's
   4% High-risk recall — but we did not find it resolved there via a recall-first
   selection criterion.

5. **Turin residential ML/EPC study** (*Energy Policy*,
   [S0301421524004270](https://www.sciencedirect.com/science/article/pii/S0301421524004270))
   — explainable ML for EPC effectiveness, **Italian residential** market.

6. A broader vein of EPC-classification-by-ML papers exists (deep-learning EPC
   classification; gated-multimodal energy-performance prediction). All residential
   or general; none combine prediction with a register-manipulation analysis.

## C. What is genuinely ours (survives a follow-up question)

The prediction task is *not* our novelty; do not claim it is. Our defensible,
checkable contributions:

1. **Fitness-for-purpose model selection.** We select the model that *catches at-risk
   buildings* (High-risk recall), not the one with the best headline accuracy —
   after showing the accuracy-winner (ANN, 68%) catches ~4% of genuinely High-risk
   buildings. LuminLab names the imbalance problem; we act on it as an explicit,
   evidenced design decision.

2. **Questioning the label, not just predicting it (Vein 1 + 2).** We provide
   counterfactual-density evidence that the register everyone (including Haptic and
   the academic work) trains on is **distorted by the MEES regulation itself** —
   bunching at the E/F letting-ban threshold (b≈0 pre-policy → ~3-3.9 under
   enforcement; placebo boundaries flat), plus a repeat-certificate panel on the
   mechanism. We found **no prior work doing this on EPC data.** This is the lead.
   **Language discipline (see VEIN1_DEFENSE.md):** frame as "a strong behavioral
   response at the compliance threshold, a minority of which carries the signature of
   strategic reporting" — NOT "the register is gamed." Bunching alone is also what
   honest do-just-enough compliance produces.

3. **Reproducibility as a differentiator.** Every commercial tool above is either
   proprietary-and-unmethodical or not-really-a-model. Ours is fully checkable:
   named public dataset (E&W non-domestic EPC register), exact scale (430,942 train /
   54,460 out-of-time test), disclosed method (temporal holdout, not a random split),
   and a full 8-run metrics table (accuracy / macro-F1 / recall) open on GitHub.
   *"Ours is the only estimate in this space that is reproducible and independently
   checkable"* is a true, defensible viva sentence.

4. **A portfolio persona no competitor serves.** The Audit Triage tool answers "I
   manage N buildings, can inspect k — which k?" — a resource-allocation problem
   built on model uncertainty, not a single-building estimator.

## D. Pitch guidance

- **Lead with C2 (register distortion), not the prediction.** "We found a strong,
  measurable behavioral response at the legal compliance threshold — with a small
  segment showing the signature of strategic reporting — and built a triage tool
  around the uncertainty that creates" is a sentence no competitor can wave away,
  and one that survives the "bunching is just compliance" objection (which the
  blunter "it's being gamed" version does not).
- Cite this landscape *out loud* — naming Haptic/LuminLab specifically reads as
  literature-review rigor (CIA-1 Innovation line), not weakness.
- Never say "nobody does this." Say "prediction is known; *our specific combination*
  — recall-first selection + register-distortion evidence + reproducibility +
  portfolio triage — is what we did not find prior art for."
