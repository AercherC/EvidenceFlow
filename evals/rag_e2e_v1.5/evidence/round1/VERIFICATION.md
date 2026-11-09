# Independent re-verification (main session, 2026-10-27; no LLM)

Code: `verify/` (copied from the main session's scratchpad). It reads Chroma directly with `chromadb`, embeds with `SentenceTransformer` directly, and imports nothing from `evidenceflow` or from the sub-agent's scripts. Its own parts are the BM25, the clause-split rule (a character scan written from the sub-agent's docstring, not from its regexes), every variant's pool / branch / round-robin logic, the context-budget simulation, slicing and the gate. It reuses three things: reranker logits (sidecar + the sub-agent's `scores_cache.jsonl`, sample-checked below), the sub-agent's hand clauses (that is its judgment), and `work/step1/entry_tokens.json`.

| check | result |
|---|---|
| dense top-100 re-embedded vs obs_v12 | 207/207 equal |
| production final top-6 from sidecar | 207/207 equal |
| CANDIDATE_MISS of new 43 / deep list | 27 / the same 15 ids |
| per-group scope (27 CM ∪ B3-tagged) | 32 groups, same set |
| per-group numbers: dense rank full / hand / rule-best, paper rank full / hand, BM25 full / hand | 32 × 7 fields, **0 mismatches** |
| clause-split rule | same pieces as the sub-agent's rule on 207/207; fires on 38 |
| reranker rescore (own `AutoModelForSequenceClassification` load, batch 1) | 150 random cache pairs of 5,933, max \|diff\| 1.2e-5 |
| missing logits for any variant | 0 |
| 15-variant grid | totals, every slice, gained / lost ids, pairs max, CONTEXT_DROP and gate **identical**; per-case included context identical in 207/207 for every variant |

**Two discrepancies, both in my first draft and not in the sub-agent's work:**
- My round-robin skipped a branch's turn when its next chunk had already been chosen. M1's F_rr instead lets the branch advance to its best unchosen chunk. I switched to the F_rr semantics, which is the defined one.
- My BM25 lowercased before tokenizing and stripped the `▁` word-start marker. The sub-agent tokenizes first, then lowercases, and keeps the marker. Both readings fit my brief. I adopted the sub-agent's documented one; the earlier mismatch showed up as 760 missing logits.

`HC8B4_orig_k6` (post-hoc) was not re-implemented.

## Additions and corrections to the sub-agent's report

1. **k = 8 is not the source of C10_rr_k8's gain** (`verify/verify_k8trig.py`). I ran production pool → final 8 on the 38 triggered cases only, with no clause branches. It reaches 141 (+1, E2E-M201/G2) and **drops a context entry on E2E-M120** (a single-doc false fire). Six of C10_rr_k8's seven gains therefore come from reranking each clause's own branch: E2E-M113/G1, M220/G1, M220/G2, M222/G2, M228/G2, M236/G1. Four of the six (M113/G1, M220/G2, M228/G2, M236/G1) were already in the production pool, so they were RERANK_MISS, not CANDIDATE_MISS.
2. **Gold-mapping gaps: confirmed by reading.** All 5 alternative chunks state the fact:
   - M216/G1: "auxiliary denoising task" at dense ranks 2 / 3 / 6 / 17 (M216/G1 is the weakest of the five: these chunks name the task but may not describe it);
   - M216/G2: discriminator description `f30b476856` at rank 13;
   - M220/G1: "Np = 128 patches" `63e85bc347` at rank 41;
   - M221/G1: Fig. 2 caption "CMLP … is our feature extractor" `f015fb1591` at rank 8;
   - M243/G1: `f30b476856` at rank 17, Abstract at rank 25.

   **Correction:** for the full question, every one of the five alternatives is inside dense top-50, M243/G1 included. The sub-agent's note that M243/G1's best alternative lies outside top-50 is wrong.
3. **Production context already holds an alternative for 4 of them:** M216/G1, M216/G2, M220/G1, M221/G1. Under a mapping widened with these alternatives:

   | variant | frozen v1.3 mapping | widened mapping | lost under widened |
   |---|---|---|---|
   | P0 | 140 | 144 | – |
   | C10_rr_k8 | 147 (+7 / −0) | 148 (+6 / −2) | M216/G2, M221/G1 |
   | C25_rrc_k6 | 148 (+10 / −2) | 149 (+9 / −4) | M216/G1, M221/G1, M232/G2, M239/G2 |
   | H_u10 | 142 | 146 | – |

   The net gain of the best variants shrinks from +7 / +8 to **+4 / +5** once the gold is fixed. The mechanism is that clause branches take slots from original-query chunks that happen to carry the fact.
