# Candidate-miss round 2: gold audit, pair-budget grid, M1-LLM ceiling (offline: GPU only, no LLM, no network; 2026-10-27)

This round follows `PREREG.md`. Its sha256, b822a488…23ae, was checked before and after the runs and is unchanged. My readings of the spec are in `DEVIATIONS.md`.

Production baseline P0: BGE-M3 exact dense top-50 → bge-reranker-base → top-6 → ContextBuilder (4096-token budget). The verdict uses the **draft v1.4 mapping**; the frozen v1.3 mapping is reported alongside.

## Checks

| check | result |
|---|---|
| reranker rescore of 120 random pairs from round-1 `scores_cache.jsonl` (seed 11), production `score_pool` | max \|diff\| 1.14e-5, mean 1.8e-6 (`rescore_check_r2.json`) |
| new pairs scored | 2 for Part B, 930 for Part C (`scores_cache_r2.jsonl`) |
| new dense rankings | 36, all Part C clause strings (`dense_cache_r2.jsonl`; every R2 piece was already in round-1's cache) |
| missing logits / dense rankings in any variant | 0 (asserted) |
| P0 and C10_rr_k8 on v1.3 | 140 and 147, identical to round 1 |
| Part A anchors through the repo mapper (`validate_source` + `map_source_to_chunks`) | 105/105 source-valid with a unique occurrence; FULLY_MAPPED onto exactly the proposed chunk |
| `mapping_draft.json` | validates as `E2EMapping`; dataset_fingerprint unchanged |
| Part C no-added-words rule | 0 violations (`resolved_clauses_check.json`); clauses frozen before the Part C simulation (`resolved_clauses.sha256` 74feb379…853d) |
| Part C: cases outside the 18 identical to W | asserted per case |
| files written | only `round2/` and `work/rag_e2e_v1.4/proposal/`. `git status` clean (no tracked file touched). One incident: an empty Chroma file I created by mistake was deleted (DEVIATIONS 18) |

## Part A: gold-audit proposal (`work/rag_e2e_v1.4/proposal/`)

- **Scope.** 105 usable multi-doc groups were searched. The search found 407 candidate rows in 63 groups: key matches within the gold paper plus the chunks round 1 named. I read and judged every row (`judg/j1.json`, `judg/j2.json`).
- **Result.** **105 alternatives accepted in 41 groups; 302 rejected.** The draft mapping adds 105 single-chunk sets.
- **Effect on P0.** P0 rises from **140 to 155** (+15, none lost). An accepted alternative is already in the production context for:
  - ARR-039/G2, ARR-040/G1, ARR-040/G2;
  - M201/G2, M202/G1, M202/G2, M203/G1, M206/G1, M206/G2, M210/G1;
  - M216/G2, M220/G1, M221/G1;
  - M227/G1, M241/G2.
- **Round-1 five:**

  | group | verdict | accepted chunk(s) | in P0 context? | note |
  |---|---|---|---|---|
  | M216/G1 | accepted, but only via f95bdcbf68 (rank 214) | f95bdcbf68 | no | the other 7 candidates fail the bar: 5 only name the "(auxiliary) denoising task" (including round 1's ranks 2/3/6/17), 1 is an ablation table, and 1 is the denoising loss without the decoder feeding |
  | M216/G2 | accepted | f30b476856, f584353605, 68e085b77c | f30b476856 yes | |
  | M220/G1 | accepted | 63e85bc347 | yes | |
  | M221/G1 | accepted | f015fb1591 | yes | also d5340000ab, aeeaeffb53 |
  | M243/G1 | accepted | f30b476856, 68e085b77c, f584353605 | no | |

- **Most debatable call.** The terse legacy term-claims (DEVIATIONS 3) account for 30 of the 105 accepted items, spread over all 8 legacy-named groups.

Details per item are in `PROPOSAL.md`: verbatim anchor, page, chunk, dense rank, whether it is in P0 context, and a one-line reason. `proposal.json` holds the full EvidenceAlternative objects.

## Part B: pair-budget grid (18 variants + C10_rr_k8 reference), `partB.py`, `sim_results_r2.json`, `partB_report.md`

**R2 firing** (the same cases as round 1; the merge only changes pieces):

| slice | firing |
|---|---|
| legacy named | 0/4 |
| legacy joined | 4/5 |
| B1 | 2/8 |
| B2, B3, B4, B5 | 6/6 each |
| spare | 4/7 |
| partial | 0/4 |
| single-doc | 4/155 |
| all | 38/207 |

- **Single-doc false fires:** E2E-M119, M120, M121, M122. None loses a group in any variant.
- **Merge:** M220, M222 and M226 go from 4 pieces to 2.
- **Cap:** never fires.

Columns: groups reached in the included context per slice (usable groups in brackets).
- *gained / lost*: vs P0 of the same mapping; the number in brackets counts the multi-doc groups lost.
- *CM27 / deep15*: the round-1 id lists, counted as reached in the pool / in the context.
- *pairs*: unique reranker pairs per case, mean over triggered cases / max over all cases.

**Draft v1.4 mapping (verdict)**

| variant | leg-named (8) | leg-joined (10) | B1 (16) | B2 (12) | B3 (12) | B4 (12) | B5 (13) | spare (14) | partial (8) | single (128) | gate set (83) | all (233) | gained / lost (multi lost) | CM27 pool / ctx | deep15 pool / ctx | pairs mean trig / max | DROP | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 3 | 6 | 11 | 5 | 3 | 7 | 10 | 9 | 3 | 98 | 45 | 155 | 0 / 0 (0) | 6 / 5 | 3 / 2 | - / 50 | 0 | PASS |
| R2_c10_0_rr_k8 | 3 | 7 | 11 | 4 | 2 | 8 | 11 | 11 | 3 | 98 | 46 | 158 | 5 / 2 (2) | 9 / 4 | 5 / 1 | 71.3158 / 80 | 0 | PASS |
| R2_c10_0_rr_k6 | 3 | 7 | 10 | 4 | 2 | 7 | 9 | 11 | 3 | 98 | 42 | 154 | 4 / 5 (5) | 9 / 4 | 5 / 1 | 71.3158 / 80 | 0 | PASS |
| R2_c10_0_rrc_k6 | 3 | 7 | 11 | 4 | 1 | 6 | 10 | 10 | 3 | 98 | 42 | 153 | 5 / 7 (7) | 7 / 4 | 3 / 1 | 21.3158 / 50 | 0 | PASS |
| R2_c10_5en_rr_k8 | 3 | 8 | 11 | 4 | 3 | 9 | 11 | 11 | 3 | 98 | 49 | 161 | 8 / 2 (2) | 13 / 6 | 7 / 1 | 75.2368 / 92 | 0 | PASS |
| R2_c10_5en_rr_k6 | 3 | 8 | 10 | 4 | 3 | 9 | 9 | 11 | 3 | 98 | 46 | 158 | 7 / 4 (4) | 13 / 6 | 7 / 1 | 75.2368 / 92 | 0 | PASS |
| R2_c10_5en_rrc_k6 | 3 | 8 | 11 | 4 | 3 | 8 | 10 | 10 | 3 | 98 | 47 | 158 | 8 / 5 (5) | 11 / 6 | 5 / 1 | 25.2368 / 50 | 0 | PASS |
| R2_c20_0_rr_k8 | 3 | 7 | 10 | 6 | 3 | 8 | 11 | 11 | 3 | 98 | 48 | 160 | 7 / 2 (2) | 11 / 7 | 7 / 4 | 91.3158 / 100 | 0 | PASS |
| R2_c20_0_rr_k6 | 3 | 7 | 10 | 5 | 3 | 7 | 9 | 11 | 3 | 98 | 44 | 156 | 5 / 4 (4) | 11 / 6 | 7 / 3 | 91.3158 / 100 | 0 | PASS |
| R2_c20_0_rrc_k6 | 3 | 7 | 10 | 5 | 3 | 7 | 10 | 11 | 3 | 98 | 45 | 157 | 6 / 4 (4) | 10 / 6 | 6 / 3 | 42.6316 / 60 | 0 | PASS |
| R2_c20_5en_rr_k8 | 3 | 8 | 10 | 5 | 4 | 9 | 11 | 11 | 3 | 98 | 50 | 162 | 9 / 2 (2) | 14 / 8 | 8 / 3 | 93.2895 / 100 | 0 | PASS |
| R2_c20_5en_rr_k6 | 3 | 8 | 10 | 5 | 4 | 9 | 9 | 11 | 3 | 98 | 48 | 160 | 8 / 3 (3) | 14 / 8 | 8 / 3 | 93.2895 / 100 | 0 | PASS |
| R2_c20_5en_rrc_k6 | 3 | 8 | 10 | 5 | 4 | 9 | 10 | 11 | 3 | 98 | 49 | 161 | 9 / 3 (3) | 13 / 8 | 7 / 3 | 45.6316 / 72 | 0 | PASS |
| R2_c25_0_rr_k8 | 3 | 7 | 10 | 6 | 3 | 8 | 11 | 11 | 3 | 98 | 48 | 160 | 7 / 2 (2) | 12 / 6 | 7 / 3 | 100.0 / 100 | 0 | PASS |
| R2_c25_0_rr_k6 | 3 | 7 | 10 | 6 | 3 | 7 | 9 | 11 | 3 | 98 | 45 | 157 | 6 / 4 (4) | 12 / 6 | 7 / 3 | 100.0 / 100 | 0 | PASS |
| R2_c25_0_rrc_k6 | 3 | 7 | 10 | 6 | 3 | 7 | 10 | 11 | 3 | 98 | 46 | 158 | 7 / 4 (4) | 11 / 6 | 6 / 3 | 53.2895 / 75 | 0 | PASS |
| R2_c25_5en_rr_k8 | 3 | 8 | 10 | 6 | 4 | 9 | 11 | 11 | 3 | 98 | 51 | 163 | 10 / 2 (2) | 15 / 8 | 8 / 3 | 100.0 / 100 | 0 | PASS |
| R2_c25_5en_rr_k6 | 3 | 8 | 10 | 6 | 4 | 9 | 9 | 11 | 3 | 98 | 49 | 161 | 9 / 3 (3) | 15 / 8 | 8 / 3 | 100.0 / 100 | 0 | PASS |
| R2_c25_5en_rrc_k6 | 3 | 8 | 10 | 6 | 4 | 9 | 10 | 11 | 3 | 98 | 50 | 162 | 10 / 3 (3) | 14 / 8 | 7 / 3 | 56.0263 / 86 | 0 | PASS |

**Frozen v1.3 mapping**

| variant | leg-named (8) | leg-joined (10) | B1 (16) | B2 (12) | B3 (12) | B4 (12) | B5 (13) | spare (14) | partial (8) | single (128) | gate set (83) | all (233) | gained / lost (multi lost) | CM27 pool / ctx | deep15 pool / ctx | pairs mean trig / max | DROP | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 0 | 6 | 4 | 4 | 2 | 6 | 10 | 8 | 2 | 98 | 32 | 140 | 0 / 0 (0) | 0 / 0 | 0 / 0 | - / 50 | 0 | PASS |
| R2_c10_0_rr_k8 | 0 | 7 | 5 | 4 | 2 | 7 | 11 | 10 | 2 | 98 | 36 | 146 | 6 / 0 (0) | 3 / 1 | 1 / 0 | 71.3158 / 80 | 0 | PASS |
| R2_c10_0_rr_k6 | 0 | 7 | 4 | 4 | 2 | 6 | 9 | 10 | 2 | 98 | 32 | 142 | 4 / 2 (2) | 3 / 1 | 1 / 0 | 71.3158 / 80 | 0 | PASS |
| R2_c10_0_rrc_k6 | 0 | 7 | 5 | 4 | 1 | 5 | 8 | 9 | 2 | 98 | 30 | 139 | 6 / 7 (7) | 3 / 1 | 1 / 0 | 21.3158 / 50 | 0 | PASS |
| R2_c10_5en_rr_k8 | 0 | 8 | 5 | 4 | 3 | 8 | 11 | 10 | 2 | 98 | 39 | 149 | 9 / 0 (0) | 7 / 3 | 3 / 0 | 75.2368 / 92 | 0 | PASS |
| R2_c10_5en_rr_k6 | 0 | 8 | 4 | 4 | 3 | 8 | 9 | 10 | 2 | 98 | 36 | 146 | 7 / 1 (1) | 7 / 3 | 3 / 0 | 75.2368 / 92 | 0 | PASS |
| R2_c10_5en_rrc_k6 | 0 | 8 | 5 | 4 | 3 | 7 | 10 | 9 | 2 | 98 | 37 | 146 | 9 / 3 (3) | 7 / 3 | 3 / 0 | 25.2368 / 50 | 0 | PASS |
| R2_c20_0_rr_k8 | 0 | 7 | 5 | 5 | 3 | 7 | 11 | 10 | 2 | 98 | 38 | 148 | 8 / 0 (0) | 5 / 3 | 3 / 2 | 91.3158 / 100 | 0 | PASS |
| R2_c20_0_rr_k6 | 0 | 7 | 4 | 4 | 3 | 6 | 9 | 10 | 2 | 98 | 33 | 143 | 5 / 2 (2) | 5 / 2 | 3 / 1 | 91.3158 / 100 | 0 | PASS |
| R2_c20_0_rrc_k6 | 0 | 7 | 5 | 4 | 3 | 6 | 10 | 10 | 2 | 98 | 35 | 145 | 7 / 2 (2) | 5 / 2 | 3 / 1 | 42.6316 / 60 | 0 | PASS |
| R2_c20_5en_rr_k8 | 0 | 8 | 5 | 4 | 4 | 8 | 11 | 10 | 2 | 98 | 40 | 150 | 10 / 0 (0) | 8 / 4 | 4 / 1 | 93.2895 / 100 | 0 | PASS |
| R2_c20_5en_rr_k6 | 0 | 8 | 4 | 4 | 4 | 8 | 9 | 10 | 2 | 98 | 37 | 147 | 8 / 1 (1) | 8 / 4 | 4 / 1 | 93.2895 / 100 | 0 | PASS |
| R2_c20_5en_rrc_k6 | 0 | 8 | 5 | 4 | 4 | 8 | 10 | 10 | 2 | 98 | 39 | 149 | 10 / 1 (1) | 8 / 4 | 4 / 1 | 45.6316 / 72 | 0 | PASS |
| R2_c25_0_rr_k8 | 0 | 7 | 5 | 5 | 3 | 7 | 11 | 10 | 2 | 98 | 38 | 148 | 8 / 0 (0) | 5 / 2 | 3 / 1 | 100.0 / 100 | 0 | PASS |
| R2_c25_0_rr_k6 | 0 | 7 | 4 | 5 | 3 | 6 | 9 | 10 | 2 | 98 | 34 | 144 | 6 / 2 (2) | 5 / 2 | 3 / 1 | 100.0 / 100 | 0 | PASS |
| R2_c25_0_rrc_k6 | 0 | 7 | 5 | 5 | 3 | 6 | 10 | 10 | 2 | 98 | 36 | 146 | 8 / 2 (2) | 5 / 2 | 3 / 1 | 53.2895 / 75 | 0 | PASS |
| R2_c25_5en_rr_k8 | 0 | 8 | 5 | 5 | 4 | 8 | 11 | 10 | 2 | 98 | 41 | 151 | 11 / 0 (0) | 8 / 4 | 4 / 1 | 100.0 / 100 | 0 | PASS |
| R2_c25_5en_rr_k6 | 0 | 8 | 4 | 5 | 4 | 8 | 9 | 10 | 2 | 98 | 38 | 148 | 9 / 1 (1) | 8 / 4 | 4 / 1 | 100.0 / 100 | 0 | PASS |
| R2_c25_5en_rrc_k6 | 0 | 8 | 5 | 5 | 4 | 8 | 10 | 10 | 2 | 98 | 40 | 150 | 11 / 1 (1) | 8 / 4 | 4 / 1 | 56.0263 / 86 | 0 | PASS |

Gained and lost ids for every variant, and the list of passing variants in selection order, are in `partB_report.md`.

## W and why

- **The PREREG rule on the draft v1.4 mapping selects W = `R2_c25_5en_rr_k8`.**
  - All 18 variants pass the gate on both mappings: 0 single-doc groups lost, 0 CONTEXT_DROP, at most 100 pairs per case.
  - W is the single best variant on the gate set, with 51 of 83 groups reached against P0's 45. The runners-up have 50: R2_c20_5en_rr_k8 and R2_c25_5en_rrc_k6. No tie-break was needed.
  - On v1.3 the same rule also gives W, at 41 against P0's 32.
- **W on v1.4:** 163/233 groups reached against P0's 155.
  - Gained (10): M113/G1, M115/G1, M213/G2, M218/G2, M220/G2, M221/G2, M222/G2, M228/G2, M229/G2, M236/G1.
  - Lost (2, both multi-doc): M203/G1, M221/G1.
  - deep-15 in context: M216/G2, M218/G2, M221/G2. P0 has M216/G2 and M221/G1.
- **W on v1.3:** 151/233 against 140; 11 gained, 0 lost. The gains add M201/G2 and M220/G1 and drop M218/G2.
- **Caveat: W has no pair headroom.** With c = 25, the original-branch depth O = 100 − Σ(clause pools), so every triggered case uses exactly 100 pairs (mean = max = 100).
  - The best variant with headroom is R2_c20_5en_rr_k8: gate set 50, all 162, mean 93.3 pairs, max 100.
  - The cheapest strong variant is R2_c25_5en_rrc_k6: gate set 50, mean 56 pairs, max 86, but it loses 3 multi-doc groups.
  - The rule ignores pair cost except as the third tie-break, so it picks the variant at the cap.
- **The lexical option is what moves the grid.**
  - At equal c and f, `5en` beats `0` on the gate set in all 9 (c, f) cells on v1.4, by 2 to 5 groups.
  - `rr_k8` beats both `rr_k6` and `rrc_k6` on the gate set in every c/bm cell, on both mappings.

## Part C: M1-LLM ceiling (`resolved_clauses.json`, `partC.py`, `ceiling_results.json`)

The 18 primary B2/B3/B4 cases use W's exact construction with the R2 pieces replaced by hand-resolved clauses. Every added word is a paper name or a word copied from the question. The other 189 cases are W.

**v1.4_draft**

| row | B2 | B3 | B4 | B2-B4 | single | gate set | all | pairs mean trig / max | DROP | gate | deep15 ctx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 5 | 3 | 7 | 15 | 98 | 45 | 155 | None / 50 | 0 | PASS | E2E-M216/G2, E2E-M221/G1 |
| W | 6 | 4 | 9 | 19 | 98 | 51 | 163 | 100.0 / 100 | 0 | PASS | E2E-M216/G2, E2E-M218/G2, E2E-M221/G2 |
| ceiling_resolved | 5 | 7 | 7 | 19 | 98 | 51 | 163 | 100.0 / 100 | 0 | PASS | E2E-M216/G2, E2E-M218/G2, E2E-M221/G1, E2E-M223/G1 |
| ceiling_name_only | 4 | 7 | 7 | 18 | 98 | 50 | 162 | 100.0 / 100 | 0 | PASS | E2E-M218/G2, E2E-M221/G1, E2E-M223/G1 |

**v1.3_frozen**

| row | B2 | B3 | B4 | B2-B4 | single | gate set | all | pairs mean trig / max | DROP | gate | deep15 ctx |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P0 | 4 | 2 | 6 | 12 | 98 | 32 | 140 | None / 50 | 0 | PASS | - |
| W | 5 | 4 | 8 | 17 | 98 | 41 | 151 | 100.0 / 100 | 0 | PASS | E2E-M221/G2 |
| ceiling_resolved | 4 | 4 | 6 | 14 | 98 | 38 | 148 | 100.0 / 100 | 0 | PASS | - |
| ceiling_name_only | 4 | 4 | 6 | 14 | 98 | 38 | 148 | 100.0 / 100 | 0 | PASS | - |

**Decision (fixed rule: ceiling ≥ W + 3 on B2–B4 on the draft v1.4 mapping, 0 single-doc change, gate passing): not met. Do not propose M1-LLM.**

- **Draft v1.4.**
  - The resolved ceiling reaches 19 B2–B4 groups, the same as W (Δ 0). The name-only ceiling reaches 18 (Δ −1).
  - Single-doc change is 0 and the gate passes.
  - Resolving gains four groups: M221/G1, M223/G1, M224/G1 and M226/G2.
  - It loses four: M213/G2, M221/G2, M228/G2 and M229/G2. On these the description words in R2's pieces had been what pulled the gold chunk.
- **v1.3.** Both ceilings reach 14 against W's 17 (Δ −3).
- **What the ceiling reaches.** On v1.4, deep-15 in context: M216/G2, M218/G2, M221/G1, M223/G1; W has M216/G2, M218/G2, M221/G2. In the pool it reaches 10 of the deep-15, against W's 8. CANDIDATE_MISS-27 in context: 10 for the resolved ceiling, 9 for name-only, 8 for W.

## Doubts

1. **Part A judgment.**
   - The acceptance bar is applied to the claim's `text`.
   - For terse legacy term-claims this accepts many restatements (30 items).
   - M216/G2 and M243/G1 accept the abstract and introduction sentences "completion loss and adversarial loss". These state the claim's text but say nothing about what the discriminator consists of. f30b476856 does both.
   - A stricter "question-level" reading would reject those two sentences and several legacy items, and would lower draft P0 (for example ARR-039/G2, ARR-040/G1 and ARR-040/G2 would likely stay unreached).
2. **M207/G2.** I rejected 4ffa08bdc2 (it is in P0 context) because it never says that the encoder stacks GDP and SFA. Accepting it would add one more P0 group.
3. **W sits at the pair cap by construction.** If the cap is meant to leave headroom, R2_c20_5en_rr_k8 is one gate group lower.
4. **CM27 / deep-15 under v1.4.** These are round-1 ids. Under v1.4, some of them (M216/G2, M221/G1, M220/G1) count as reached because of new alternatives, not because of better retrieval.

## Files

| file | content |
|---|---|
| `r2common.py` | loaders (round-1 code imported unmodified; selectable mapping; round-2 score cache) |
| `partA_candidates.py` → `partA_candidates.json`, `.md` | candidate search (407 rows, full chunk text) |
| `partA_view.py` | reading aid (key-hit windows) |
| `judg/j1.json`, `judg/j2.json` | hand judgments (A/R, reason, anchor hint) |
| `partA_build.py` → `partA_build.log`, `work/rag_e2e_v1.4/proposal/{proposal.json, mapping_draft.json, DRAFT_NOT_FROZEN.txt}` | anchors, repo-mapper check, draft mapping |
| `partA_report.py` → `work/rag_e2e_v1.4/proposal/PROPOSAL.md` | proposal write-up |
| `gpu_r2.py` → `dense_cache_r2.jsonl`, `scores_cache_r2.jsonl`, `rescore_check_r2.json`, `gpu_r2_partB.log`, `gpu_r2_partC.log` | GPU service writing only round-2 caches |
| `partB.py` → `r2_splits.json`, `r2_queries.json`, `r2_pairs.json`, `sim_results_r2.json`, `partB_report.md`, `partB_report.log` | Part B |
| `resolved_clauses.json`, `resolved_clauses.sha256`, `resolved_clauses_check.json` | Part C clauses, frozen before simulation |
| `partC.py` → `c_queries.json`, `c_pairs.json`, `ceiling_results.json`, `partC_report.md`, `partC_report.log` | Part C |
| `DEVIATIONS.md` | readings of the spec |

`sim_results_r2.json` layout: `mappings.{v1.4_draft, v1.3_frozen}.per_variant_per_case.<variant>.<case_id>`, where each entry holds `{triggered, final, included, skipped, groups_reached, pairs}`. The variant summaries and the selection are alongside.

Run order, from the repo root with `PYTHONPATH=src`, HF offline, `D:\miniconda\envs\agent-gpu\python.exe -X utf8`:
1. `partA_candidates.py`, `partA_build.py`
2. `partB.py splits | queries | needs`, then `gpu_r2.py score r2_pairs.json` and `gpu_r2.py check 120`
3. `partB.py report`, `partA_report.py`
4. `partC.py check | queries`, then `gpu_r2.py dense c_queries.json`
5. `partC.py needs`, then `gpu_r2.py score c_pairs.json`
6. `partC.py report`
