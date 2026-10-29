# Round 2: readings of PREREG.md where a choice was needed

PREREG.md was not edited (sha256 b822a488…23ae, equal to PREREG.sha256 after the runs). No variant was added, dropped or retuned; there are no post-hoc rows.

## Part A

1. **Key rule.** I found no separate "attribute key rule" in the repo under that name, so candidates use `evidenceflow.eval.e2e.key_match.claim_hits` only. Its normalisation folds thousands separators.
2. **Gold paper and claims.** The gold paper is the document(s) of the group's mapped chunks. A group's claims are the required claims whose `evidence_anchors` share an anchor string with the group's alternatives (round-1 rule, `diagnose.py`). Round-1-named chunks enter the list even when they are already key matches. The candidate source is recorded per row.
3. **Terse legacy claims.** Several legacy claims are bare terms: "dynamic queries", "Patch Seeds", "Skip-Transformer", "hypergraph", "self-structure", "point pyramid decoder". For these I read the bar as follows: the chunk's prose presents the term as the gold paper's own component, of the kind the question asks about. Example: "SeedFormer introduces a new shape representation, namely Patch Seeds" counts for "compare … representations". Titles, references, figure-label residue, tables, equation-only chunks and ablation or results text that only presupposes the component are rejected. **This reading produces many accepted legacy items (all 8 legacy-named groups change). It is the most debatable judgment in Part A.**
4. **Anchors.**
   - Each anchor is cut verbatim from the chunk text, with line-end hyphenation joined and whitespace collapsed.
   - The locator uses the chunk's own page range, the occurrence, and 20-character left and right contexts taken from the normalized source.
   - When the mapper's unique-context proof did not fall inside the chunk, the context was shortened to 12, then 6, then 0 characters. Seven alternatives ended with no context: E2E-M202/G2 2a7b98ee14, M210/G1 823c5102d3, M217/G1 and M220/G2 and M243/G2 b9bb954958, M223/G1 8f2544df98, M236/G2 63e85bc347. Two (M216/G2 and M243/G1 f30b476856) have 11-character contexts.
   - One hint was changed so that it falls inside a single source segment: M236/G2 now uses "Ns = 256 seed points (Fig. 6(c))".
   - All 105 alternatives validate uniquely and map FULLY_MAPPED onto exactly the proposed chunk.
5. **Draft mapping.** Extra sets go only into `groups[].chunk_sets`, one single-chunk set per accepted alternative. PREREG speaks only of group alternatives, so `claim_anchors` are unchanged. `dataset_fingerprint` is unchanged, and the draft status is recorded in `DRAFT_NOT_FROZEN.txt`.
6. **Same chunk in two groups.** A chunk accepted for two groups is counted once per group. Examples: 6224e921d9 for M238/G1 and G2; b514a8de50 for ARR-040/G1, M206/G1 and M241/G2.

## Part B

7. **Variant names.** The bm token is `0` or `5en`, for example `R2_c25_5en_rr_k8`.
8. **5en pool.** The BM25 top-5 is united with the clause's dense top-c after dedup, so a pool holds between c and c+5 chunks. Σ clause pool sizes (after BM25) gives O = min(50, 100 − Σ). A guard max(0, ·) was coded; it never binds.
9. **R2 cap.** The cap (more than 3 pieces) never fires on this dataset. After the merge the largest piece count is 3: M217, M229 and M232.
10. **Gate set.** The gate set is the legacy 9 cases plus the primary 32 cases, which hold 83 usable groups.
11. **CANDIDATE_MISS-27 and deep-15.** These are the round-1 id lists, defined on v1.3. They are not recomputed for the draft mapping, so under v1.4 some of them are already "reached" by P0 through new alternatives.
12. **Reference row.** `C10_rr_k8` is re-run exactly as in round 1: round-1 split, clause dense top-10 reranked by the clause, original dense top-50, F_rr, k = 8 on triggered cases. On v1.3 it reproduces round 1's 147.
13. **Rescore check.** 120 pairs were sampled with seed 11 from round-1 `scores_cache.jsonl` and rescored with the production scorer: max |diff| 1.14e-5, mean 1.8e-6. Part B needed only 2 new pairs and Part C 930. The 36 new clause strings were embedded into `dense_cache_r2.jsonl`, because round-1 caches are read-only.

## Part C

14. **What gets replaced.** The paper name replaces the whole noun phrase that describes the paper, including any relative clause and anaphors such as "that earlier paper", "那篇方法" and "某个方法". When the description is the main clause of the sentence and the question refers back to it ("A transformer completion model feeds noised ground-truth centers …; how does it keep those queries …"), only the subject noun phrase is replaced and the predicate is kept. This applies to M222 and M226/G1. In the B4 citing-context clause the cited paper's description is also replaced, because PREREG says "every description".
15. **Sub-questions and restoration.**
    - M216: the shared "What does each of these additions consist of?" becomes two clauses, "What does the extra training task AdaPoinTr adds for its decoder consist of?" and "What does the discriminator PF-Net also trains consist of?".
    - M229: "and how many for testing?" becomes "and how many models did PCN hold out for testing?".
    - M232: "and why?" becomes "and why is the first up-sampling factor in SnowflakeNet set to that?".
    - M217: the subject piece split off by R2 is joined back to its predicate piece.
    - Every added word is either a paper name or appears in the question; `resolved_clauses_check.json` lists them, with 0 violations.
16. **Name-only variant.** The R2 pieces with only the description→name substitution, keeping the R2 piece structure.
17. **"0 single-doc change"** is read as single-doc groups reached equal to W. The 18 cases are all multi-doc and the other 189 cases are identical to W (asserted in code), so this is 0 by construction.

## Incident

18. My first `gpu_r2.py score` call ran from `round2/` instead of the repo root and failed. Because `Settings` resolves paths relative to the working directory, it created an empty `round2/data/chroma/chroma.sqlite3` (184 KB). I deleted that directory. Nothing else was written, and every later GPU call ran from the repo root.
