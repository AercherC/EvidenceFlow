# Round 2 pre-registration (written 2026-10-27 before any round-2 run; do not edit after the runs start)

Offline only: GPU, no LLM / DeepSeek. Baseline: production 50/6. Data: frozen v1.3 dataset, plus (a) the frozen v1.3 mapping and (b) the draft v1.4 mapping from Part A. **The verdict uses (b); (a) is reported alongside.** If the user rejects Part A items later, the selection is re-read on the approved mapping with the same rule.

## Part A: gold-audit proposal (v1.4, for the user's approval; nothing is frozen)

- **Scope:** every usable multi-doc group (legacy 9 + new 43 = 105 groups).
- **Candidate search:** for each claim of the group, find every chunk in the 817 that matches the claim's answer keys under the repo's key matcher (`claim_hits` / the attribute key rule, thousands-separator folding included), restricted to the group's gold paper. Also take every chunk the round-1 sub-agent named (`per_group.json` `alt_chunks_with_answer_key`, `causes.json`).
- **Acceptance (judged by reading):** a chunk is proposed as an extra alternative only if its text states the claim as written in the claim's `text` field, at the level of detail the question asks for. Merely containing the key words is not enough. M216/G1 is the test case: "auxiliary denoising task" without what it consists of fails a question that asks "what does each addition consist of".
- **Each accepted item records:** case/group/claim, document, an exact verbatim anchor quote from the chunk (the form `required_evidence_groups[].alternatives[]` needs, with page and locator if derivable), chunk id, the dense rank for the full question, and a one-line reason. Rejected candidates are listed with their reason.
- **Draft mapping:** the v1.3 mapping plus one extra single-chunk set per accepted alternative. Same schema; `dataset_fingerprint` unchanged and marked draft in a side file. Never written into `work/rag_e2e_v1.3/frozen/`.

## Part B: pair-budget round (the deterministic clause-branch step)

**Split rule R2** = the round-1 rule (`../split_rule.py`, unchanged), then:
1. **merge:** every piece that ends in `;` or `；` is joined (with one space) to the piece that follows it, repeatedly;
2. **cap:** if more than 3 pieces remain, pieces 3..n are joined (with one space) into one third piece;
3. R2 fires iff ≥ 2 pieces remain.

**Branch construction (triggered cases only; every other case = production 50/6):**
- clause pool size c ∈ {10, 20, 25}, the dense top-c of each clause;
- lexical option bm ∈ {0, 5en}. With 5en, a clause with no CJK character (U+3000–U+9FFF, U+FF00–U+FFEF) gets its BM25 top-5 (round-1 BM25 spec) added to its pool after dedup. A CJK clause gets nothing;
- original branch depth O = min(50, 100 − Σ clause pool sizes), the dense top-O of the original query reranked by the original query;
- each clause pool is reranked by its own clause.

**Final selection f ∈ {rr_k8, rr_k6, rrc_k6}:**
- rr: round-robin with F_rr semantics (each branch adds its best not-yet-chosen chunk per round), clause branches first in clause order, then the original branch;
- rrc: clause branches only (no original branch, O = 0 pairs);
- k as named.

**Grid:** 3 × 2 × 3 = 18 variants, named `R2_c{c}_{bm}_{f}`, plus round-1 `C10_rr_k8` re-run for reference.

**Gate (§8.2 as used in round 1):** 0 single-doc groups lost vs P0, 0 CONTEXT_DROP, reranker pairs ≤ 100 for every case (unique (query, chunk) pairs).

**Selection rule (fixed now):**
1. among variants that pass the gate on the draft v1.4 mapping, pick the most groups reached on the gate set (legacy 9 + primary 32);
2. tie → fewest multi-doc groups lost vs P0;
3. tie → lowest mean pairs on triggered cases;
4. tie → smaller c.

The winner is called **W**.

**Report:** per variant and per mapping, the totals per slice (legacy named/joined, B1–B5, spare, partial, single-doc, all), gained/lost ids vs P0, CANDIDATE_MISS-27 and deep-15 reached (pool / context), pairs mean (triggered) / max, CONTEXT_DROP, gate, R2 firing per slice and its single-doc false fires.

## Part C: M1-LLM ceiling (hand-written resolved clauses; no LLM call)

- **Cases:** the 18 primary B2, B3 and B4 cases.
- **Resolved clauses:** for each case, one resolved clause per sub-question, written by hand under these rules:
  - replace every description of a paper ("the method that self-projects the partial cloud into depth maps", "那篇基于超图…的方法", "an earlier network") with that paper's short method name as in the corpus manifest (e.g. SVDFormer, PCN, PF-Net, Hyper-PCN);
  - restore an elided predicate from the sibling clause (M214: "And what about the early encoder-decoder network" → "What optimizer and epochs does PCN use?");
  - keep all other original words in the original language;
  - **add no word that is not in the question or the paper's name.** In particular no answer-key term and no wording taken from the gold chunk;
  - B4 chained cases: the sub-question about the earlier paper names that paper; the first sentence (the citing context) stays as its own clause, with the citing paper's name kept.

  Store them as `resolved_clauses.json` together with a diff against R2's pieces. Freeze them before running Part C.
- **Simulation:** W's exact branch construction and selection, with R2's pieces replaced by the resolved clauses for these 18 cases. The other 189 cases are identical to W.
- **Decision rule (fixed now):** propose M1-LLM (an LLM resolver) only if the ceiling reaches ≥ 3 more B2–B4 groups than W on the draft v1.4 mapping, with 0 single-doc change and the gate still passing. Otherwise do not propose it. Also report: the ceiling for "name substitution only" (no predicate restoration), and the deep-15 / CANDIDATE_MISS groups the ceiling reaches.
