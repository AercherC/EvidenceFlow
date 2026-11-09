# Candidate misses in multi-doc retrieval: diagnosis and next step (offline, no LLM; 2026-10-27)

Evidence: `per_group.md` (32 groups), `sim_report.md` (16 variants on 207 cases), `VERIFICATION.md` (independent recount; all numbers below re-verified). Baseline: production 50/6 on frozen v1.3, 140/233 groups in context.

## 1. Why multi-doc gold never reaches the dense pool

The `bge_m3_hybrid` collection is dense only ("hybrid" is the chunking strategy). The BM25 used below is self-implemented. The cause classes are the sub-agent's judgment, which I spot-read for the 15 deep misses.

| cause (primary) | 27 CANDIDATE_MISS | 15 deep misses | what it means |
|---|---|---|---|
| compound dilution | 10 | 5 | The clause alone reaches dense top-50 but the compound question does not. Fixable on the query side. |
| vocabulary gap | 11 | 6 | Even the clause misses. Two sub-patterns: (a) the paper is described, not named ("the method that self-projects…", B2/B3), often in a zh question over English chunks; (b) the fact sits in a long multi-topic chunk (Hyper-PCN "4.1 Experiment Settings" is gold for M214/G1, M223/G1 and M226/G1; SnowflakeNet "3.1 Overview"). BM25 rescues 5 of 11, all English, no zh. |
| gold / mapping | 5 | 3 | An unmapped chunk states the fact, always inside dense top-50. For 4 groups it is already in the production context. |
| chunking / caption | 1 | 1 | M219/G1 is an 87-token fragment with no subject. Also seen as secondary causes: figure-label residue at the start of M210/G2 and M243/G2. |

Other notes:
- M243/G2 is a negative question (ProxyFormer has no discriminator), so retrieval cannot match it.
- B3 (12 primary groups): 3 already in top-50, 6 vocabulary gap, 1 each of the others.

## 2. Cheapest deterministic fixes, simulated (gate: 0 single-doc lost, 0 CONTEXT_DROP, ≤ 100 pairs/case)

| variant | all | legacy 9 (named/joined) | B1/B2/B3/B4/B5 | spare | partial | single | gained / lost | pairs max | gate |
|---|---|---|---|---|---|---|---|---|---|
| P0 production | 140 | 0/6 | 4/4/2/6/10 | 8 | 2 | 98 | – | 50 | ✔ |
| D75 / D100 deeper pool | 140 / 140 | 0/6 · 0/7 | 3/3/2/8/10 · 3/3/3/8/10 | 8 · 7 | 1 · 1 | 99 · 98 | 7/7 · 8/8 | 75 / 100 | ✘ single-doc lost |
| H_rrf (dense+BM25 RRF) | 139 | 0/8 | 5/2/2/8/10 | 7 | 1 | 96 | 9/10 | 50 | ✘ 6 single-doc lost |
| H_u10 (dense 40 ∪ BM25 10) | 142 | 0/7 | 4/3/2/6/10 | 8 | 1 | 101 | 4/2 | 50 | ✔ but multi-doc −1 |
| C10_orig_k6 (clause candidates, reranked by the original query) | 141 | 0/7 | 4/4/2/6/10 | 8 | 2 | 98 | 1/0 | 72 | ✔ |
| **C10_rr_k8** (clause branches reranked by their clause, round-robin, k=8, triggered only) | **147** | 0/7 | 5/4/3/7/11 | 10 | 2 | 98 | **7/0** | 90 | ✔ |
| C25_rrc_k6 (clause branches only, k=6) | 148 | 0/7 | 5/5/5/6/10 | 10 | 2 | 98 | 10/2 | **100 (at cap)** | ✔ |
| C25_rr_k8 | 150 | 0/7 | 5/5/5/7/11 | 10 | 2 | 98 | 10/0 | 150 | ✘ pairs |

**Readings:**
- **Pool depth is not the lever.** Deeper global pools and global BM25 fusion lose single-doc groups and gain nothing net.
- **Neither are clause candidates alone.** Reranked by the original query they add +1.
- **What works is reranking each clause's own pool with that clause.** In C10_rr_k8, k=8 alone contributes 1 of the 7 gains and causes a drop when used without branches.
- **No passing variant reaches any of the 15 deep misses.** C25 pools reach 2 (M221/G2, M226/G1), but only at the pair cap.
- **Gate set (legacy 9 + primary 32):** P0 32 → C10_rr_k8 37, C25_rrc_k6 38 (for reference, M1 was 39, of which +4 were legacy named).
- **Caveat: the gold fix shrinks the gain.** With the 5 mapping gaps fixed, P0 becomes 144 and C10_rr_k8 nets +4 (it loses M216/G2 and M221/G1, whose fact sits in an original-query chunk that a clause branch displaces).

## 3. Recommendation

1. **Build next: a deterministic clause-branch retrieval step (C10_rr_k8 shape).**
   - **Trigger:** the punctuation / second-question-marker split. It fires on 38/207: 30/43 new, B2–B5 6/6 each, B1 2/8, partial 0/4. It has 4 single-doc false fires (E2E-M119..M122, "…? Is that correct?"), and none of them loses a group.
   - **Branches:** each clause's dense top-10 reranked by that clause, plus the production dense top-50 reranked by the original query. Round-robin, clause branches first, final k=8 for triggered cases only. Every other case keeps production 50/6.
   - It composes in `runtime.py` like M1 (`agent_design.md` §2.3). No LLM and no agent loop.
   - Do not write production code before the two pre-conditions below.
2. **Pre-condition A: gold audit, v1.4 mapping.** Add the 5 alternative chunks (M216/G1, M216/G2, M220/G1, M221/G1, M243/G1) after reading them, then re-run the gate. With today's numbers the step still passes at +4 with 0 single-doc lost, but it loses 2 multi-doc groups.
3. **Pre-condition B: one pre-registered offline round to fix the pair budget.** C25 recall reaches the deep misses, but C25 breaks the cap because M220/M222/M226 split into 4 pieces at `;`. Test, fixed in advance:
   - merge a `;` description piece into the question piece that follows it;
   - at most 3 clauses;
   - original branch top-30 + clauses top-20;
   - BM25 top-5 added only to English clause pools.

   Pick by the same gate, now on the v1.4 mapping.
4. **Then an answer check [needs the user's OK].** Candidate ×2 on the 38 triggered cases. The 30 new ones already have production runs, re-scored on v1.3. Add baseline ×2 on the 8 triggered old cases (legacy joined 4, single-doc 4). That is ≈ 92 calls, ≈ 0.3M tokens. Proposed cap: 100 calls / 0.4M tokens, with the usual deterministic pre-check.
5. **M1-LLM for B2–B4: not needed as a decomposer; possibly needed as a resolver, and only after an offline bound.**
   - The deterministic split already fires on all 18 B2–B4 cases, so the LLM would no longer be what finds the sub-questions.
   - What B2–B4 still miss:
     - paper identification from descriptions (vocabulary gap: M212/G1, M214/G2, M223/G1, M226/G1/G2, M219/G2, M213/G1, M224/G1);
     - elliptical clauses with no predicate (M214: "And what about the early encoder-decoder network…?").
   - An LLM rewrite targets exactly these. Before any spend, measure its ceiling offline: hand-write the resolved clauses for the 18 B2–B4 cases (paper name + full predicate, original wording otherwise) and run them through the C10_rr_k8 machinery. Propose M1-LLM only if this oracle adds ≥ 3 groups over C10_rr_k8 on B2–B4. C10_rr_k8 reaches 4, 3 and 7 of 12 there.
6. **Not query-side; route to the parser / gold work of `agent_design.md` §1.4:**
   - split multi-topic "Experiment Settings" / "Overview" chunks at their run-in headings ("Implementation Details", "Seed generation module");
   - strip figure-label residue from body chunks;
   - fix the fragment chunks that have no subject.

   This is where the 6 vocabulary-gap deep misses that BM25 does not rescue can move.

**Caveats:**
- The split rule depends on the drafters having separated sub-questions with `?` / `；` / "另外". That is natural punctuation, not a fixed marker as in v1.1's `T_split`, but real users may write single-sentence compounds. The B1 "X 和 Y 分别…" phrasing fires only 2/8.
- All gains are on retrieval only. Answer-level value is untested.
- 87 new groups is a small sample: one case is about 1% of the multi-doc groups.
