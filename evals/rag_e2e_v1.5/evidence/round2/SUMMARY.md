# Round 2: verdict and verification (main session, 2026-10-27; offline, no LLM)

Spec: `PREREG.md`, sha256 `b822a488…23ae`, unchanged after the runs. Sub-agent write-up: `README.md`, `DEVIATIONS.md`, `partB_report.md`, `partC_report.md`. Gold proposal: `work/rag_e2e_v1.4/proposal/` (draft, **awaiting the user's approval**).

## Verdict

| part | result |
|---|---|
| A. gold audit | Proposes 105 extra alternatives in 41 of 105 multi-doc groups (302 candidates rejected). Under the draft v1.4 mapping, production P0 reaches **155** groups instead of 140 (+15, none lost); gate set 32 → 45. |
| B. pair-budget round | All 18 pre-registered variants pass the gate on both mappings. **W = `R2_c25_5en_rr_k8`**, picked by the pre-registered rule with no tie-break. |
| C. M1-LLM ceiling | Hand-resolved clauses reach **19 B2–B4 groups, the same as W** (Δ 0 on v1.4; Δ −3 on v1.3). The rule needs ≥ +3, so **M1-LLM is not proposed.** |

**W** works as follows:
- The round-1 split is applied; a `;` description is merged into its question and the clause count is capped at 3.
- Each clause's pool is its dense top-25, plus BM25 top-5 for English clauses, and is reranked by that clause.
- The original branch is the dense top-(100 − clause pools), reranked by the original question.
- The final list is a round-robin (clause branches first) to k = 8, on triggered cases only (38/207).

| mapping | P0 | W | gained / lost vs P0 | gate set (83) | single-doc | pairs (triggered) |
|---|---|---|---|---|---|---|
| draft v1.4 | 155 | **163** | 10 / 2 (lost M203/G1, M221/G1, both multi-doc) | 45 → **51** | 98 → 98 | exactly 100 on every triggered case |
| frozen v1.3 | 140 | **151** | 11 / 0 | 32 → 41 | 98 → 98 | same |

Round 1's C10_rr_k8 gets 159 / 147 on the same two mappings. In every cell of the grid, adding BM25 to English clauses beats dense-only, and rr_k8 beats rr_k6 and rrc_k6.

## Independent verification

Code is in `verify/` (own split / merge / cap, CJK test, pools, original-branch depth, F_rr, context simulation, slicing, gate and selection; reranker logits reused from the caches).

| check | result |
|---|---|
| missing logits | 0 |
| R2 firing | 38/207; the single-doc false fires are E2E-M119..M122, the same as the sub-agent's |
| 19 variants × 2 mappings | totals, gate set, gained / lost, single-doc lost, drops, pairs max / mean and gate all **identical** |
| per-case included context, draft v1.4 | 19 × 207 = 3,933 lists, 0 mismatches |
| selection | same W on both mappings |
| Part C ceiling (resolved and name-only, both mappings) | B2–B4 counts and gained / lost ids **identical** |
| sub-agent rescore | 120 pairs, max \|diff\| 1.1e-5 (round 1: 150 pairs with my own model load, 1.2e-5) |

**Part A, read by me:** the 18 items that move P0 (chunks already in production context whose group was unreached under v1.3), plus the round-1 five.
- All 18 state their claim.
- **Borderline (user's call):**
  - M201/G2: "PF-Net estimates the missing point cloud hierarchically" does not say "instead of the whole object".
  - M206/G2: the Upsample Transformer is named as part of decoding, not called the point generator.
  - M241/G2: the anchor names SVFNet without the depth-image detail.
- **Round-1 five:**
  - M216/G1: round 1's ranks 2/3/6/17 are correctly rejected because they only name the task. The one accepted chunk is at rank 214.
  - M216/G2, M220/G1, M221/G1, M243/G1: accepted.
- I did **not** read the other 82 accepted items. They do not move P0, but they do move variant counts under v1.4.

## Caveats for the decision

1. **Acceptance strictness.**
   - Terse legacy claims ("hypergraph", "self-attention and cross-attention") accept many restatements. All 8 legacy-named groups change.
   - A stricter, question-level bar would lower draft P0 and leave ARR-039/G2 and ARR-040/G1/G2 unreached.
   - W's margin over P0 is 6 to 9 gate-set groups on both mappings, so its lead does not hinge on this.
2. **No pair headroom.**
   - W uses exactly 100 reranker pairs on every triggered case, which is 2× production on 38 cases.
   - `R2_c20_5en_rr_k8` gives up 1 gate-set group (50) at 93 pairs mean, max 100.
   - If the cap is meant to leave headroom, prefer it. The pre-registered rule says W.
3. **Why the ceiling fails.** Resolving descriptions to names gains M221/G1, M223/G1, M224/G1 and M226/G2, but loses M213/G2, M221/G2, M228/G2 and M229/G2, where the description words were what pulled the gold in. An LLM resolver therefore has no measured upside here.
4. **Untested at answer level.** The 4 single-doc false fires change context but lose no group.

## Next steps

1. **User decision on the v1.4 proposal** (`work/rag_e2e_v1.4/proposal/PROPOSAL.md`): accept all, accept with the borderline items dropped, or apply a stricter bar.
2. **Freeze v1.4.** Add the alternatives to `required_evidence_groups`, then `freeze` + `map`, as v1.3 was built. Re-run P0 and W on the frozen mapping (deterministic, minutes).
3. **Answer check [needs OK].**
   - W ×2 on the 38 triggered cases, plus production ×2 on the 8 triggered old cases. The 30 new cases already have production runs, which would be re-scored on v1.4.
   - That is ≈ 92 calls, ≈ 0.3M tokens; proposed cap 100 calls / 0.4M tokens, with the deterministic pre-check.
4. **Build W as a deterministic retrieval step** only if 3 passes (§8.3). No LLM resolver.

## Update 2026-10-29: v1.4 frozen (user decision: proposal approved, the three borderline anchors dropped)

**Frozen v1.4.**
- Location: `work/rag_e2e_v1.4/frozen/`. Fingerprint `464fda05…8254`; sha256 `14611544…0c00` (`sha256sum -c` OK).
- validate: 0 issues. map: 255 groups, 0 source failures.
- 102 alternatives in 41 groups. Dropped: M201/G2 `68e085b77c`, M206/G2 `e9d5beb3f1`, M241/G2 `b514a8de50`. The other alternatives in those groups stay.

**My structural check** (`verify/verify_v14_struct.py`):
- Versus v1.3, only `dataset_id`, `fingerprint` and the 102 appended alternatives differ. No existing alternative changed, and no dropped anchor is present.
- The mapping keeps every v1.3 set and status.
- The mapping equals the draft minus exactly the 3 dropped sets.

**Frozen v1.4 result** (sub-agent rerun and my recount agree on every count and id):

| variant | all | gate set | single | gained / lost vs P0 | pairs mean / max | gate |
|---|---|---|---|---|---|---|
| P0 | 152 | 43 | 98 | – | 50 | ✔ |
| **W** `R2_c25_5en_rr_k8` | **161** | **50** | 98 | 11 / 2 | 100 / 100 | ✔ |
| runner-up `R2_c20_5en_rr_k8` | 160 | 49 | 98 | 10 / 2 | 93.3 / 100 | ✔ |
| C10_rr_k8 (round 1) | 157 | 46 | 98 | 7 / 2 | 72.9 / 90 | ✔ |

- **W gained:** M113/G1, M115/G1, M201/G2, M213/G2, M218/G2, M220/G2, M221/G2, M222/G2, M228/G2, M229/G2, M236/G1.
- **W lost:** M203/G1, M221/G1. Both are multi-doc; no single-doc group is lost.
- The pre-registered rule still selects W on the frozen mapping.
- **Part C ceiling on frozen v1.4:** 19 vs W's 19 (Δ 0). M1-LLM stays dropped.
