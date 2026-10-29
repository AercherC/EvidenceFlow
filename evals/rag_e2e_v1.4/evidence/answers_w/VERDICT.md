# W answer check on frozen v1.4: verdict (main session, 2026-10-29)

- **Approval:** user-approved spend, with a hard cap of 100 calls / 400k tokens.
- **Spent:** 92 HTTP attempts and 271,218 tokens (base1 8, cand1 38, base2 8, cand2 38). One call per case, with 0 retries, 0 repairs and 0 re-runs.
- **Pre-checks (zero LLM):** all passed. The live trigger matched on 207/207, baseline retrieval on 8/8 and W retrieval on 38/38, order included. Equality was re-asserted before every call.
- **Sub-agent write-up:** `answers_w_report.md`.
- **My independent recount:** `verify/verify_answers_w.py`, with its own PASS logic on the raw obs files, reusing only `claim_hits`. It gives identical PASS counts, gains, losses, flips, abstentions and tokens.

## Result (the 38 triggered cases)

| | base1 | base2 | cand1 | cand2 |
|---|---|---|---|---|
| PASS | 8 | 8 | 14 | 14 |
| answerable abstentions | 19 | 21 | 9 | 10 |
| prompt tokens / case | 2,358 | 2,423 | 2,871 | 2,871 |

Baseline sources:
- the 30 triggered new cases use the stored v1.2 runs of 2026-10-27, re-scored on v1.4;
- the 8 others (E2E-M113, M115, M116, M118, M119–M122) are new runs.

**Stable gains (6):** E2E-M115, M201, M218, M220, M229, M236.
- **Stable losses:** 0. **Within-pair flips:** 0.
- **Single-doc false fires** (M119–M122): 2/4 PASS in every run, with no change.
- **Citation contract errors:** 0.

**§8.3 answer gate: passes.**

| condition | result | met |
|---|---|---|
| net gains ≥ max(3, 2 × flips) | 6 ≥ 3 | ✔ |
| exact McNemar p (6 vs 0) | 0.031 | – |
| 0 stable single-doc losses | 0 | ✔ |
| citation errors | 0 | ✔ |
| prompt tokens ≤ 1.5 × 2,408 = 3,612 | 2,871 | ✔ |
| LLM calls per answer | 1 | ✔ |

## Read by hand

**Five gains are clean:** M115, M218, M220, M229, M236.
- In each, both baseline runs answer one half and abstain on the other ("the knowledge base does not provide…").
- W puts the missing evidence into context, and the candidate answers it correctly with a citation: Hunyuan3D-Omni / PCN; SVDFormer needs no paired colour images; MSE proxy-alignment loss; PCN's 100 / 150 split; point proxies / Ns = 256.

**E2E-M201 is a scoring artefact of the v1.4 decision.**
- The baseline answer says PF-Net "estimates the missing point cloud hierarchically" and cites `68e085b77c`. That is exactly the borderline anchor the user dropped.
- Under v1.4 the baseline is GROUP_MISSING.
- With that anchor counted, the baseline would pass and M201 would not be a gain.
- The candidate's answer is the more complete one: "outputs only the missing part … rather than the whole object".

**Robust figure:** +5 / −0, McNemar p = 0.0625. That still clears the net ≥ 3 bar.

## Caveats

- **Endpoint drift.** The 30 new-case baselines are from 2026-10-27; the candidate runs are from 2026-10-29. The 8 fresh baseline cases agree run to run (same PASS on every case), which suggests little drift. It is not proof.
- **The two candidate runs are near-deterministic.** Prompt totals and layers are identical across cand1 and cand2. The ×2 protocol therefore guards against noise less than it appears to.
- **Offline vs answer level.** Offline, W also gained M113/G1 and lost M203/G1 and M221/G1. None of these changed a case-level PASS.
- **B3 stays at 0/6** in every arm. The deep candidate misses there remain parser / chunking work.

## Next step

W has passed both the offline retrieval gate (§8.2) and the answer gate (§8.3). Building it is a production code change, which needs the user's go-ahead:
- a deterministic multi-doc retrieval step composed in `runtime.py` (`agent_design.md` §2.3): R2 split → per-clause dense top-25 (+ BM25 top-5 for English clauses) reranked by the clause → original branch filling the pair budget to 100 → F_rr round-robin to k = 8 on triggered cases;
- a BM25 index over the chunk texts (needs no new dependency);
- trace events for branches;
- tests that pin the 38 offline final lists.
