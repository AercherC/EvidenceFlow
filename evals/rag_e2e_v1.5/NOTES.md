# rag_e2e_v1.5 notes

**v1.5 = v1.4 with the index identity re-bound.** Created on 2026-11-10, after the staging index was
activated on 2026-11-09. Dataset, mapping groups, claim anchors, retrieval fixtures and every expected
metric are byte-identical to v1.4; only `mapping.json`'s `collection_name` / `index_signature` and the
fixture's `mapping_sha256` changed. See `README.md` for the measurements proving the evidence is still
valid (same 817 chunks, zero content differences, 483 `heading_path` metadata differences, P0 152/233).

The provenance below is inherited unchanged from v1.4.

## Provenance (inherited from v1.4)

This directory's dataset was frozen on 2026-10-29 with `scripts/e2e_dataset.py freeze`, which wrote `dataset.json`, `dataset.sha256` and `README.md`.
- `mapping.json` was made by `scripts/e2e_dataset.py map` on the same dataset (same fingerprint).
- `changelog.json` lists every change from v1.3, the 3 dropped proposal items and the user's decision.
- The freeze is by convention: do not edit these files. Any change needs a new version directory.
- Build: `work/rag_e2e_v1.4/tools/build_v14.py` (rebuilds `work/rag_e2e_v1.4/dataset.json` and `changelog.json` from frozen v1.3 + `work/rag_e2e_v1.4/proposal/proposal.json`; two runs give byte-identical output).
- Checks: `work/rag_e2e_v1.4/tools/checks_v14.py` -> `work/rag_e2e_v1.4/checks.json`.

**Identity.**
- Fingerprint `464fda056cd9d380f05704141269a81506167ac149eba106777b347516df8254`.
- sha256 `14611544333b525950aeae7c67327475749e0ab7fc22c3a0d92f9ac6b57a0c00` (`sha256sum -c dataset.sha256` -> OK).
- `validate` = 0 issues (CLI default, as for v1.3; report `work/rag_e2e_v1.4/validate_report.json`); freeze also validated with approval required, machine approval allowed, grounding enforced: 0 issues.
- `map`: groups=255, claim_anchors=263, usable=473, unusable=45, source_failures=0 (same counts as v1.3).

## What changed from v1.3 (Part D: gold audit of the multi-doc evidence groups)

Only `required_evidence_groups[].alternatives` of 41 groups changed (102 alternatives appended after the existing ones), plus `dataset_id`. Questions, claims, answer keys, tags, provenance and every existing alternative are byte-identical to v1.3.

- **Source.** The 105 accepted alternatives of the offline gold-audit proposal (`work/rag_e2e_v1.4/proposal/PROPOSAL.md`, `proposal.json`). Each was judged by hand under the acceptance rule in the proposal and checked with the repo mapper (validates uniquely, maps FULLY_MAPPED onto exactly its proposed chunk).
- **Dropped (user decision, 2026-10-29)**, 3 borderline items:
  - E2E-M201/G2 on chunk 68e085b77c ("PF-Net estimates the missing point cloud hierarchically");
  - E2E-M206/G2 on chunk e9d5beb3f1 ("Based on the designed Patch Seeds and Upsample Transformer...");
  - E2E-M241/G2 on chunk b514a8de50 ("we design a Self-View Fusion Network (SVFNet)...").
  The other alternatives in those three groups are applied.
- Groups changed: ARR-037/G1, G2; ARR-038/G1, G2; ARR-039/G1, G2; ARR-040/G1, G2; E2E-M201/G1, G2; M202/G1, G2; M203/G1; M204/G2; M206/G1, G2; M207/G2; M208/G1; M210/G1; M216/G1, G2; M217/G1; M218/G2; M219/G1; M220/G1, G2; M221/G1; M223/G1; M224/G1; M225/G2; M227/G1; M234/G1; M236/G2; M238/G1, G2; M241/G1, G2; M242/G2; M243/G1, G2; M244/G1.

## Checks at freeze (`work/rag_e2e_v1.4/checks.json`)

- **Diff vs v1.3 dataset.json:** top-level changes are exactly `dataset_id` and `fingerprint`; no case field other than `required_evidence_groups[].alternatives` differs; existing alternatives unchanged; 102 appended alternatives, equal to the changelog.
- **Mapping vs v1.3:** all 255 groups keep id, order, status and every v1.3 chunk_set; 0 status changes; 263 claim anchors identical. The only differences are 102 added single-chunk sets in 41 groups, and each of the 102 items' chunk is a chunk set of its group. chunk_count, chunk_id_hash, collection_name, index_signature and schema_version unchanged; `dataset_fingerprint` equals the v1.4 fingerprint.
- **Mapping vs the proposal draft (`proposal/mapping_draft.json`):** identical except exactly the 3 dropped single-chunk sets (338 -> 335 sets); no dropped chunk is also reached by another accepted alternative in the same group.
- **Retrieval (production P0, offline replay):** 152/233 usable groups reach the context (v1.3: 140; draft: 155). The 3 groups lost vs the draft are exactly the dropped items' groups (E2E-M201/G2, E2E-M206/G2, E2E-M241/G2), which P0 reached only via the dropped anchors. See `work/agent_design/candidate_miss/round2/v14_frozen_rerun.md`.

## Caveats

- **Alternatives were fitted to the chunk texts** of the current 817-chunk index; a re-chunk or corpus change requires re-mapping (and re-checking the 10 short-context anchors listed in the proposal).
- **Context-reach numbers rise because gold widened,** not because retrieval changed; compare retrieval variants only within one dataset version.
- **All v1.3 caveats still hold** (`work/rag_e2e_v1.3/frozen/NOTES.md`), including the 22 UNMAPPED v1.1 groups.
