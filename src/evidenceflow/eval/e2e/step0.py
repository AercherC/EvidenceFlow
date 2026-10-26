"""Step-0 retrieval diagnostics for rag_e2e_v1: which mechanism explains evidence that misses the
Reranker's candidate pool. No LLM is called.

Five analyses, all computed from artifacts written by ``scripts/e2e_diagnose.py observe
--rerank-depth`` (observations plus the Reranker-score sidecar, see ``RerankScores``) and
``scripts/e2e_step0.py``:
1. depth sweep: one Reranker scoring pass over the Dense Top-100, re-sorted with
   ``rank_scored_candidates`` to simulate a Dense pool of K = 20 / 50 / 100 (final Top-5);
2. truncation: how many Query–Passage pairs exceed the Reranker ``max_length``, and whether
   truncated pairs are over-represented among evidence chunks that drop out at rerank;
3. chunk type: standalone element type (table / caption / equation) vs body vs mixed, per stage;
4. back matter split: references + acknowledgements vs appendix + supplementary material;
5. oracle paper-scoped retrieval: Dense search restricted to the gold documents. It uses the gold
   document IDs, so it is an upper bound for paper linking, never a deployable result.

The scoring helper ``score_pool`` takes any ``PairScorerProtocol``; every other function is pure.
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.evaluation_v2 import ContractModel
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.dataset import E2EDataset, EvalCase
from evidenceflow.eval.e2e.diagnosis import (
    E2EObservation,
    RetrievalStage,
    bundle_rank,
    fully_in,
    is_back_matter,
    retrieval_stage,
)
from evidenceflow.eval.e2e.mapping import E2EMapping, is_usable
from evidenceflow.eval.e2e.slices import SLICE_DEFINITION, SLICES, case_slice, percentile
from evidenceflow.eval.e2e.step1 import SimulatedRun, rerank_ids, run_variant
from evidenceflow.ingestion.chunker import HYBRID_STANDALONE_TYPES

STEP0_REPORT_SCHEMA_VERSION = "rag-e2e-step0-report-v1"
RERANK_SCORES_SCHEMA_VERSION = "rag-e2e-rerank-scores-v1"
DEPTHS = (20, 50, 100)
SCOPED_DEPTHS = (5, 10)

BackMatterCategory = Literal["references_ack", "appendix_supp"]

# Split of ``diagnosis._BACK_MATTER``: the union of the two patterns is exactly the old regex, so
# ``back_matter_category`` is non-None if and only if ``is_back_matter`` holds.
_REFERENCES_ACK = re.compile(r"\b(references?|bibliography|acknowledg\w*)\b", re.I)
_APPENDIX_SUPP = re.compile(r"\b(appendix|appendices|supplementar\w*)\b", re.I)
HEADING_SEPARATOR = " > "


# --------------------------------------------------------------------------------------
# Reranker scoring (one pass), the scores sidecar and depth simulation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredPool:
    """One scoring pass over a frozen candidate list, in candidate order."""

    scores: list[float]
    # Per-pair "untruncated length > max_length"; None when the scorer exposes no tokenizer.
    truncated: list[bool] | None
    batch_seconds: list[float]


def pair_truncation_flags(tokenizer: Any, pairs: Sequence[tuple[str, str]], max_length: int) -> list[bool]:
    """Per pair: whether the untruncated joint encoding exceeds ``max_length``.

    Same call as ``TransformersCrossEncoderScorer`` uses for its batch count, so the per-pair flags
    sum to the scorer's ``last_truncated_pair_count`` for the same batch.
    """

    encoded = tokenizer(list(pairs), padding=False, truncation=False)
    return [len(input_ids) > max_length for input_ids in encoded["input_ids"]]


def score_pool(scorer: Any, query: str, chunks: Sequence[DocumentChunk], *, batch_size: int) -> ScoredPool:
    """Score every candidate once, in the same batch layout as ``LearnedReranker.rerank``.

    Batches are cut from the start of the candidate list, so when ``batch_size`` divides the online
    pool size the first batches are exactly the online batches and the pool-depth scores reproduce
    the online scores. ``LearnedReranker`` is bypassed on purpose: its ``max_candidates`` equals the
    online pool, and this pass must cover the Dense Top-100.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    tokenizer = getattr(scorer, "tokenizer", None)
    max_length = getattr(scorer, "max_length", None)
    scores: list[float] = []
    flags: list[bool] | None = [] if tokenizer is not None and max_length is not None else None
    batch_seconds: list[float] = []
    for start in range(0, len(chunks), batch_size):
        pairs = [(query, chunk.content) for chunk in chunks[start : start + batch_size]]
        started = time.perf_counter()
        batch_scores = scorer.score_pairs(pairs)
        # score_pairs returns CPU floats, so the GPU work is finished when the clock stops.
        batch_seconds.append(time.perf_counter() - started)
        if len(batch_scores) != len(pairs) or any(not math.isfinite(score) for score in batch_scores):
            raise ValueError("pair scorer returned an unexpected or non-finite score")
        scores.extend(float(score) for score in batch_scores)
        if flags is not None:
            flags.extend(pair_truncation_flags(tokenizer, pairs, max_length))
    return ScoredPool(scores=scores, truncated=flags, batch_seconds=batch_seconds)


class RerankScoresHeader(ContractModel):
    """First line of the scores sidecar: the run constants of the one scoring pass."""

    schema_version: Literal["rag-e2e-rerank-scores-v1"] = RERANK_SCORES_SCHEMA_VERSION
    model: str = Field(min_length=1, description="生产 Reranker 模型名（reranker_model）。")
    batch_size: int = Field(gt=0)
    max_length: int | None = Field(default=None, gt=0, description="scorer 没有 max_length 时为 None。")
    rerank_depth: int = Field(gt=0, description="一次性打分覆盖的 Dense 前 N 名。")


class CaseRerankScores(ContractModel):
    """One case of the scores sidecar; item i belongs to the observation's ``dense_ranking[i]``."""

    case_id: str = Field(min_length=1)
    dense_scores: list[float] = Field(description="与 dense_ranking 逐项对应的 Dense 相似度。")
    rerank_scores: list[float] = Field(description="Reranker 对 Dense 前 N 名的一次性打分。")
    rerank_truncated: list[bool] | None = Field(
        default=None, description="与 rerank_scores 对应：该 Query–Passage 对未截断长度是否超过 max_length。"
    )
    rerank_batch_seconds: list[float] = Field(description="打分各 batch 的耗时（秒），按候选顺序切分。")

    @model_validator(mode="after")
    def validate_alignment(self):
        """Truncation flags, when present, pair up with the scores."""

        if self.rerank_truncated is not None and len(self.rerank_truncated) != len(self.rerank_scores):
            raise ValueError("rerank_truncated must align with rerank_scores")
        return self


@dataclass(frozen=True)
class RerankScores:
    """A whole sidecar: the header plus per-case scores keyed by case_id."""

    header: RerankScoresHeader
    cases: dict[str, CaseRerankScores]


def scores_path(observations: Path) -> Path:
    """The sidecar next to an observation file: ``obs.jsonl`` -> ``obs.scores.jsonl``."""

    return observations.with_suffix(".scores.jsonl")


def rerank_scores_text(header: RerankScoresHeader, cases: Sequence[CaseRerankScores]) -> str:
    """Sidecar JSONL: the header line, then one line per case."""

    return "".join(item.model_dump_json() + "\n" for item in (header, *cases))


def read_rerank_scores(path: Path) -> RerankScores:
    """Read a sidecar written by ``observe --rerank-depth``; reject a repeated case_id."""

    lines = [line for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")
    cases = [CaseRerankScores.model_validate_json(line) for line in lines[1:]]
    by_id = {item.case_id: item for item in cases}
    if len(by_id) != len(cases):
        raise ValueError("duplicate case_id in rerank scores")
    return RerankScores(RerankScoresHeader.model_validate_json(lines[0]), by_id)


def simulate_depth(observation: E2EObservation, scores: CaseRerankScores, k: int) -> SimulatedRun:
    """Dense Top-K -> stored Reranker scores -> final Top-``final_top_k``; no model call."""

    if len(scores.rerank_scores) < k or len(observation.dense_ranking) < k:
        raise ValueError(f"observation {observation.case_id} has no reranker scores to depth {k}")
    pool = [(item.chunk_id, item.document_id) for item in observation.dense_ranking[:k]]
    reranked = rerank_ids(pool, scores.rerank_scores[:k])
    return SimulatedRun([c for c, _ in pool], reranked, reranked[: observation.final_top_k])


def depth_latency(batch_seconds: Sequence[float], k: int, batch_size: int) -> float:
    """Scoring time of the first K candidates: the batches that a K-deep online pool would run."""

    return float(sum(batch_seconds[: math.ceil(k / batch_size)]))


# --------------------------------------------------------------------------------------
# Chunk type and back matter
# --------------------------------------------------------------------------------------


def chunk_type_label(element_types: Sequence[str]) -> str:
    """Standalone type (table / table_caption / figure_caption / equation), ``body`` or ``mixed``.

    The hybrid chunker packs each standalone element alone, so ``mixed`` (standalone and body
    elements, or two standalone types, in one chunk) should not occur; it is kept as a label so a
    violation shows up in the counts instead of being folded into another class.
    """

    if not element_types:
        return "unknown"
    kinds = set(element_types)
    standalone = kinds & HYBRID_STANDALONE_TYPES
    if not standalone:
        return "body"
    return standalone.pop() if len(kinds) == 1 else "mixed"


def back_matter_category(heading_path: str | None) -> BackMatterCategory | None:
    """Split back matter into references+acknowledgements vs appendix+supplementary.

    The deepest matching heading decides, so "Supplementary Material > References" is references
    and "References > Appendix A" is appendix.
    """

    if not is_back_matter(heading_path):
        return None
    for part in reversed(heading_path.split(HEADING_SEPARATOR)):
        if _REFERENCES_ACK.search(part):
            return "references_ack"
        if _APPENDIX_SUPP.search(part):
            return "appendix_supp"


def check_element_reproduction(
    indexed: Mapping[str, Sequence[str]],
    rebuilt: Mapping[str, Sequence[str]],
    element_types: Mapping[str, str],
) -> dict:
    """Compare indexed chunk provenance with a re-parse + re-chunk of the source PDFs.

    ``indexed`` / ``rebuilt`` map chunk_id -> element_ids; ``element_types`` comes from the
    re-parse. ``ok`` requires exact reproduction: the same chunk_id set (chunk_ids hash the
    content), the same element_ids per chunk, and every element_id resolving to a type.
    """

    indexed_elements = {e for ids in indexed.values() for e in ids}
    missing = sorted(indexed_elements - set(element_types))
    common = set(indexed) & set(rebuilt)
    differ = sorted(c for c in common if list(indexed[c]) != list(rebuilt[c]))
    only_index = sorted(set(indexed) - set(rebuilt))
    only_rebuild = sorted(set(rebuilt) - set(indexed))
    return {
        "indexed_chunks": len(indexed),
        "rebuilt_chunks": len(rebuilt),
        "indexed_element_ids": len(indexed_elements),
        "unresolved_element_ids": len(missing),
        "chunk_ids_only_in_index": len(only_index),
        "chunk_ids_only_in_rebuild": len(only_rebuild),
        "element_ids_differ": len(differ),
        "examples": {"unresolved": missing[:5], "only_in_index": only_index[:5], "differ": differ[:5]},
        "ok": not (missing or only_index or only_rebuild or differ),
    }


# --------------------------------------------------------------------------------------
# Oracle paper-scoped retrieval
# --------------------------------------------------------------------------------------


def paper_scoped_search(
    full_ranking: Sequence[Sequence[Any]], document_ids: Sequence[str], top_k: int | None = None
) -> list[str]:
    """Dense search restricted to the given documents: the full Exact Dense ranking filtered.

    ``full_ranking`` rows are (chunk_id, document_id, ...) in the online order (similarity DESC,
    chunk_id ASC) over the whole index. Filtering it is exactly a search over the documents' own
    chunks, with no second matrix product whose float32 rounding could reorder near-ties.
    """

    wanted = set(document_ids)
    ids = [row[0] for row in full_ranking if row[1] in wanted]
    return ids if top_k is None else ids[:top_k]


# --------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------


def gold_documents(case: EvalCase) -> list[str]:
    """Documents named by the case's required evidence groups."""

    return sorted({a.document_id for g in case.required_evidence_groups for a in g.alternatives})


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _scoped_bucket(rank: int | None) -> str:
    if rank is None:
        return "not_found"
    return next((label for label, bound in (("rank_1_5", 5), ("rank_6_10", 10), ("rank_11_20", 20)) if rank <= bound),
                "rank_gt_20")


def _sorted(counter: Mapping) -> dict:
    return dict(sorted(counter.items()))


def build_step0_report(
    dataset: E2EDataset,
    mapping: E2EMapping,
    observations: Sequence[E2EObservation],
    rerank: RerankScores,
    chunk_table: Mapping[str, Mapping[str, Any]],
    oracle: Mapping[str, Mapping[str, Any]] | None,
    *,
    depths: Sequence[int] = DEPTHS,
    scoped_depths: Sequence[int] = SCOPED_DEPTHS,
    baseline: Mapping[str, E2EObservation] | None = None,
) -> dict:
    """Run the five step-0 analyses on stored artifacts and return the JSON report.

    ``rerank`` is the scores sidecar of ``observations``. ``chunk_table`` rows carry
    ``document_id``, ``label``, ``element_types``, ``heading_path`` and ``parsed_section_path``
    (written by ``scripts/e2e_step0.py chunk-types``).
    """

    depths = sorted(depths)
    cases = {case.case_id: case for case in dataset.cases}
    usable_by_case: dict[str, list] = {}
    for g in mapping.groups:
        if is_usable(g.status):
            usable_by_case.setdefault(g.case_id, []).append(g)
    unknown = sorted({c for gs in usable_by_case.values() for g in gs for b in g.chunk_sets for c in b} - set(chunk_table))
    if unknown:
        raise ValueError(f"{len(unknown)} mapped chunks are missing from the chunk table, e.g. {unknown[0]}")

    def slices_of(case_id: str) -> tuple[str, str]:
        return case_slice(cases[case_id]), "all"

    def group_chunks(g) -> set[str]:
        return {c for bundle in g.chunk_sets for c in bundle}

    # ---- 1. depth sweep + sanity check ----
    k_prod = observations[0].candidate_pool_k if observations else min(depths)
    sweep = {k: {s: Counter() for s in SLICES} for k in depths}
    latency = {k: {s: [] for s in SLICES} for k in depths}
    sanity: list[dict] = []
    outcomes: dict[tuple[str, str, int], RetrievalStage] = {}
    runs: dict[tuple[str, int], SimulatedRun] = {}
    for obs in observations:
        scored = rerank.cases[obs.case_id]
        for k in depths:
            run = runs[(obs.case_id, k)] = simulate_depth(obs, scored, k)
            for s in slices_of(obs.case_id):
                latency[k][s].append(depth_latency(scored.rerank_batch_seconds, k, rerank.header.batch_size))
            for g in usable_by_case.get(obs.case_id, []):
                outcome = outcomes[(obs.case_id, g.group_id, k)] = retrieval_stage(g.chunk_sets, run.pool_ids, run.final_ids)
                previous = outcomes.get((obs.case_id, g.group_id, k_prod))
                for s in slices_of(obs.case_id):
                    sweep[k][s][outcome] += 1
                    # Groups that a deeper pool newly admits, and how many of those reach the final Top-K.
                    if previous == "CANDIDATE_MISS" and outcome != "CANDIDATE_MISS":
                        sweep[k][s]["newly_pooled"] += 1
                        sweep[k][s]["newly_pooled_reached"] += outcome == "REACHED"
                    # Groups that reached the final Top-K at the production depth but are pushed out
                    # by the extra candidates.
                    sweep[k][s]["displaced"] += previous == "REACHED" and outcome != "REACHED"
        run_prod = runs.get((obs.case_id, obs.candidate_pool_k))
        if run_prod is not None and run_prod.reranked_ids != obs.reranked_pool:
            production_final = obs.reranked_pool[: obs.final_top_k]
            sanity.append({"case_id": obs.case_id, "final_equal": run_prod.final_ids == production_final,
                           "simulated_final": run_prod.final_ids, "production_final": production_final})
    baseline_check = None
    if baseline is not None:
        differ = [o.case_id for o in observations if o.case_id in baseline and (
            [c.chunk_id for c in o.dense_ranking] != [c.chunk_id for c in baseline[o.case_id].dense_ranking]
            or o.reranked_pool != baseline[o.case_id].reranked_pool
            or o.context_chunk_ids != baseline[o.case_id].context_chunk_ids)]
        baseline_check = {"compared": sum(o.case_id in baseline for o in observations), "differ": differ}

    depth_section = {
        str(k): {
            s: {
                "usable_groups": sum(sweep[k][s][o] for o in ("CANDIDATE_MISS", "RERANK_MISS", "REACHED")),
                "candidate_miss": sweep[k][s]["CANDIDATE_MISS"],
                "rerank_drop": sweep[k][s]["RERANK_MISS"],
                "reached_final": sweep[k][s]["REACHED"],
                f"newly_pooled_vs_k{k_prod}": sweep[k][s]["newly_pooled"],
                f"newly_pooled_reached_vs_k{k_prod}": sweep[k][s]["newly_pooled_reached"],
                f"displaced_vs_k{k_prod}": sweep[k][s]["displaced"],
                "cases": len(latency[k][s]),
                "reranker_seconds_median": percentile(latency[k][s], 50),
                "reranker_seconds_p95": percentile(latency[k][s], 95),
            }
            for s in SLICES
        }
        for k in depths
    }

    # ---- 2. truncation ----
    k0, kmax = min(depths), max(depths)
    pop: dict[str, list[int]] = {}
    trunc_cases = []
    for obs in observations:
        flags = rerank.cases[obs.case_id].rerank_truncated
        if flags is None:
            raise ValueError(f"observation {obs.case_id} has no per-pair truncation flags")
        trunc_of = {item.chunk_id: flag for item, flag in zip(obs.dense_ranking, flags)}
        trunc_cases.append({"case_id": obs.case_id, f"top{k0}": sum(flags[:k0]), f"top{kmax}": sum(flags[:kmax])})
        for k in (k0, kmax):
            run = runs[(obs.case_id, k)]
            reached, dropped = set(), set()
            for g in usable_by_case.get(obs.case_id, []):
                outcome = outcomes[(obs.case_id, g.group_id, k)]
                if outcome == "REACHED":
                    reached |= group_chunks(g) & set(run.final_ids)
                elif outcome == "RERANK_MISS":
                    dropped |= (group_chunks(g) & set(run.pool_ids)) - set(run.final_ids)
            for key, ids in ((f"top{k}_all_pairs", run.pool_ids), (f"top{k}_evidence_reached_final", reached),
                             (f"top{k}_evidence_dropped_at_rerank", dropped)):
                row = pop.setdefault(key, [0, 0])
                row[0] += sum(trunc_of[c] for c in ids)
                row[1] += len(ids)
    truncation = {
        "max_length": [rerank.header.max_length],
        f"cases_with_truncation_top{k0}": sum(c[f"top{k0}"] > 0 for c in trunc_cases),
        f"cases_with_truncation_top{kmax}": sum(c[f"top{kmax}"] > 0 for c in trunc_cases),
        f"truncated_pairs_top{k0}": sum(c[f"top{k0}"] for c in trunc_cases),
        f"truncated_pairs_top{kmax}": sum(c[f"top{kmax}"] for c in trunc_cases),
        f"max_truncated_per_case_top{k0}": max((c[f"top{k0}"] for c in trunc_cases), default=0),
        f"max_truncated_per_case_top{kmax}": max((c[f"top{kmax}"] for c in trunc_cases), default=0),
        "populations": {key: {"truncated": t, "pairs": n, "rate": _rate(t, n)} for key, (t, n) in sorted(pop.items())},
        "per_case": trunc_cases,
    }

    # ---- 3. chunk type by stage (production pool depth) ----
    label_of = {cid: row["label"] for cid, row in chunk_table.items()}
    labels_by_stage = {s: {} for s in SLICES}
    group_kind_by_stage = {s: {} for s in SLICES}
    for obs in observations:
        for g in usable_by_case.get(obs.case_id, []):
            stage = outcomes[(obs.case_id, g.group_id, obs.candidate_pool_k)]
            labels = [label_of[c] for c in sorted(group_chunks(g))]
            standalone = [lbl != "body" for lbl in labels]
            kind = "all_standalone" if all(standalone) else "has_standalone" if any(standalone) else "body_only"
            for s in slices_of(obs.case_id):
                labels_by_stage[s].setdefault(stage, Counter()).update(labels)
                group_kind_by_stage[s].setdefault(stage, Counter())[kind] += 1
    chunk_types = {
        "index_label_counts": _sorted(Counter(label_of.values())),
        "evidence_chunk_labels_by_stage": {s: {st: _sorted(c) for st, c in sorted(v.items())}
                                           for s, v in labels_by_stage.items()},
        "evidence_groups_by_composition_and_stage": {s: {st: _sorted(c) for st, c in sorted(v.items())}
                                                     for s, v in group_kind_by_stage.items()},
    }

    # ---- 4. back matter split ----
    # Two readings of a chunk's section: ``heading_path`` as indexed (body chunks only; what the
    # existing report's is_back_matter sees), and ``effective`` = heading_path, else the section
    # path that the re-parse attributes to the chunk's first element (covers standalone chunks).
    def heading_only(chunk_id: str) -> str | None:
        return chunk_table[chunk_id].get("heading_path")

    def effective(chunk_id: str) -> str | None:
        row = chunk_table[chunk_id]
        return row.get("heading_path") or row.get("parsed_section_path")

    def has_reference_entry(chunk_id: str) -> bool:
        return "reference_entry" in chunk_table[chunk_id].get("element_types", [])

    def tags_of(chunk_ids, path_of: Callable[[str], str | None]) -> set[str]:
        found = {back_matter_category(path_of(c)) for c in chunk_ids} - {None}
        # A "References"-headed chunk with no reference-list element is body text (e.g. an appendix
        # whose own headings were not detected) inheriting the References heading.
        if any(back_matter_category(path_of(c)) == "references_ack" and not has_reference_entry(c) for c in chunk_ids):
            found.add("references_ack_without_reference_entry")
        if any(map(has_reference_entry, chunk_ids)):
            found.add("has_reference_entry_element")
        if any(heading_only(c) is None for c in chunk_ids):
            found.add("no_heading_path")
        return found

    answerable = [o for o in observations if cases[o.case_id].expected_status == AnswerStatus.ANSWERED]
    back_matter: dict[str, Any] = {
        "answerable_cases": {s: sum(1 for o in answerable if s in slices_of(o.case_id)) for s in SLICES},
        "usable_groups": {s: sum(len(usable_by_case.get(o.case_id, [])) for o in observations
                                 if s in slices_of(o.case_id)) for s in SLICES},
    }
    for reading, path_of in (("heading_path", heading_only), ("effective", effective)):
        final_cases = {s: Counter() for s in SLICES}
        final_chunks = {s: Counter() for s in SLICES}
        gold = {s: Counter() for s in SLICES}
        for obs in answerable:
            final = obs.reranked_pool[: obs.final_top_k]
            present = tags_of(final, path_of)
            if present & {"references_ack", "appendix_supp"}:
                present.add("any_back_matter")
            for s in slices_of(obs.case_id):
                final_cases[s].update(present)
                final_chunks[s].update(back_matter_category(path_of(c)) or "other" for c in final)
        for obs in observations:
            for g in usable_by_case.get(obs.case_id, []):
                for s in slices_of(obs.case_id):
                    gold[s].update(tags_of(group_chunks(g), path_of))
        back_matter[reading] = {
            "cases_with_category_in_final": {s: _sorted(c) for s, c in final_cases.items()},
            "final_chunks_by_category": {s: _sorted(c) for s, c in final_chunks.items()},
            "gold_groups_with_mapped_chunk_in_category": {s: _sorted(c) for s, c in gold.items()},
            "index_chunks_by_category": _sorted(Counter(
                back_matter_category(path_of(c)) or "other" for c in chunk_table)),
            "index_references_ack_chunks_without_reference_entry": sum(
                back_matter_category(path_of(c)) == "references_ack" and not has_reference_entry(c) for c in chunk_table),
        }
    no_heading = [c for c in chunk_table if heading_only(c) is None]
    back_matter["index_chunks_without_heading_path"] = {
        "total": len(no_heading),
        "of_index": len(chunk_table),
        "by_label": _sorted(Counter(label_of[c] for c in no_heading)),
        "by_effective_category": _sorted(Counter(back_matter_category(effective(c)) or "other" for c in no_heading)),
        "without_any_section": sum(effective(c) is None for c in no_heading),
    }

    # ---- 5. oracle paper-scoped retrieval ----
    oracle_section = None
    if oracle is not None:
        doc_of = {cid: row["document_id"] for cid, row in chunk_table.items()}
        variants = [((f"top{k_prod}" if p == 0 else f"top{k_prod}+scoped_p{p}") + ("+reserve" if r else ""), p, r)
                    for r in (0, 1) for p in (0, *scoped_depths)]
        buckets = {s: Counter() for s in SLICES}
        miss_buckets = {s: Counter() for s in SLICES}
        recall = {s: Counter() for s in SLICES}
        groups_total = Counter()
        multi_paper_cases = Counter()
        dense_top_mismatch = []
        per_group = []
        for obs in observations:
            usable = usable_by_case.get(obs.case_id, [])
            record = oracle.get(obs.case_id)
            if not usable:
                continue
            if record is None:
                raise ValueError(f"oracle has no record for case {obs.case_id} with usable groups")
            if not record.get("dense_top_matches_observation", True):
                dense_top_mismatch.append(obs.case_id)
            ranking = record["scoped_ranking"]
            scores = dict(zip((item.chunk_id for item in obs.dense_ranking), rerank.cases[obs.case_id].rerank_scores))
            scores.update({c: float(v) for c, v in record.get("extra_rerank_scores", {}).items()})
            per_paper = {doc: paper_scoped_search(ranking, [doc]) for doc in record["gold_documents"]}
            dense_ids = [item.chunk_id for item in obs.dense_ranking]
            for s in slices_of(obs.case_id):
                multi_paper_cases[s] += len(per_paper) >= 2
            # Global pool ∪ each gold paper's scoped Top-p; r=1 reserves one final slot per gold
            # paper (step 1 applies it only with ≥2 papers).
            finals = {name: run_variant(dense_ids, per_paper, sorted(per_paper), scores, doc_of, k=obs.candidate_pool_k,
                                        p=p, final_k=obs.final_top_k, r=r).final_ids
                      for name, p, r in variants}
            case = cases[obs.case_id]
            for g in usable:
                group = next(x for x in case.required_evidence_groups if x.group_id == g.group_id)
                docs = sorted({a.document_id for a in group.alternatives})
                rank = bundle_rank(g.chunk_sets, paper_scoped_search(ranking, docs))
                stage = outcomes[(obs.case_id, g.group_id, obs.candidate_pool_k)]
                row = {"case_id": obs.case_id, "group_id": g.group_id, "within_paper_rank": rank,
                       f"stage_at_k{obs.candidate_pool_k}": stage,
                       **{name: fully_in(g.chunk_sets, final) for name, final in finals.items()}}
                per_group.append(row)
                for s in slices_of(obs.case_id):
                    buckets[s][_scoped_bucket(rank)] += 1
                    if stage == "CANDIDATE_MISS":
                        miss_buckets[s][_scoped_bucket(rank)] += 1
                    groups_total[s] += 1
                    for name, _, _ in variants:
                        recall[s][name] += row[name]
        oracle_section = {
            "label": "ORACLE paper linking: scoped by gold document IDs (upper bound, not deployable)",
            "within_paper_rank_buckets": {s: _sorted(c) for s, c in buckets.items()},
            f"within_paper_rank_buckets_candidate_miss_at_k{k_prod}": {s: _sorted(c) for s, c in miss_buckets.items()},
            "cases_with_2plus_gold_papers": {s: multi_paper_cases[s] for s in SLICES},
            "dense_top_mismatch_cases": dense_top_mismatch,
            "group_recall_at_final_top_k": {
                s: {name: {"reached": recall[s][name], "groups": groups_total[s],
                           "recall": _rate(recall[s][name], groups_total[s])} for name, _, _ in variants}
                for s in SLICES
            },
            "per_group": per_group,
        }

    return {
        "schema_version": STEP0_REPORT_SCHEMA_VERSION,
        "cases": len(observations),
        "slice_definition": {"multi_doc": SLICE_DEFINITION},
        "depths": list(depths),
        "candidate_pool_k": k_prod,
        "final_top_k": sorted({o.final_top_k for o in observations}),
        "unusable_groups": sum(not is_usable(g.status) for g in mapping.groups
                               if g.case_id in {o.case_id for o in observations}),
        "depth_sweep": depth_section,
        "sanity_production_k_matches_simulation": {
            "checked": len(observations),
            "order_mismatches": sanity,
            "final_mismatches": sum(not m["final_equal"] for m in sanity),
        },
        "baseline_observation_check": baseline_check,
        "truncation": truncation,
        "chunk_types": chunk_types,
        "back_matter": back_matter,
        "oracle_scoped": oracle_section,
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _counts(counter: Mapping) -> str:
    return ", ".join(f"{k} {v}" for k, v in counter.items()) or "-"


def render_step0_markdown(report: dict, *, commands: Sequence[str] = ()) -> str:
    """Tables for the five analyses; findings are written by the caller below these tables."""

    head = "| | multi-doc | single-doc | all |\n| --- | --- | --- | --- |"

    def row(label: str, values: Mapping[str, Any]) -> str:
        return f"| {label} | " + " | ".join(_fmt(values.get(s, 0)) for s in SLICES) + " |"

    sd = report["slice_definition"]
    kp = report["candidate_pool_k"]
    lines = [
        "# Step 0 retrieval diagnostics (no LLM)", "",
        f"Cases observed: {report['cases']}. Slices: multi-doc = {sd['multi_doc']}; "
        f"single-doc = every other case, unanswerable ones included (they have no usable groups). Unusable (unmapped) groups excluded: "
        f"{report['unusable_groups']}.", "",
        "## 1. Depth sweep (one scoring pass, simulated K, final top-5)", "",
        f"| K | slice | usable groups | candidate miss | rerank drop | reached final | newly pooled vs K={kp} "
        f"| of which reached | displaced vs K={kp} | reranker s/case median | p95 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for k, by_slice in report["depth_sweep"].items():
        for s in SLICES:
            r = by_slice[s]
            lines.append(f"| {k} | {s} | {r['usable_groups']} | {r['candidate_miss']} | {r['rerank_drop']} | "
                         f"{r['reached_final']} | {r[f'newly_pooled_vs_k{kp}']} | {r[f'newly_pooled_reached_vs_k{kp}']} | "
                         f"{r[f'displaced_vs_k{kp}']} | "
                         f"{_fmt(r['reranker_seconds_median'])} | {_fmt(r['reranker_seconds_p95'])} |")
    san = report["sanity_production_k_matches_simulation"]
    lines += ["", f"Sanity: simulated K={kp} vs production on {san['checked']} cases: "
              f"{san['final_mismatches']} final top-5 mismatches, {len(san['order_mismatches'])} full-order mismatches."]
    for m in san["order_mismatches"]:
        lines.append(f"- {m['case_id']}: final equal = {m['final_equal']}")
    if report.get("baseline_observation_check"):
        b = report["baseline_observation_check"]
        lines.append(f"\nNew observations vs baseline obs_retrieval.jsonl (dense ids, reranked pool, context): "
                     f"{b['compared']} compared, {len(b['differ'])} differ.")

    t = report["truncation"]
    kmin, kmax = min(report["depths"]), max(report["depths"])
    lines += ["", "## 2. Reranker truncation", "",
              f"max_length = {t['max_length']}. Truncated pairs: top-{kmin} {t[f'truncated_pairs_top{kmin}']} "
              f"in {t[f'cases_with_truncation_top{kmin}']} cases (max {t[f'max_truncated_per_case_top{kmin}']}/case); "
              f"top-{kmax} {t[f'truncated_pairs_top{kmax}']} in {t[f'cases_with_truncation_top{kmax}']} cases "
              f"(max {t[f'max_truncated_per_case_top{kmax}']}/case).", "",
              "| population (evidence = mapped chunks of usable groups) | truncated | pairs | rate |",
              "| --- | --- | --- | --- |"]
    for key, r in t["populations"].items():
        lines.append(f"| {key} | {r['truncated']} | {r['pairs']} | {_fmt(r['rate'])} |")

    ct = report["chunk_types"]
    lines += ["", f"## 3. Chunk type of evidence (stages at production K={kp})", "",
              "Index: " + _counts(ct["index_label_counts"]), "",
              "| slice | stage | evidence chunk labels | groups by composition |", "| --- | --- | --- | --- |"]
    for s in SLICES:
        for stage, labels in ct["evidence_chunk_labels_by_stage"][s].items():
            lines.append(f"| {s} | {stage} | {_counts(labels)} | "
                         f"{_counts(ct['evidence_groups_by_composition_and_stage'][s].get(stage, {}))} |")

    bm = report["back_matter"]
    lines += ["", "## 4. Back matter split", "",
              "`heading_path` = the indexed heading path (body chunks only; the existing report's reading). "
              "`effective` = heading_path, else the section the re-parse assigns to the chunk's first element.", ""]
    for reading in ("heading_path", "effective"):
        r = bm[reading]
        lines += [f"### Reading: {reading}", "", "Answerable cases with ≥1 final top-5 chunk in the category:", "", head,
                  row("answerable cases", bm["answerable_cases"])]
        for c in ("references_ack", "references_ack_without_reference_entry", "appendix_supp", "any_back_matter",
                  "has_reference_entry_element", "no_heading_path"):
            lines.append(row(c, {s: r["cases_with_category_in_final"][s].get(c, 0) for s in SLICES}))
        lines += ["", "Final top-5 chunks by category:", "", head]
        for c in ("references_ack", "appendix_supp", "other"):
            lines.append(row(c, {s: r["final_chunks_by_category"][s].get(c, 0) for s in SLICES}))
        lines += ["", "Usable gold groups with ≥1 mapped chunk in the category:", "", head,
                  row("usable groups", bm["usable_groups"])]
        for c in ("references_ack", "references_ack_without_reference_entry", "appendix_supp",
                  "has_reference_entry_element", "no_heading_path"):
            lines.append(row(c, {s: r["gold_groups_with_mapped_chunk_in_category"][s].get(c, 0) for s in SLICES}))
        lines += ["", "Index chunks by category: " + _counts(r["index_chunks_by_category"])
                  + f"; references_ack chunks without any reference_entry element: "
                  f"{r['index_references_ack_chunks_without_reference_entry']}.", ""]
    nh = bm["index_chunks_without_heading_path"]
    lines.append(f"Index chunks without heading_path: {nh['total']} of {nh['of_index']} (by label: {_counts(nh['by_label'])}; "
                 f"by effective category: {_counts(nh['by_effective_category'])}; no section at all: {nh['without_any_section']}).")

    oc = report.get("oracle_scoped")
    if oc:
        lines += ["", "## 5. ORACLE paper-scoped retrieval (uses gold document IDs; upper bound only)", "",
                  "Within-paper Dense rank of the evidence bundle (original query, search restricted to the group's "
                  "gold document(s); a set ranks at its last chunk):", "", head]
        miss = oc[f"within_paper_rank_buckets_candidate_miss_at_k{kp}"]
        for b in ("rank_1_5", "rank_6_10", "rank_11_20", "rank_gt_20", "not_found"):
            lines.append(row(b, {s: oc["within_paper_rank_buckets"][s].get(b, 0) for s in SLICES}))
        lines += ["", f"Same, only groups that are a candidate miss at K={kp}:", "", head]
        for b in ("rank_1_5", "rank_6_10", "rank_11_20", "rank_gt_20", "not_found"):
            lines.append(row(b, {s: miss[s].get(b, 0) for s in SLICES}))
        lines += ["", f"Cases with ≥2 gold papers (reserve variants apply only there): "
                  f"{_counts(oc['cases_with_2plus_gold_papers'])}. Dense top mismatch vs observation: "
                  f"{len(oc['dense_top_mismatch_cases'])} cases.", "",
                  "Group recall at final top-5 (pool = global top-20 ∪ per-gold-paper top-p; reranked with stored scores):",
                  "", head]
        for name, r in oc["group_recall_at_final_top_k"]["all"].items():
            cells = [f"{oc['group_recall_at_final_top_k'][s][name]['reached']}/"
                     f"{oc['group_recall_at_final_top_k'][s][name]['groups']} "
                     f"({_fmt(oc['group_recall_at_final_top_k'][s][name]['recall'])})" for s in SLICES]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if commands:
        lines += ["", "## Reproduce", "", "```powershell", *commands, "```"]
    return "\n".join(lines) + "\n"
