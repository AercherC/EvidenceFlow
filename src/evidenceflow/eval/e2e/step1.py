"""Step-1 offline retrieval experiments for rag_e2e_v1 (no LLM), built on the step-0 artifacts.

Every variant re-sorts one base run (``work/step0/obs_step0.jsonl`` and its scores sidecar) with
stored Reranker scores; extra chunks are scored only where a variant pools them. Three experiments:
A. pool composition: global Dense Top-K ∪ per-linked-paper scoped Top-p, final Top-5, with paper
   linking either ORACLE (gold document IDs) or the name matcher below;
B. reranker swap: the same pools rescored by another cross-encoder (scores come from
   ``scripts/e2e_step1.py rescore``; this module only consumes them);
C. multi-document final selection: r reserved slots per linked paper × final k, with the context
   token total of every final set.

The matcher is the first realistic name-to-paper linker. It lives here, in eval code, on purpose:
it is a diagnostic, not a production component. Every function is pure.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.eval.e2e.diagnosis import fully_in
from evidenceflow.eval.e2e.slices import SLICES
from evidenceflow.retrieval.reranker import rank_scored_candidates

STEP1_REPORT_SCHEMA_VERSION = "rag-e2e-step1-report-v1"
LINKINGS = ("oracle", "matcher")
GLOBAL_DEPTHS = (20, 50)
SCOPED_P = (0, 5, 10)
RESERVE_R = (0, 1, 2)
FINAL_KS: tuple[int | str, ...] = (5, 6, 8, "adaptive")
ADAPTIVE_PER_PAPER = 3
BASELINE = {"k": 20, "p": 0, "linking": "oracle", "final_k": 5, "r": 0}

# ASCII 字母数字作为 token 边界：Python 的 \b 把 CJK 也当成 \w，中文问句里紧贴中文的方法名会匹配失败。
_TOKEN_BEFORE = r"(?<![A-Za-z0-9])"
_TOKEN_AFTER = r"(?![A-Za-z0-9])"


# --------------------------------------------------------------------------------------
# Name matcher
# --------------------------------------------------------------------------------------


def method_name(title: str) -> str:
    """The method name of an academic title: the text before the first ':' ("PCN: Point ...")."""

    name = title.split(":", 1)[0].strip()
    if not name or name == title.strip():
        raise ValueError(f"title has no 'Method: subtitle' form: {title!r}")
    return name


def method_aliases(papers: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """alias (lower case) -> document_id, with hyphen variants derived mechanically.

    "Hyper-PCN" also yields "hyperpcn" and "hyper pcn"; no per-case tuning. Two papers sharing an
    alias is a configuration error, not something to resolve silently.
    """

    aliases: dict[str, str] = {}
    for paper in papers:
        name = method_name(paper["title"])
        for variant in {name, name.replace("-", ""), name.replace("-", " ")}:
            key = variant.lower()
            if aliases.setdefault(key, paper["document_id"]) != paper["document_id"]:
                raise ValueError(f"alias {variant!r} names two documents")
    return aliases


def resolve_manifest_documents(
    papers: Sequence[Mapping[str, Any]], index_documents: Mapping[str, str]
) -> tuple[list[dict], dict]:
    """Map manifest papers onto the active index's document IDs by file_name.

    ``index_documents`` maps file_name -> document_id of the index. Returns the papers with index
    document IDs and a check record (how many IDs already matched, how many were remapped).
    """

    resolved, remapped, missing = [], [], []
    for paper in papers:
        doc = index_documents.get(paper["file_name"])
        if doc is None:
            missing.append(paper["file_name"])
            continue
        if doc != paper["document_id"]:
            remapped.append(paper["file_name"])
        resolved.append({**paper, "document_id": doc})
    if missing:
        raise ValueError(f"manifest papers missing from the index: {missing}")
    return resolved, {"papers": len(papers), "ids_equal": len(papers) - len(remapped), "remapped_by_file_name": remapped}


@dataclass(frozen=True)
class LinkHit:
    """One accepted alias match in the query."""

    surface: str
    document_id: str
    start: int


def link_papers(query: str, aliases: Mapping[str, str]) -> list[LinkHit]:
    """Case-insensitive, whole-token, longest-match-first name linking.

    All alias matches are collected, then accepted from the longest down; a match that overlaps an
    accepted one is dropped, so "Hyper-PCN" does not also link PCN. "AdaPoinTr" never contains a
    whole-token "PoinTr" in the first place. Hits come back in query order.
    """

    found = []
    for alias, doc in aliases.items():
        pattern = re.compile(_TOKEN_BEFORE + re.escape(alias) + _TOKEN_AFTER, re.I)
        found += [(m.start(), m.end(), doc) for m in pattern.finditer(query)]
    accepted: list[tuple[int, int, str]] = []
    for start, end, doc in sorted(found, key=lambda m: (m[0] - m[1], m[0])):
        if all(end <= s or start >= e for s, e, _ in accepted):
            accepted.append((start, end, doc))
    return [LinkHit(query[s:e], d, s) for s, e, d in sorted(accepted)]


def linked_documents(hits: Sequence[LinkHit]) -> list[str]:
    """Distinct linked documents, sorted (pool order uses sorted documents, like step 0)."""

    return sorted({h.document_id for h in hits})


def evaluate_matcher(
    cases: Sequence[tuple[str, str, Sequence[str]]],
    aliases: Mapping[str, str],
    *,
    name_of: Mapping[str, str],
    ambiguous_name: str = "PCN",
) -> dict:
    """Per-case precision/recall of the matcher against the gold documents, plus every error.

    ``cases`` rows are (case_id, query, gold_documents). A wrong link is a linked document not
    among the gold ones; it counts as an ``ambiguous_name`` error when the hit that produced it
    was that name (PCN is both a paper and the benchmark dataset).
    """

    per_case, wrong, missed = [], [], []
    tp = n_linked = n_gold = 0
    for case_id, query, gold in cases:
        hits = link_papers(query, aliases)
        linked = linked_documents(hits)
        gold_set = set(gold)
        hit_ok = len(gold_set & set(linked))
        tp, n_linked, n_gold = tp + hit_ok, n_linked + len(linked), n_gold + len(gold_set)
        for doc in sorted(set(linked) - gold_set):
            surfaces = sorted({h.surface for h in hits if h.document_id == doc})
            wrong.append({"case_id": case_id, "linked": name_of[doc], "surface": surfaces,
                          "gold": [name_of[d] for d in sorted(gold_set)], "query": query,
                          "ambiguous_name": name_of[doc] == ambiguous_name})
        for doc in sorted(gold_set - set(linked)):
            missed.append({"case_id": case_id, "missed": name_of[doc], "query": query})
        per_case.append({"case_id": case_id, "linked": [name_of[d] for d in linked],
                         "gold": [name_of[d] for d in sorted(gold_set)],
                         "precision": round(hit_ok / len(linked), 4) if linked else None,
                         "recall": round(hit_ok / len(gold_set), 4) if gold_set else None})
    with_gold = [r for r in per_case if r["recall"] is not None]
    return {
        "micro_precision": round(tp / n_linked, 4) if n_linked else None,
        "micro_recall": round(tp / n_gold, 4) if n_gold else None,
        "cases": len(per_case),
        "cases_with_gold": len(with_gold),
        "cases_exact": sum(r["linked"] == r["gold"] for r in with_gold),
        "cases_linked_nothing": sum(not r["linked"] for r in per_case),
        "cases_linked_nothing_with_gold": sum(not r["linked"] for r in with_gold),
        "wrong_links": wrong,
        "wrong_links_from_ambiguous_name": sum(w["ambiguous_name"] for w in wrong),
        "missed_links": missed,
        "per_case": per_case,
    }


# --------------------------------------------------------------------------------------
# Pool composition and final selection
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimulatedRun:
    """Candidate pool, Reranker ordering and final set of one simulated configuration."""

    pool_ids: list[str]
    reranked_ids: list[str]
    final_ids: list[str]


def rerank_ids(candidates: Sequence[tuple[str, str]], scores: Sequence[float]) -> list[str]:
    """Order (chunk_id, document_id) candidates by precomputed scores with the online tie rule.

    The chunks handed to ``rank_scored_candidates`` are identity-only stubs: the sort reads only
    the candidate order, the score and chunk_id.
    """

    stubs = [DocumentChunk(chunk_id=c, document_id=d, content="", token_count=0, chunk_index=0,
                           file_name="-", content_hash="-") for c, d in candidates]
    return [item.chunk.chunk_id for item in rank_scored_candidates(stubs, list(scores))]


def merge_pool(global_ids: Sequence[str], extra_ids: Sequence[str]) -> list[str]:
    """Union of the global Dense pool and scoped extras, deduplicated.

    The global pool keeps its Dense order and extras follow in the order given; order only matters
    for Reranker ties.
    """

    return list(dict.fromkeys([*global_ids, *extra_ids]))


def compose_pool(dense_ids: Sequence[str], k: int, per_paper: Mapping[str, Sequence[str]],
                 linked: Sequence[str], p: int) -> list[str]:
    """Global Dense Top-K ∪ each linked paper's scoped Top-p (sorted documents, as in step 0)."""

    if len(dense_ids) < k:
        raise ValueError(f"dense ranking shorter than K={k}")
    missing = [d for d in linked if p and d not in per_paper]
    if missing:
        raise ValueError(f"no scoped ranking for linked documents {missing}")
    return merge_pool(dense_ids[:k], [c for d in sorted(linked) for c in (list(per_paper[d])[:p] if p else [])])


def final_k_for(final_k: int | str, n_linked: int) -> int:
    """Fixed k, or "adaptive" = 3 × linked papers with a floor of 5."""

    if final_k == "adaptive":
        return max(5, ADAPTIVE_PER_PAPER * n_linked)
    if isinstance(final_k, bool) or not isinstance(final_k, int) or final_k <= 0:
        raise ValueError("final_k must be a positive integer or 'adaptive'")
    return final_k


def select_reserved(ranked_ids: Sequence[str], doc_of: Mapping[str, str], *, k: int,
                    documents: Sequence[str], r: int) -> list[str]:
    """Final Top-k reserving each listed document's best ``r`` chunks; only with ≥2 documents.

    Reservations are granted level by level (every document's 1st chunk, then every 2nd) in rank
    order. Unused slots fill in rank order; the result keeps rank order.
    """

    if r <= 0 or len(set(documents)) < 2:
        return list(ranked_ids[:k])
    position = {c: i for i, c in enumerate(ranked_ids)}
    wanted = set(documents)
    level: dict[str, int] = {}
    reserved = []
    for chunk_id in ranked_ids:
        doc = doc_of.get(chunk_id)
        if doc in wanted and level.get(doc, 0) < r:
            reserved.append((level.get(doc, 0), position[chunk_id], chunk_id))
            level[doc] = level.get(doc, 0) + 1
    chosen = {c for _, _, c in sorted(reserved)[:k]}
    for chunk_id in ranked_ids:
        if len(chosen) >= k:
            break
        chosen.add(chunk_id)
    return [c for c in ranked_ids if c in chosen]


def run_variant(dense_ids: Sequence[str], per_paper: Mapping[str, Sequence[str]], linked: Sequence[str],
                scores: Mapping[str, float], doc_of: Mapping[str, str], *, k: int, p: int,
                final_k: int | str, r: int) -> SimulatedRun:
    """One case under one configuration: pool -> stored scores (online tie rule) -> final set."""

    pool = compose_pool(dense_ids, k, per_paper, linked, p)
    missing = [c for c in pool if c not in scores]
    if missing:
        raise ValueError(f"no reranker score for {len(missing)} pooled chunks, e.g. {missing[0]}")
    reranked = rerank_ids([(c, doc_of[c]) for c in pool], [scores[c] for c in pool])
    final = select_reserved(reranked, doc_of, k=final_k_for(final_k, len(linked)), documents=linked, r=r)
    return SimulatedRun(pool, reranked, final)


def context_tokens(final_ids: Sequence[str], entry_tokens: Mapping[str, int], separator_tokens: int,
                   label_delta: Sequence[int] = ()) -> int:
    """ContextBuilder total for a final set: rendered entries plus the separators between them.

    ``entry_tokens`` are measured with label "[S1]"; ``label_delta[i]`` corrects the entry at
    position i (the BGE-M3 tokenizer splits "[S4]" into more pieces than "[S1]").
    """

    if len(final_ids) > len(label_delta) and label_delta:
        raise ValueError("label_delta is shorter than the final set")
    extra = sum(label_delta[: len(final_ids)]) if label_delta else 0
    return sum(entry_tokens[c] for c in final_ids) + separator_tokens * max(0, len(final_ids) - 1) + extra


def rerank_seconds(pairs: int, batch_seconds: Sequence[float], batch_size: int) -> float:
    """Estimated scoring time: batches needed × this case's mean step-0 batch time."""

    if not batch_seconds:
        return 0.0
    return math.ceil(pairs / batch_size) * sum(batch_seconds) / len(batch_seconds)


def union_to_score(dense_ids: Sequence[str], per_paper: Mapping[str, Sequence[str]],
                   linked_sets: Sequence[Sequence[str]], *, max_k: int, max_p: int) -> list[str]:
    """Every chunk that some A/C variant pools for a case: Dense Top-max_k ∪ linked Top-max_p."""

    docs = sorted({d for linked in linked_sets for d in linked})
    return compose_pool(dense_ids, max_k, per_paper, docs, max_p)


# --------------------------------------------------------------------------------------
# Reranker swap (B)
# --------------------------------------------------------------------------------------


def score_reproduction(stored: Mapping[str, Mapping[str, float]], rescored: Mapping[str, Mapping[str, float]]) -> dict:
    """Max |stored - rescored| over every chunk both sides scored, per case and overall."""

    diffs = {case: max((abs(stored[case][c] - s) for c, s in scores.items() if c in stored.get(case, {})), default=0.0)
             for case, scores in rescored.items()}
    compared = sum(1 for case, scores in rescored.items() for c in scores if c in stored.get(case, {}))
    worst = max(diffs, key=diffs.get, default=None)
    return {"pairs_compared": compared, "max_abs_diff": max(diffs.values(), default=0.0), "worst_case": worst}


# --------------------------------------------------------------------------------------
# Variant evaluation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseInput:
    """Everything one case needs to run any variant; built once by the CLI."""

    case_id: str
    slice: str
    groups: list[tuple[str, list[list[str]]]]  # (group_id, chunk_sets) of usable groups
    dense_ids: list[str]
    per_paper: dict[str, list[str]]
    links: dict[str, list[str]]  # linking mode -> linked documents
    batch_seconds: list[float]
    batch_size: int
    scores: dict[str, float] = field(default_factory=dict)


def variant_name(v: Mapping[str, Any]) -> str:
    """Stable label, e.g. ``K50+p10/matcher/k8/r1``; linking shows whenever p or r uses it."""

    base = f"K{v['k']}" + (f"+p{v['p']}" if v["p"] else "") + (f"/{v['linking']}" if v["p"] or v["r"] else "")
    fk = "A" if v["final_k"] == "adaptive" else v["final_k"]
    return base + f"/k{fk}" + (f"/r{v['r']}" if v["r"] else "")


def evaluate_variant(inputs: Sequence[CaseInput], doc_of: Mapping[str, str], v: Mapping[str, Any]) -> dict:
    """Run one configuration over every case: reached groups, final sets, pool sizes, cost."""

    reached, finals, pairs, seconds = set(), {}, {}, {}
    for case in inputs:
        linked = case.links[v["linking"]]
        run = run_variant(case.dense_ids, case.per_paper, linked, case.scores, doc_of,
                          k=v["k"], p=v["p"], final_k=v["final_k"], r=v["r"])
        finals[case.case_id] = run.final_ids
        pairs[case.case_id] = len(run.pool_ids)
        seconds[case.case_id] = rerank_seconds(len(run.pool_ids), case.batch_seconds, case.batch_size)
        reached |= {(case.case_id, g) for g, bundles in case.groups if fully_in(bundles, set(run.final_ids))}
    return {"variant": dict(v), "reached": reached, "finals": finals, "pairs": pairs, "seconds": seconds}


def summarize(result: Mapping[str, Any], baseline: Mapping[str, Any], inputs: Sequence[CaseInput], *,
              entry_tokens: Mapping[str, int], separator_tokens: int, max_context_tokens: int,
              label_delta: Sequence[int] = ()) -> dict:
    """Report row of one evaluated variant against the production baseline."""

    slice_of = {c.case_id: c.slice for c in inputs}
    groups = [(c.case_id, g) for c in inputs for g, _ in c.groups]
    gained = sorted(result["reached"] - baseline["reached"])
    lost = sorted(baseline["reached"] - result["reached"])

    def by_slice(items) -> dict:
        return {s: sum(1 for case, _ in items if s == "all" or slice_of[case] == s) for s in SLICES}

    tokens = {case: context_tokens(final, entry_tokens, separator_tokens, label_delta) for case, final in result["finals"].items()}
    pairs, secs = list(result["pairs"].values()), list(result["seconds"].values())
    return {
        "name": variant_name(result["variant"]),
        **result["variant"],
        "groups": by_slice(groups),
        "reached": by_slice(result["reached"]),
        "gained_vs_baseline": [list(x) for x in gained],
        "lost_vs_baseline": [list(x) for x in lost],
        "lost_single_doc": sum(slice_of[c] == "single_doc" for c, _ in lost),
        "pairs_mean": round(sum(pairs) / len(pairs), 2),
        "pairs_max": max(pairs),
        "rerank_seconds_mean": round(sum(secs) / len(secs), 4),
        "rerank_seconds_max": round(max(secs), 4),
        "context_tokens_mean": round(sum(tokens.values()) / len(tokens), 1),
        "context_tokens_max": max(tokens.values()),
        "sets_over_budget": sorted(c for c, t in tokens.items() if t > max_context_tokens),
    }


def experiment_a(inputs, doc_of, *, depths=GLOBAL_DEPTHS, ps=SCOPED_P) -> list[dict]:
    """A: global K × scoped p × linking at final Top-5 (p=0 needs no linking: run once)."""

    variants = [{"k": k, "p": p, "linking": linking, "final_k": 5, "r": 0}
                for k in depths for p in ps for linking in (LINKINGS if p else ("oracle",))]
    return [evaluate_variant(inputs, doc_of, v) for v in variants]


def best_pools(rows: Sequence[Mapping[str, Any]], linking: str, n: int = 2) -> list[tuple[int, int]]:
    """Top-n scoped (K, p) pools of one linking mode: most groups reached, then fewest pairs."""

    scoped = [r for r in rows if r["p"] and r["linking"] == linking]
    scoped.sort(key=lambda r: (-r["reached"]["all"], r["lost_single_doc"], r["pairs_mean"]))
    return [(r["k"], r["p"]) for r in scoped[:n]]


def experiment_c(inputs, doc_of, pools: Sequence[tuple[int, int]], *, rs=RESERVE_R, final_ks=FINAL_KS) -> list[dict]:
    """C: r reserved slots per linked paper × final k, on the given pools and both linking modes.

    A p=0 pool is a control (does a larger k alone do it?): linking then only picks the papers r
    reserves for, and with r=0 the two modes coincide, so that combination runs once.
    """

    variants = [{"k": k, "p": p, "linking": linking, "final_k": fk, "r": r}
                for k, p in pools for linking in LINKINGS for r in rs for fk in final_ks
                if p or r or linking == "oracle"]
    return [evaluate_variant(inputs, doc_of, v) for v in variants]


def compare_score_sources(a: Sequence[Mapping[str, Any]], b: Sequence[Mapping[str, Any]]) -> dict:
    """Same variants under two score sources: reached-set and final-set differences."""

    out = {"variants": len(a), "reached_differs": [], "final_set_differences": 0}
    for x, y in zip(a, b, strict=True):
        if x["variant"] != y["variant"]:
            raise ValueError("variant lists are not aligned")
        if x["reached"] != y["reached"]:
            out["reached_differs"].append(variant_name(x["variant"]))
        out["final_set_differences"] += sum(set(x["finals"][c]) != set(y["finals"][c]) for c in x["finals"])
    return out


def recommend(rows: Sequence[Mapping[str, Any]], linking: str, *, allow_single_doc_loss: bool = False) -> dict | None:
    """Best row of one linking mode with no single-doc loss and no over-budget set.

    Rows without linking (p=0) count for either mode. Ties prefer more groups (all, then
    multi-doc), then fewer reranker pairs, then fewer context tokens. ``allow_single_doc_loss``
    is the fallback when nothing passes: it drops the no-loss rule, ranks by fewest single-doc
    losses first and leaves remaining ties in row order (no context-token tie-break). It never
    passes the promotion bar; it only shows how close a reranker gets.
    """

    def key(r: Mapping[str, Any]) -> tuple:
        rank = (r["lost_single_doc"], -r["reached"]["all"], -r["reached"]["multi_doc"], r["pairs_mean"])
        return rank if allow_single_doc_loss else (*rank, r["context_tokens_mean"])

    ok = [r for r in rows if (r["linking"] == linking or not r["p"]) and not r["sets_over_budget"]
          and (allow_single_doc_loss or r["lost_single_doc"] == 0)]
    return min(ok, key=key, default=None)
