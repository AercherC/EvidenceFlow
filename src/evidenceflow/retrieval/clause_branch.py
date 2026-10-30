"""Clause-branch retrieval (W = R2_c25_5en_rr_k8 from the round-2 pre-registration).

A query that the R2 rule splits into 2-3 clauses gets one branch per clause plus the original query:
- each clause branch pools the clause's dense top-c, plus its BM25 top-b when the clause has no CJK
  character, and is reranked by the clause itself;
- the original branch takes the original query's dense top-O, O = min(reranker_input_k,
  pair_budget - sum of clause pool sizes), reranked by the original query;
- the final list is a round-robin over the ranked branches (clauses in order, then the original),
  skipping chunks already chosen, up to final_top_k.
Every other query goes to the production DenseRerankerRetriever unchanged.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass

from evidenceflow.domain.chunk import RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.retrieval.clause_bm25 import BM25Index
from evidenceflow.retrieval.reranker import DenseRerankerRetriever, RerankerError
from evidenceflow.retrieval.retriever import InvalidQueryError
from evidenceflow.tracing import TraceRecorder, resolve_recorder, safe_record

logger = logging.getLogger(__name__)

# Round-1 split rule (fixed 2026-10-27, work/agent_design/candidate_miss/split_rule.py), unchanged:
# 1. cut after ? ？ ； ; 。 and after '.' followed by whitespace and an uppercase ASCII letter;
# 2. cut before an explicit second-question marker that follows a comma (zh and en lists below);
# 3. strip, drop pieces shorter than 6 characters.
_SENTENCE = re.compile(r"(?<=[?？；;。])|(?<=\.)(?=\s+[A-Z])")
_ZH_MARKER = re.compile(r"(?<=[，,])\s*(?=另外|还有|另一篇|另一个|又)")
_EN_MARKER = re.compile(
    r"(?<=,)\s*(?=(?i:also,|another\b|a different\b|and what about\b"
    r"|and (?:how|what|which|where|when|why|at how|does)\b))"
)
MIN_CLAUSE_CHARS = 6
MAX_CLAUSES = 3


def split_clauses(query: str) -> list[str] | None:
    """Apply the R2 rule; return the clauses when it fires (>= 2 remain), otherwise None.

    R2 = the round-1 split, then every piece ending in ';' or '；' is joined to the next piece
    (repeatedly), then pieces 3..n are joined into one third piece.
    """

    pieces = _SENTENCE.split(query)
    for marker in (_ZH_MARKER, _EN_MARKER):
        pieces = [part for piece in pieces for part in marker.split(piece)]
    pieces = [piece.strip() for piece in pieces]
    pieces = [piece for piece in pieces if len(piece) >= MIN_CLAUSE_CHARS]
    if len(pieces) < 2:
        return None
    while True:
        index = next(
            (i for i in range(len(pieces) - 1) if pieces[i].endswith((";", "；"))), None
        )
        if index is None:
            break
        pieces[index : index + 2] = [pieces[index] + " " + pieces[index + 1]]
    if len(pieces) > MAX_CLAUSES:
        pieces = pieces[: MAX_CLAUSES - 1] + [" ".join(pieces[MAX_CLAUSES - 1 :])]
    return pieces if len(pieces) >= 2 else None


def has_cjk(text: str) -> bool:
    """CJK symbols/ideographs (U+3000–U+9FFF) or full-width forms (U+FF00–U+FFEF)."""

    return any(0x3000 <= ord(ch) <= 0x9FFF or 0xFF00 <= ord(ch) <= 0xFFEF for ch in text)


def round_robin(branches: Sequence[Sequence[str]], k: int) -> list[str]:
    """Each round, every branch in order adds its best chunk not chosen yet, until k are chosen."""

    chosen: list[str] = []
    seen: set[str] = set()
    pointers = [0] * len(branches)
    while len(chosen) < k:
        progressed = False
        for i, ids in enumerate(branches):
            if len(chosen) >= k:
                break
            while pointers[i] < len(ids) and ids[pointers[i]] in seen:
                pointers[i] += 1
            if pointers[i] < len(ids):
                chosen.append(ids[pointers[i]])
                seen.add(ids[pointers[i]])
                pointers[i] += 1
                progressed = True
        if not progressed:
            break
    return chosen


@dataclass(frozen=True)
class _Branch:
    """One reranked branch: its query, kind, pool composition and ranked results."""

    query: str
    kind: str  # "clause" or "original"
    dense_count: int
    bm25_count: int
    ranked: list[RerankedRetrievedChunk]


class ClauseBranchRetriever:
    """Route multi-clause queries through clause branches; delegate the rest to production retrieval."""

    retrieval_strategy = "clause_branch"

    def __init__(
        self,
        base: DenseRerankerRetriever,
        bm25: BM25Index | None,
        *,
        dense_depth: int,
        clause_pool_k: int = 25,
        bm25_k: int = 5,
        pair_budget: int = 100,
        final_top_k: int = 8,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """``dense_depth`` is the index size: clause rankings are taken in full so a BM25-only
        chunk still carries its dense rank and cosine for the clause."""

        for value, name in (
            (dense_depth, "dense_depth"),
            (clause_pool_k, "clause_pool_k"),
            (pair_budget, "pair_budget"),
            (final_top_k, "final_top_k"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(bm25_k, bool) or not isinstance(bm25_k, int) or bm25_k < 0:
            raise ValueError("bm25_k must be a non-negative integer")
        if bm25_k and bm25 is None:
            raise ValueError("bm25_k > 0 needs a BM25 index")
        if clause_pool_k + bm25_k > base.reranker.max_candidates:
            raise ValueError("a clause pool must fit the reranker's max_candidates")
        self.base = base
        self.bm25 = bm25
        self.dense_depth = dense_depth
        self.clause_pool_k = clause_pool_k
        self.bm25_k = bm25_k
        self.pair_budget = pair_budget
        self.final_top_k = final_top_k
        self.trace_recorder = trace_recorder

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RerankedRetrievedChunk]:
        """Clause branches when R2 fires, otherwise exactly ``base.retrieve``."""

        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("query must not be empty or whitespace")
        clauses = split_clauses(query)
        if clauses is None:
            return self.base.retrieve(query, top_k)
        effective_top_k = self.final_top_k if top_k is None else top_k
        if (
            isinstance(effective_top_k, bool)
            or not isinstance(effective_top_k, int)
            or effective_top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")

        recorder = resolve_recorder(self.trace_recorder)
        started = time.perf_counter()
        pools = [self._clause_pool(clause) for clause in clauses]
        original_depth = max(
            0,
            min(self.base.reranker_input_k, self.pair_budget - sum(len(p[0]) for p in pools)),
        )
        safe_record(
            recorder,
            "retrieval.branch.planned",
            clause_count=len(clauses),
            bm25_clause_count=sum(1 for _, bm25_count in pools if bm25_count),
            original_depth=original_depth,
        )
        branches = [
            self._rerank(clause, "clause", pool, bm25_count)
            for clause, (pool, bm25_count) in zip(clauses, pools)
        ]
        if original_depth:
            original = self.base.dense_retriever.retrieve(query, top_k=original_depth)
            branches.append(self._rerank(query, "original", original, 0))

        chosen = round_robin(
            [[item.chunk.chunk_id for item in branch.ranked] for branch in branches],
            effective_top_k,
        )
        # A chunk is reported with the scores of the first branch (in branch order) that ranks it.
        source: dict[str, tuple[int, RerankedRetrievedChunk]] = {}
        for index, branch in enumerate(branches):
            for item in branch.ranked:
                source.setdefault(item.chunk.chunk_id, (index, item))
        results = [
            source[chunk_id][1].model_copy(update={"rank": rank})
            for rank, chunk_id in enumerate(chosen, start=1)
        ]
        pair_count = len(
            {(branch.query, item.chunk.chunk_id) for branch in branches for item in branch.ranked}
        )
        logger.info(
            "retrieval completed",
            extra={
                "retrieval_strategy": self.retrieval_strategy,
                "clause_count": len(clauses),
                "reranker_pair_count": pair_count,
                "reranker_output_count": len(results),
                "latency": time.perf_counter() - started,
            },
        )
        if recorder is not None:
            safe_record(
                recorder,
                "retrieval.branch.completed",
                pair_count=pair_count,
                output_count=len(results),
                branches=[
                    {
                        "branch": index,
                        "kind": branch.kind,
                        "dense_count": branch.dense_count,
                        "bm25_count": branch.bm25_count,
                    }
                    for index, branch in enumerate(branches)
                ],
                results=[
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.chunk_id,
                        "branch": source[item.chunk.chunk_id][0],
                        "candidate_rank": item.candidate_rank,
                        "reranker_score": item.reranker_score,
                    }
                    for item in results
                ],
            )
        return results

    def _clause_pool(self, clause: str) -> tuple[list[RetrievedChunk], int]:
        """Dense top-c of the clause plus, for a clause without CJK, its BM25 top-b not already in
        the pool. Returns the pool and how many chunks BM25 added."""

        ranking = self.base.dense_retriever.retrieve(clause, top_k=self.dense_depth)
        pool = list(ranking[: self.clause_pool_k])
        added = 0
        if self.bm25_k and not has_cjk(clause):
            by_id = {item.chunk.chunk_id: item for item in ranking}
            in_pool = {item.chunk.chunk_id for item in pool}
            for chunk_id in self.bm25.search(clause, self.bm25_k):
                if chunk_id not in in_pool:
                    pool.append(by_id[chunk_id])
                    in_pool.add(chunk_id)
                    added += 1
        return pool, added

    def _rerank(
        self, branch_query: str, kind: str, pool: list[RetrievedChunk], bm25_count: int
    ) -> _Branch:
        """Rank the whole pool by the branch's own query, keeping each chunk's dense evidence."""

        try:
            reranked = self.base.reranker.rerank(branch_query, [item.chunk for item in pool])
        except Exception as exc:
            # As in DenseRerankerRetriever: never fall back to an unreranked order.
            raise RerankerError("learned reranker failed") from exc
        by_id = {item.chunk.chunk_id: item for item in pool}
        ranked = [
            RerankedRetrievedChunk(
                chunk=item.chunk,
                rank=item.rank,
                candidate_rank=by_id[item.chunk.chunk_id].rank,
                similarity_score=by_id[item.chunk.chunk_id].similarity_score,
                reranker_score=item.relevance_score,
            )
            for item in reranked
        ]
        return _Branch(branch_query, kind, len(pool) - bm25_count, bm25_count, ranked)
