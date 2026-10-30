"""Named-paper coverage and the one deterministic widen step (no LLM, no score threshold).

Coverage: a paper the question names by its short name (from the corpus manifest titles, e.g.
"PoinTr: Diverse Point Cloud ..." -> PoinTr) is covered when a selected chunk comes from it.
Clause coverage alone cannot fail under clause-branch retrieval (every branch places its top chunk),
so named papers are the only coverage signal.

Widen: for each uncovered paper (at most ``max_papers``), take that paper's top ``paper_depth``
chunks in the dense ranking of the clause that names it, rerank them with that clause, and append
the best one not already selected. Nothing selected before is removed or reordered.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from evidenceflow.domain.chunk import RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.retrieval.reranker import LearnedReranker, RerankerError
from evidenceflow.retrieval.retriever import DenseRetriever

# A short name is the title prefix before ":" when that prefix is one token (PoinTr, Hyper-PCN).
_SHORT_NAME = re.compile(r"^([A-Za-z0-9][A-Za-z0-9-]{1,30}):")


def load_paper_names(manifest_path: Path | None) -> dict[str, str]:
    """{short name: file_name} from a corpus manifest; empty when the file is absent."""

    if manifest_path is None or not Path(manifest_path).is_file():
        return {}
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    names = {}
    for paper in manifest.get("papers", []):
        match = _SHORT_NAME.match(paper.get("title", ""))
        if match and paper.get("file_name"):
            names[match.group(1)] = paper["file_name"]
    return names


def _name_pattern(name: str) -> re.Pattern[str]:
    # Case-insensitive, hyphen optional (PF-Net / PFNet), and not inside a longer name
    # (PoinTr in AdaPoinTr, PCN in Hyper-PCN).
    body = "-?".join(re.escape(part) for part in name.split("-"))
    return re.compile(rf"(?<![A-Za-z0-9-]){body}(?![A-Za-z0-9])", re.IGNORECASE)


def named_papers(text: str, papers: Mapping[str, str]) -> list[str]:
    """Short names of the papers ``text`` names, in the order the papers are listed."""

    return [name for name in papers if _name_pattern(name).search(text)]


@dataclass(frozen=True)
class WidenResult:
    added: tuple[RerankedRetrievedChunk, ...]
    pair_count: int


class Widener:
    """Paper-scoped deeper retrieval for uncovered named papers."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        reranker: LearnedReranker,
        papers: Mapping[str, str],
        *,
        dense_depth: int,
        paper_depth: int = 25,
        max_papers: int = 2,
    ) -> None:
        self.dense_retriever = dense_retriever
        self.reranker = reranker
        self.papers = dict(papers)
        self.dense_depth = dense_depth
        self.paper_depth = paper_depth
        self.max_papers = max_papers

    def max_pairs(self, paper_count: int) -> int:
        return min(paper_count, self.max_papers) * self.paper_depth

    def widen(
        self,
        targets: Sequence[tuple[str, str]],
        selected: Sequence[RetrievedChunk | RerankedRetrievedChunk],
    ) -> WidenResult:
        """``targets`` are (short name, branch query) pairs; returns the chunks to append."""

        chosen = {item.chunk.chunk_id for item in selected}
        added: list[RerankedRetrievedChunk] = []
        pairs = 0
        for name, branch_query in list(targets)[: self.max_papers]:
            file_name = self.papers[name]
            ranking = self.dense_retriever.retrieve(branch_query, top_k=self.dense_depth)
            pool = [
                item for item in ranking if item.chunk.file_name == file_name and item.chunk.chunk_id not in chosen
            ][: self.paper_depth]
            if not pool:
                continue
            pairs += len(pool)
            try:
                reranked = self.reranker.rerank(branch_query, [item.chunk for item in pool], top_k=1)
            except Exception as exc:
                raise RerankerError("learned reranker failed") from exc
            best = reranked[0]
            source = next(item for item in pool if item.chunk.chunk_id == best.chunk.chunk_id)
            added.append(
                RerankedRetrievedChunk(
                    chunk=best.chunk,
                    rank=len(selected) + len(added) + 1,
                    candidate_rank=source.rank,
                    similarity_score=source.similarity_score,
                    reranker_score=best.relevance_score,
                )
            )
            chosen.add(best.chunk.chunk_id)
        return WidenResult(added=tuple(added), pair_count=pairs)
