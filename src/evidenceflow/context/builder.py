"""Assemble the formal retrieval results into a budget-constrained, complete LLM Context.

Two entry points share one renderer:

- :meth:`ContextBuilder.build` is the historical path over production retrieval results; it keeps
  ordering strictly by the retriever's rank and is used by ``ask``, the fixed agent and evaluation.
- :meth:`ContextBuilder.build_evidence` accepts explicit :class:`~evidenceflow.domain.context.ContextEvidence`
  items, so an adaptive run can include opened neighbourhood chunks without inventing a retrieval
  rank or a similarity score for them (PROJECT_SPEC §10.3).
"""

from __future__ import annotations

from collections.abc import Sequence

from evidenceflow.domain.answer import SourceReference
from evidenceflow.domain.chunk import (
    DocumentChunk,
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.context import BuiltContext, ContextEntry, ContextEvidence
from evidenceflow.ingestion.tokenizer import TokenizerError, TokenizerProtocol


class ContextBuilder:
    """Include complete Chunks in the given order, with no truncation and no re-ranking."""

    _ENTRY_SEPARATOR = "\n\n"

    def __init__(self, tokenizer: TokenizerProtocol, max_context_tokens: int) -> None:
        """Inject the Tokenizer that matches Embedding plus a positive-integer default budget."""

        self._tokenizer = tokenizer
        self._max_context_tokens = self._validate_budget(max_context_tokens)

    @staticmethod
    def _validate_budget(max_tokens: int) -> int:
        """Reject booleans, zero, and negatives to avoid ambiguous budget semantics."""

        if type(max_tokens) is not int or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        return max_tokens

    @property
    def max_context_tokens(self) -> int:
        """公开实际证据预算，供 Agent 判断是否需要寻找更紧凑的完整支持。"""
        return self._max_context_tokens

    def _count_tokens(self, text: str) -> int:
        """Count with the formal Tokenizer and verify that test doubles also honour the interface contract."""

        token_ids = self._tokenizer.encode(text)
        if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
            raise TokenizerError("Tokenizer encode returned invalid token IDs")
        return len(token_ids)

    @staticmethod
    def _render_entry(source_label: str, chunk: DocumentChunk) -> str:
        """Render the label, source header, and unmodified full body using a stable template."""

        lines = [source_label, f"File: {chunk.file_name}"]
        if chunk.page_start is not None:
            # The DocumentChunk model already guarantees a complete page range; a single page
            # avoids the redundant 12-12.
            page_range = (
                str(chunk.page_start)
                if chunk.page_start == chunk.page_end
                else f"{chunk.page_start}-{chunk.page_end}"
            )
            lines.append(f"Pages: {page_range}")
        lines.extend(["Content:", chunk.content])
        return "\n".join(lines)

    @staticmethod
    def _source_reference(source_label: str, chunk: DocumentChunk) -> SourceReference:
        """Map the existing SourceReference from the complete Chunk without carrying retrieval-internal scores."""

        return SourceReference(
            source_label=source_label,
            document_id=chunk.document_id,
            file_name=chunk.file_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            chunk_id=chunk.chunk_id,
        )

    @staticmethod
    def _ranked_candidates(
        retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
    ) -> list[RetrievedChunk | RerankedRetrievedChunk]:
        """Restore the formal order from the explicit Rank, rejecting ambiguous duplicate identities or ranks."""

        candidates = list(retrieved_chunks)
        ranks = [item.rank for item in candidates]
        chunk_ids = [item.chunk.chunk_id for item in candidates]
        if len(set(ranks)) != len(ranks):
            raise ValueError("retrieved_chunks must not contain duplicate ranks")
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("retrieved_chunks must not contain duplicate chunk_id values")
        # Order by the Retriever's formal Rank only; introduce no second relevance judgment.
        return sorted(candidates, key=lambda item: item.rank)

    def build(
        self,
        retrieved_chunks: Sequence[RetrievedChunk | RerankedRetrievedChunk],
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Fit as many complete retrieval candidates as the budget allows, in rank order."""

        candidates = self._ranked_candidates(retrieved_chunks)
        evidence = [
            ContextEvidence(
                chunk=item.chunk,
                origin="retrieval",
                retrieval_rank=item.rank,
            )
            for item in candidates
        ]
        return self.build_evidence(evidence, max_tokens=max_tokens)

    def build_evidence(
        self,
        evidence: Sequence[ContextEvidence],
        max_tokens: int | None = None,
    ) -> BuiltContext:
        """Fit complete evidence items in the caller's order, preserving each item's real origin.

        A retrieval item keeps its real rank and (in an adaptive run) its search index; an opened
        neighbourhood item keeps ``retrieval_rank = None`` so nothing downstream can mistake an
        opened neighbour for a scored retrieval hit. Duplicate ``chunk_id`` values are rejected
        instead of silently collapsing two evidence origins into one entry.
        """

        budget = self._max_context_tokens if max_tokens is None else self._validate_budget(max_tokens)
        candidates = list(evidence)
        for item in candidates:
            if not isinstance(item, ContextEvidence):
                raise TypeError(
                    f"build_evidence expects ContextEvidence, not {type(item).__name__}"
                )
        chunk_ids = [item.chunk.chunk_id for item in candidates]
        if len(set(chunk_ids)) != len(chunk_ids):
            raise ValueError("evidence must not contain duplicate chunk_id values")
        entries: list[ContextEntry] = []
        rendered_entries: list[str] = []
        total_tokens = 0

        for item in candidates:
            # The label follows the number of actually included entries, so skipped candidates
            # leave no gaps.
            source_label = f"[S{len(entries) + 1}]"
            rendered_entry = self._render_entry(source_label, item.chunk)
            prospective_text = self._ENTRY_SEPARATOR.join([*rendered_entries, rendered_entry])
            # Re-encode the whole prospective Context so separators between entries and token
            # boundaries also count against the budget.
            prospective_tokens = self._count_tokens(prospective_text)
            if prospective_tokens > budget:
                continue

            entries.append(
                ContextEntry(
                    source_label=source_label,
                    retrieval_rank=item.retrieval_rank,
                    chunk_id=item.chunk.chunk_id,
                    content=item.chunk.content,
                    source_reference=self._source_reference(source_label, item.chunk),
                    token_count=self._count_tokens(rendered_entry),
                    origin=item.origin,
                    search_index=item.search_index,
                )
            )
            rendered_entries.append(rendered_entry)
            total_tokens = prospective_tokens

        context_text = self._ENTRY_SEPARATOR.join(rendered_entries)
        return BuiltContext(
            context_text=context_text,
            entries=entries,
            total_tokens=total_tokens,
            candidate_count=len(candidates),
            included_count=len(entries),
            skipped_count=len(candidates) - len(entries),
        )
