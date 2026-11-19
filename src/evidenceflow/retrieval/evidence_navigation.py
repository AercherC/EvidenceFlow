"""Read-only evidence navigation over one already-loaded index snapshot.

Given a ``chunk_id`` that exists in the snapshot, the navigator returns that Chunk together with a
small number of Chunks at adjacent ``chunk_index`` values of the *same* ``document_id``. It reads
only the in-memory snapshot it was built with: it opens no database, loads no model, calls no
network and makes no decision for an Agent. Every result carries the index identity it came from,
and a missing ``chunk_id`` fails loudly instead of falling back to another document or index.

``runtime.build_evidence_navigator`` exposes it directly, and ``evidenceflow.agent.tools`` binds it
with the existing retriever over the same snapshot for the production evidence Agent.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.evidence import EvidenceIndexIdentity, EvidenceNeighborhood, IndexedDocument

# A neighbourhood is deliberately small: the anchor plus at most three neighbours on each side.
MAX_NEIGHBOR_RADIUS = 3
_CAPTION_START = re.compile(r"^(?:fig\.?|figure|table)\s+(?:\d+|[IVX]+)\b", re.IGNORECASE)


class EvidenceNavigationError(RuntimeError):
    """The snapshot cannot be navigated (duplicate identity, wrong type, or size mismatch)."""


class UnknownChunkError(EvidenceNavigationError):
    """The requested ``chunk_id`` is not part of this index snapshot."""


def _validate_radius(value: int, *, name: str) -> int:
    """Accept only 0..MAX_NEIGHBOR_RADIUS, rejecting booleans that pose as integers."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_NEIGHBOR_RADIUS
    ):
        raise ValueError(f"{name} must be an integer between 0 and {MAX_NEIGHBOR_RADIUS}")
    return value


class EvidenceNavigator:
    """Answer "what is next to this Chunk?" from one frozen, read-only index snapshot."""

    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
        identity: EvidenceIndexIdentity,
    ) -> None:
        """Index the snapshot by chunk ID and by (document_id, chunk_index).

        The snapshot must already describe one index: duplicate chunk IDs, or two Chunks claiming
        the same position inside one document, make navigation ambiguous and are rejected here.
        """

        chunk_list = list(chunks)
        by_id: dict[str, DocumentChunk] = {}
        by_document: dict[str, dict[int, DocumentChunk]] = {}
        for chunk in chunk_list:
            if not isinstance(chunk, DocumentChunk):
                raise EvidenceNavigationError(
                    f"snapshot contains {type(chunk).__name__}, expected DocumentChunk"
                )
            if chunk.chunk_id in by_id:
                raise EvidenceNavigationError(f"duplicate chunk_id in snapshot: {chunk.chunk_id}")
            positions = by_document.setdefault(chunk.document_id, {})
            if chunk.chunk_index in positions:
                raise EvidenceNavigationError(
                    "duplicate chunk_index in snapshot: "
                    f"{chunk.document_id}#{chunk.chunk_index}"
                )
            by_id[chunk.chunk_id] = chunk
            positions[chunk.chunk_index] = chunk
        if len(chunk_list) != identity.chunk_count:
            raise EvidenceNavigationError(
                f"snapshot has {len(chunk_list)} chunks but identity declares "
                f"{identity.chunk_count}"
            )
        self._by_id = by_id
        self._by_document = by_document
        self._identity = identity

    @property
    def identity(self) -> EvidenceIndexIdentity:
        """Return the immutable identity of the snapshot this navigator reads."""

        return self._identity

    @property
    def chunk_count(self) -> int:
        """Return the number of Chunks in the snapshot."""

        return len(self._by_id)

    def chunk(self, chunk_id: str) -> DocumentChunk:
        """Return one Chunk of this snapshot by identity, without navigation.

        This is the identity probe used by callers that need to prove a retrieval result really came
        from this snapshot: a chunk_id absent here raises ``UnknownChunkError`` rather than being
        looked up in another index. The returned Chunk is a deep copy, so a caller cannot mutate the
        snapshot through it.
        """

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        chunk = self._by_id.get(chunk_id)
        if chunk is None:
            raise UnknownChunkError(
                f"chunk_id {chunk_id!r} is not in index {self._identity.collection_name!r}"
            )
        return chunk.model_copy(deep=True)

    def list_documents(self) -> list[IndexedDocument]:
        """按文件名列出当前快照的逻辑文档，保留同名来源的独立身份。"""

        documents = []
        for document_id, positions in self._by_document.items():
            chunks = positions.values()
            first = positions[min(positions)]
            documents.append(IndexedDocument(
                document_id=document_id,
                file_name=first.file_name,
                chunk_count=len(positions),
                page_start=min((chunk.page_start for chunk in chunks if chunk.page_start is not None), default=None),
                page_end=max((chunk.page_end for chunk in chunks if chunk.page_end is not None), default=None),
            ))
        return sorted(documents, key=lambda doc: (doc.file_name.casefold(), doc.document_id))

    def document_chunk(self, document_id: str, offset: int) -> DocumentChunk:
        """按文档内已索引片段的 0-based 顺序读取；跳过旧索引中的位置缺口。"""

        positions = self._by_document.get(document_id)
        if positions is None or not 0 <= offset < len(positions):
            raise UnknownChunkError("document passage is not in the index snapshot")
        return self.chunk(positions[sorted(positions)[offset]].chunk_id)

    def neighborhood(
        self,
        chunk_id: str,
        *,
        before: int = 1,
        after: int = 1,
    ) -> EvidenceNeighborhood:
        """Return the anchor and up to ``before`` / ``after`` adjacent Chunks of its own document.

        Adjacency is the exact ``chunk_index`` sequence: a missing index stops the walk instead of
        jumping over the gap, and the walk never leaves the anchor's ``document_id``. A ``chunk_id``
        absent from this snapshot raises ``UnknownChunkError``; there is no cross-document or
        cross-index fallback.
        """

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError("chunk_id must not be empty")
        before = _validate_radius(before, name="before")
        after = _validate_radius(after, name="after")
        anchor = self._by_id.get(chunk_id)
        if anchor is None:
            raise UnknownChunkError(
                f"chunk_id {chunk_id!r} is not in index {self._identity.collection_name!r}"
            )
        return self._neighbourhood(anchor, before=before, after=after)

    def previous_page_continuation(self, chunk_id: str) -> DocumentChunk | None:
        """为以英文小写词起头的正文查找前页同章节，最多越过一条插入图表题注。"""
        near = self.neighborhood(chunk_id, before=2, after=0)
        anchor = near.anchor
        heading = anchor.metadata.get("heading_path")
        page = anchor.page_start
        text = anchor.content.lstrip()
        if not heading or page is None or page <= 1 or not re.match(r"[a-z]{2,}\b", text) or _CAPTION_START.match(text):
            return None
        for previous in reversed(near.previous):
            if previous.metadata.get("heading_path") != heading:
                return None
            caption = _CAPTION_START.match(previous.content.lstrip())
            if caption and previous.page_start == page:
                continue
            if not caption and previous.page_start == page - 1 and previous.page_end in (page - 1, page):
                return previous
            return None
        return None

    def _neighbourhood(
        self,
        anchor: DocumentChunk,
        *,
        before: int,
        after: int,
    ) -> EvidenceNeighborhood:
        """Walk exact adjacent positions inside the anchor's document and compute the real page span."""

        # Neighbours are looked up only in the anchor's own document table, so another document that
        # happens to use the same chunk_index values can never appear here.
        positions = self._by_document[anchor.document_id]
        previous: list[DocumentChunk] = []
        for offset in range(1, before + 1):
            neighbor = positions.get(anchor.chunk_index - offset)
            if neighbor is None:
                break
            previous.append(neighbor)
        # Emit in document order: farthest first, nearest last.
        previous.reverse()

        following: list[DocumentChunk] = []
        for offset in range(1, after + 1):
            neighbor = positions.get(anchor.chunk_index + offset)
            if neighbor is None:
                break
            following.append(neighbor)

        page_starts = [
            chunk.page_start
            for chunk in (*previous, anchor, *following)
            if chunk.page_start is not None
        ]
        page_ends = [
            chunk.page_end
            for chunk in (*previous, anchor, *following)
            if chunk.page_end is not None
        ]
        return EvidenceNeighborhood(
            anchor=anchor.model_copy(deep=True),
            previous=tuple(chunk.model_copy(deep=True) for chunk in previous),
            following=tuple(chunk.model_copy(deep=True) for chunk in following),
            identity=self._identity,
            page_start=min(page_starts) if page_starts else None,
            page_end=max(page_ends) if page_ends else None,
        )
