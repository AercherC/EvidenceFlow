"""Read-only evidence tools bound to one active-index snapshot for a whole run.

``search`` delegates to the existing production retriever; ``open`` reads a neighbourhood through
the Step-1 :class:`~evidenceflow.retrieval.evidence_navigation.EvidenceNavigator`. Both tools are
constructed by the composition root from a *single* collection read, so one run can never mix
results from two indexes (for example, an index that was activated between search and open).

Provenance is enforced at this boundary rather than assumed:

- the navigator must be a real ``EvidenceNavigator`` (a raw chunk list, a dict, or any other
  external snapshot object is rejected), because only the navigator validates duplicate identities
  and its declared chunk count;
- every chunk a retriever returns is looked up again in that same snapshot, and its
  ``document_id`` / ``chunk_index`` / ``content`` / ``content_hash`` must match, so a retriever that
  answers from another index fails loudly instead of being reported as evidence.

The tools themselves write nothing, resolve no active pointer and make no decision
about which action to take next. Search still invokes the existing retrieval models.
CLI 和 WebUI 的证据 Agent 使用这两个工具收集可追溯的原始文档证据。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from evidenceflow.domain.chunk import DocumentChunk, RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.domain.evidence import EvidenceIndexIdentity, EvidenceNeighborhood, IndexedDocument
from evidenceflow.retrieval.clause_branch import ClauseBranchRetriever
from evidenceflow.retrieval.evidence_navigation import (
    EvidenceNavigator,
    UnknownChunkError,
)


class EvidenceProvenanceError(RuntimeError):
    """A retrieval result cannot be re-identified in the snapshot the tools were built from."""


class EvidenceRetrieverProtocol(Protocol):
    """The minimal read-only retrieval interface the tools need."""

    def retrieve(self, query: str, *, document_id: str | None = None) -> list[RetrievedChunk | RerankedRetrievedChunk]: ...


class EvidenceTools:
    """``search`` and ``open`` over the one index snapshot both were assembled from."""

    def __init__(
        self,
        retriever: EvidenceRetrieverProtocol,
        navigator: EvidenceNavigator,
    ) -> None:
        """Bind an already-built retriever to the navigator of the same snapshot.

        The constructor takes no raw chunk sequence. Production callers must use
        ``runtime.build_evidence_tools`` to construct both dependencies from one read;
        the navigator validates duplicate identities and its declared chunk count.
        """

        if not isinstance(navigator, EvidenceNavigator):
            raise TypeError(
                "navigator must be an EvidenceNavigator built from one index snapshot, "
                f"not {type(navigator).__name__}"
            )
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must expose retrieve(query)")
        self._retriever = retriever
        self._navigator = navigator

    @property
    def identity(self) -> EvidenceIndexIdentity:
        """Return the identity shared by every result these tools can produce."""

        return self._navigator.identity

    @property
    def chunk_count(self) -> int:
        """Return the number of chunks in the bound snapshot."""

        return self._navigator.chunk_count

    def search(
        self, query: str, *, retrieve: Callable[[str], list[RetrievedChunk | RerankedRetrievedChunk]] | None = None,
    ) -> list[RetrievedChunk | RerankedRetrievedChunk]:
        """Run the production retriever and prove every hit belongs to the bound snapshot."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        # 实验补证可换候选策略，但证据仍由同一个快照验证。
        results = (retrieve or self._retriever.retrieve)(query)
        for item in results:
            self._require_same_snapshot(item.chunk)
        return results

    def open(
        self,
        chunk_id: str,
        *,
        before: int = 1,
        after: int = 1,
    ) -> EvidenceNeighborhood:
        """Return the anchor and its exact adjacent chunks from the bound snapshot.

        A ``chunk_id`` outside the snapshot raises ``UnknownChunkError``; there is no fallback to
        another document, another index, or a freshly opened collection.
        """

        return self._navigator.neighborhood(chunk_id, before=before, after=after)

    def list_documents(self) -> list[IndexedDocument]:
        """列出问答所用快照的文档，供 WebUI 浏览。"""
        return self._navigator.list_documents()

    def document_chunk(self, document_id: str, offset: int) -> DocumentChunk:
        """从问答所用快照按文档顺序读取一个索引片段。"""
        return self._navigator.document_chunk(document_id, offset)

    def search_with_branches(
        self, query: str,
    ) -> tuple[list[RetrievedChunk | RerankedRetrievedChunk], tuple[tuple[str, ...], ...], tuple[str, ...]]:
        """返回同次检索的命中、子句排名和对应论文身份。"""
        if not isinstance(self._retriever, ClauseBranchRetriever):
            return self.search(query), (), ()
        result = self._retriever.retrieve_with_branches(query)
        for item in result.hits:
            self._require_same_snapshot(item.chunk)
        return list(result.hits), result.clause_hits, result.document_scopes

    def search_in_document(self, query: str, document_id: str) -> list[RetrievedChunk | RerankedRetrievedChunk]:
        """按明确的文档身份召回，所有结果都从当前版本重新核对。"""
        retriever = self._retriever
        if isinstance(retriever, ClauseBranchRetriever):
            hits = self.search(query, retrieve=lambda text: list(retriever.retrieve_with_branches(
                text, document_scope=document_id,
            ).hits))
        else:
            hits = self.search(query, retrieve=lambda text: retriever.retrieve(text, document_id=document_id))
        if any(hit.chunk.document_id != document_id for hit in hits):
            raise EvidenceProvenanceError("scoped search returned a chunk from another document")
        return hits

    def previous_page_continuation(self, chunk_id: str) -> DocumentChunk | None:
        """只从工具绑定的版本中读取可能续接的前页原文。"""
        return self._navigator.previous_page_continuation(chunk_id)

    def _require_same_snapshot(self, chunk: DocumentChunk) -> None:
        """Re-identify one returned chunk against the snapshot, or refuse the result.

        ``chunk_id`` alone is not provenance: an index rebuilt after an ingest can reuse an ID whose
        document, position or text has moved. The identity fields are therefore compared as well.
        """

        if not isinstance(chunk, DocumentChunk):
            raise EvidenceProvenanceError(
                f"retriever returned {type(chunk).__name__}, expected DocumentChunk"
            )
        try:
            known = self._navigator.chunk(chunk.chunk_id)
        except UnknownChunkError as exc:
            raise EvidenceProvenanceError(
                f"retriever returned chunk_id {chunk.chunk_id!r} that is not part of index "
                f"{self.identity.collection_name!r}"
            ) from exc
        observed = (chunk.document_id, chunk.chunk_index, chunk.content, chunk.content_hash)
        expected = (known.document_id, known.chunk_index, known.content, known.content_hash)
        if observed != expected:
            raise EvidenceProvenanceError(
                f"retriever result for chunk_id {chunk.chunk_id!r} does not match index "
                f"{self.identity.collection_name!r}; it did not come from this snapshot"
            )
