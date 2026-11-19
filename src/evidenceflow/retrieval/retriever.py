"""Implement Dense Retrieval by composing an EmbeddingProvider with Dense Search."""

from typing import Protocol

from evidenceflow.domain.chunk import RetrievedChunk
from evidenceflow.embedding.base import EmbeddingProvider


class DenseSearch(Protocol):
    """Define the minimal candidate-search interface DenseRetriever needs."""

    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        *,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """Return Dense candidates in the project's stable ordering."""

        ...


class InvalidQueryError(ValueError):
    """Signals that the user Query is empty, whitespace-only, or not a string."""


def _validate_top_k(top_k: int) -> None:
    """Guarantees Dense Top-K is a positive integer, avoiding pointless underlying queries."""

    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")


class DenseRetriever:
    """Encode the raw Query into a vector and delegate to VectorStore for the Dense Top-K."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: DenseSearch,
        top_k: int = 5,
    ) -> None:
        """Inject an existing Provider and Store and keep the validated default Top-K."""

        _validate_top_k(top_k)
        self.embedding_provider = embedding_provider
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        document_id: str | None = None,
    ) -> list[RetrievedChunk]:
        """生成查询向量，按指定文档或全库召回 Dense 候选。"""

        if not isinstance(query, str) or not query.strip():
            # strip() is only used to detect whitespace; the processed text is not passed to the
            # Embedding Provider.
            raise InvalidQueryError("query must not be empty or whitespace")
        effective_top_k = self.top_k if top_k is None else top_k
        _validate_top_k(effective_top_k)

        # The Retriever does not load an index, compute cosine, or rerank; it only orchestrates the
        # injected search interface.
        query_embedding = self.embedding_provider.embed_query(query)
        if document_id is not None:
            return self.vector_store.search(query_embedding, effective_top_k, document_id=document_id)
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=effective_top_k,
        )
