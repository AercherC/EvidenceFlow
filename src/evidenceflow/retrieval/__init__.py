"""Public interfaces for formal Dense Retrieval and the Learned Reranker."""

from evidenceflow.retrieval.reranker import (
    DenseRerankerRetriever,
    LearnedReranker,
    RerankerError,
    RerankedChunk,
    TransformersCrossEncoderScorer,
    rank_scored_candidates,
)
from evidenceflow.retrieval.retriever import DenseRetriever, InvalidQueryError
from evidenceflow.retrieval.exact_dense import ExactDenseIndex, ExactDenseIndexError
from evidenceflow.retrieval.clause_bm25 import BM25Index, SentencePieceTerms
from evidenceflow.retrieval.clause_branch import ClauseBranchRetriever, split_clauses
from evidenceflow.retrieval.evidence_navigation import (
    EvidenceNavigationError,
    EvidenceNavigator,
    UnknownChunkError,
)

__all__ = [
    "BM25Index",
    "ClauseBranchRetriever",
    "DenseRetriever",
    "EvidenceNavigationError",
    "EvidenceNavigator",
    "ExactDenseIndex",
    "ExactDenseIndexError",
    "DenseRerankerRetriever",
    "InvalidQueryError",
    "LearnedReranker",
    "RerankerError",
    "RerankedChunk",
    "SentencePieceTerms",
    "TransformersCrossEncoderScorer",
    "UnknownChunkError",
    "rank_scored_candidates",
    "split_clauses",
]
