"""In-process BM25 over the indexed chunk texts, used only for clause pools in clause-branch retrieval.

This is not the retired full-query BM25 + RRF chain (`evidenceflow.retrieval.bm25`, still banned by
tests/test_architecture_boundaries.py): it never ranks a whole query and never fuses scores; it only
adds up to b chunks to an English clause's dense pool before the reranker.

The spec is fixed by the round-2 pre-registration (work/agent_design/candidate_miss): terms are the
embedding model's SentencePiece pieces, lowercased, with '▁'-only and pure-punctuation pieces
dropped; k1 = 1.2, b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5)); ties break on chunk_id.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def bm25_terms(pieces: Sequence[str]) -> list[str]:
    """Lowercase tokenizer pieces and drop the ones that carry no term.

    A leading '▁' is ignored for the punctuation test but kept in the term, so a word-initial piece
    and the same piece inside a word stay distinct terms.
    """

    terms = []
    for piece in pieces:
        piece = piece.lower()
        core = piece.replace("▁", "")
        if not core or _PUNCT.match(core):
            continue
        terms.append(piece)
    return terms


class SentencePieceTerms:
    """Turn text into BM25 terms with the embedding model's Hugging Face tokenizer."""

    def __init__(self, model_name: str) -> None:
        """Load only the tokenizer files; no model weights."""

        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Fast tokenizers are not safe for concurrent calls (same reason as BgeM3Tokenizer).
        self._lock = threading.Lock()

    def __call__(self, text: str) -> list[str]:
        with self._lock:
            pieces = self._tokenizer.tokenize(text)
        return bm25_terms(pieces)


class BM25Index:
    """Okapi BM25 over a fixed set of documents, ranked by score DESC, chunk_id ASC."""

    def __init__(
        self,
        documents: Mapping[str, str],
        terms: Callable[[str], list[str]],
        *,
        k1: float = 1.2,
        b: float = 0.75,
    ) -> None:
        """Tokenize every document once; the index is read-only afterwards."""

        if not documents:
            raise ValueError("BM25Index needs at least one document")
        self._terms = terms
        self._ids = tuple(sorted(documents))
        self.k1, self.b = k1, b
        self._tf = [Counter(terms(documents[chunk_id])) for chunk_id in self._ids]
        self._lengths = [sum(tf.values()) for tf in self._tf]
        self._average_length = sum(self._lengths) / len(self._lengths)
        document_frequency: Counter[str] = Counter()
        for tf in self._tf:
            document_frequency.update(tf.keys())
        n = len(self._ids)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    @property
    def size(self) -> int:
        return len(self._ids)

    def search(self, query: str, top_k: int) -> list[str]:
        """Return the top_k chunk IDs; a query term counts once per occurrence."""

        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        query_terms = self._terms(query)
        scored = []
        for chunk_id, tf, length in zip(self._ids, self._tf, self._lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self._average_length)
            for term in query_terms:
                frequency = tf.get(term, 0)
                if frequency:
                    score += self._idf[term] * frequency * (self.k1 + 1) / (frequency + norm)
            scored.append((score, chunk_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [chunk_id for _, chunk_id in scored[:top_k]]
