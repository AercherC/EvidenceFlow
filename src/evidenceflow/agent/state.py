"""Typed, frozen state that flows through the agent's fixed path (plan -> retrieve -> synthesize -> terminate).

Each node returns ``state.model_copy(update=...)``; no node mutates the state it receives. The state
lives for one question only and is discarded after the run; nothing here is persisted.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.chunk import RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.generation import GeneratedAnswer
from evidenceflow.agent.verify import Verification

TerminationReason = Literal["answered", "insufficient_context", "budget_exhausted", "fallback"]
Support = Literal["full", "partial", "none"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Clause(_Frozen):
    text: str = Field(min_length=1)
    lang: Literal["en", "zh"]


class Plan(_Frozen):
    """``single``: the split rule did not fire; ``multi``: one clause per sub-question (2-3)."""

    mode: Literal["single", "multi"]
    clauses: tuple[Clause, ...]


class Coverage(_Frozen):
    """Papers the question names (short names) and those with no selected chunk."""

    targets: tuple[str, ...]
    uncovered: tuple[str, ...]


class Budget(_Frozen):
    """Per-answer limits and what has been used so far. Reranker pairs are charged at the retriever's
    configured cap for the round (an upper bound), since the retriever does not report them."""

    llm_calls: int = 0
    llm_calls_max: int = Field(default=2, ge=1)
    rerank_pairs: int = 0
    rerank_pairs_max: int = Field(default=150, gt=0)
    rounds: int = 0
    rounds_max: int = Field(default=2, ge=1)
    deadline_monotonic: float


class Terminal(_Frozen):
    reason: TerminationReason
    support: Support


class AgentState(_Frozen):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    query: str = Field(min_length=1)
    plan: Plan | None = None
    selected: tuple[RetrievedChunk | RerankedRetrievedChunk, ...] = ()
    seen_signatures: frozenset[str] = frozenset()
    coverage: Coverage | None = None
    widened: bool = False
    context: BuiltContext | None = None
    generated: GeneratedAnswer | None = None
    answer: RAGAnswer | None = None
    verification: Verification | None = None
    budget: Budget
    terminal: Terminal | None = None
