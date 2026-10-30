"""Budget, deadline and loop-guard checks; each raises before the step that would break a limit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

from evidenceflow.agent.state import AgentState


class AgentPolicyError(RuntimeError):
    """A limit would be broken; the service stops the agent path."""


class BudgetExceeded(AgentPolicyError):
    """An LLM-call, reranker-pair or round limit would be exceeded."""


class DeadlineExceeded(AgentPolicyError):
    """The wall-clock deadline has passed."""


class LoopDetected(AgentPolicyError):
    """The same retrieval step was about to run twice."""


def check_deadline(state: AgentState, clock: Callable[[], float]) -> None:
    if clock() > state.budget.deadline_monotonic:
        raise DeadlineExceeded("agent deadline exceeded")


def charge_round(state: AgentState, pairs: int) -> AgentState:
    """Charge one retrieval round of at most ``pairs`` reranker pairs."""

    budget = state.budget
    if budget.rounds + 1 > budget.rounds_max:
        raise BudgetExceeded("retrieval round limit reached")
    if budget.rerank_pairs + pairs > budget.rerank_pairs_max:
        raise BudgetExceeded("reranker pair limit reached")
    return state.model_copy(
        update={"budget": budget.model_copy(update={"rounds": budget.rounds + 1, "rerank_pairs": budget.rerank_pairs + pairs})}
    )


def require_llm_call(state: AgentState) -> None:
    """At least one LLM call must remain before generation starts."""

    if state.budget.llm_calls + 1 > state.budget.llm_calls_max:
        raise BudgetExceeded("LLM call limit reached")


def charge_llm(state: AgentState, calls: int) -> AgentState:
    """Record the calls the generator actually made (1, or 2 with its format repair)."""

    budget = state.budget
    return state.model_copy(update={"budget": budget.model_copy(update={"llm_calls": budget.llm_calls + calls})})


def guard_loop(state: AgentState, step: str, *parts: object) -> AgentState:
    """Second line of defence: the path is acyclic, but a repeated step signature still stops the run."""

    signature = hashlib.sha256(json.dumps([step, *parts], ensure_ascii=False, default=str).encode("utf-8")).hexdigest()
    if signature in state.seen_signatures:
        raise LoopDetected(f"step {step!r} repeated")
    return state.model_copy(update={"seen_signatures": state.seen_signatures | {signature}})
