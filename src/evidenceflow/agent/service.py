"""AgentService: runs the fixed agent path and falls back to the plain RAGService on agent errors."""

from __future__ import annotations

import time
from collections.abc import Callable


from evidenceflow.agent.result import AgentAnswer
from evidenceflow.agent.nodes import (
    AgentDeps,
    AgentRetrieverProtocol,
    coverage_node,
    plan_node,
    retrieve_node,
    synthesize_node,
    terminate_node,
    verify_node,
    widen_node,
)
from evidenceflow.agent.coverage import Widener
from evidenceflow.agent.state import AgentState, Budget, Support, Terminal, TerminationReason
from evidenceflow.agent.verify import Verification
from evidenceflow.citation import CitationValidationError
from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.generation import GenerationContractError
from evidenceflow.llm import LLMError
from evidenceflow.rag.service import ContextBuilderProtocol, GeneratorProtocol
from evidenceflow.tracing import TraceRecorder, resolve_recorder, safe_record

# Errors of the generation contract itself. They propagate unchanged, exactly as from RAGService:
# the fallback would repeat the same LLM call and fail the same way.
PROPAGATED_ERRORS = (LLMError, GenerationContractError, CitationValidationError)


class AgentService:
    """plan -> retrieve -> synthesize -> terminate, under per-answer budgets and a deadline."""

    def __init__(
        self,
        retriever: AgentRetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        generator: GeneratorProtocol,
        *,
        fallback: Callable[[str], RAGAnswer] | None,
        single_round_pairs: int,
        multi_round_pairs: int,
        llm_calls_max: int = 2,
        rerank_pairs_max: int = 150,
        rounds_max: int = 2,
        deadline_seconds: float = 20.0,
        clock: Callable[[], float] = time.monotonic,
        widener: Widener | None = None,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """``fallback`` is normally ``RAGService.answer`` over the same components; None disables it."""

        self.retriever = retriever
        self.context_builder = context_builder
        self.generator = generator
        self.fallback = fallback
        self.single_round_pairs = single_round_pairs
        self.multi_round_pairs = multi_round_pairs
        self.llm_calls_max = llm_calls_max
        self.rerank_pairs_max = rerank_pairs_max
        self.rounds_max = rounds_max
        self.deadline_seconds = deadline_seconds
        self.clock = clock
        self.widener = widener
        self.trace_recorder = trace_recorder

    def answer(self, query: str) -> RAGAnswer:
        """Drop-in for ``RAGService.answer``."""

        return self.run(query).answer

    def run(self, query: str) -> AgentAnswer:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        recorder = resolve_recorder(self.trace_recorder)
        safe_record(recorder, "run.started")
        deps = AgentDeps(
            retriever=self.retriever,
            context_builder=self.context_builder,
            generator=self.generator,
            single_round_pairs=self.single_round_pairs,
            multi_round_pairs=self.multi_round_pairs,
            clock=self.clock,
            recorder=recorder,
            widener=self.widener,
        )
        state = AgentState(
            query=query,
            budget=Budget(
                llm_calls_max=self.llm_calls_max,
                rerank_pairs_max=self.rerank_pairs_max,
                rounds_max=self.rounds_max,
                deadline_monotonic=self.clock() + self.deadline_seconds,
            ),
        )
        stage = "plan"
        fallback_error = None
        try:
            for stage, node in (
                ("plan", plan_node),
                ("retrieval", retrieve_node),
                ("coverage", coverage_node),
                ("widen", widen_node),
                ("synthesis", synthesize_node),
                ("verify", verify_node),
            ):
                state = node(state, deps)
            stage = "terminate"
            state = terminate_node(state, deps)
        except PROPAGATED_ERRORS as exc:
            safe_record(recorder, "run.failed", error_type=type(exc).__name__, stage=stage)
            raise
        except Exception as exc:
            if self.fallback is None:
                safe_record(recorder, "run.failed", error_type=type(exc).__name__, stage=stage)
                raise
            # Any agent-side failure (policy limit, deadline, a bug in a node) still returns the
            # answer the plain pipeline would have given.
            fallback_error = type(exc).__name__
            safe_record(recorder, "agent.fallback", error_type=fallback_error, stage=stage)
            try:
                answer = self.fallback(query)
            except Exception as fallback_exc:
                safe_record(recorder, "run.failed", error_type=type(fallback_exc).__name__, stage="fallback")
                raise
            support: Support = "full" if answer.status == AnswerStatus.ANSWERED else "none"
            state = state.model_copy(update={"answer": answer, "terminal": Terminal(reason="fallback", support=support)})
        result = AgentAnswer(
            answer=state.answer,
            mode=state.plan.mode if state.plan else None,
            clauses=tuple(c.text for c in state.plan.clauses) if state.plan else (),
            support=state.terminal.support,
            termination_reason=state.terminal.reason,
            llm_calls=state.budget.llm_calls,
            rerank_pairs=state.budget.rerank_pairs,
            rounds=state.budget.rounds,
            coverage_gaps=state.coverage.uncovered if state.coverage else (),
            verification=state.verification,
            fallback_error=fallback_error,
        )
        safe_record(
            recorder,
            "agent.terminated",
            reason=result.termination_reason,
            support=result.support,
            mode=result.mode,
            llm_calls=result.llm_calls,
            rerank_pairs=result.rerank_pairs,
            rounds=result.rounds,
        )
        safe_record(recorder, "run.completed")
        return result
