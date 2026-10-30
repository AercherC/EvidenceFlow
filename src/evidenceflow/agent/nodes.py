"""Agent nodes: pure ``node(state, deps) -> state`` steps over the existing pipeline components."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from evidenceflow.agent.policy import charge_llm, charge_round, check_deadline, guard_loop, require_llm_call
from evidenceflow.agent.state import AgentState, Clause, Plan, Terminal
from evidenceflow.agent.verify import verify_against_context
from evidenceflow.citation import assemble_rag_answer, parse_citations
from evidenceflow.domain.chunk import RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.rag.service import ContextBuilderProtocol, GeneratorProtocol
from evidenceflow.retrieval.clause_branch import has_cjk, split_clauses
from evidenceflow.tracing import TraceRecorder, safe_record


class AgentRetrieverProtocol(Protocol):
    def retrieve(self, query: str) -> list[RetrievedChunk | RerankedRetrievedChunk]: ...


@dataclass(frozen=True)
class AgentDeps:
    """The tools a node may use. ``single_round_pairs`` / ``multi_round_pairs`` are the reranker-pair
    caps of one retrieval round for each plan mode."""

    retriever: AgentRetrieverProtocol
    context_builder: ContextBuilderProtocol
    generator: GeneratorProtocol
    single_round_pairs: int
    multi_round_pairs: int
    clock: Callable[[], float]
    recorder: TraceRecorder | None


def plan_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Deterministic plan: the R2 split rule (the same one the clause-branch retriever applies)."""

    clauses = split_clauses(state.query)
    if clauses is None:
        plan = Plan(mode="single", clauses=(Clause(text=state.query, lang=_lang(state.query)),))
    else:
        plan = Plan(mode="multi", clauses=tuple(Clause(text=c, lang=_lang(c)) for c in clauses))
    safe_record(deps.recorder, "agent.plan.completed", mode=plan.mode, clause_count=len(plan.clauses))
    return state.model_copy(update={"plan": plan})


def retrieve_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """One retrieval round: production retrieval for ``single``, clause branches (W) for ``multi``."""

    check_deadline(state, deps.clock)
    pairs = deps.multi_round_pairs if state.plan.mode == "multi" else deps.single_round_pairs
    state = charge_round(state, pairs)
    state = guard_loop(state, "retrieve", [c.text for c in state.plan.clauses])
    selected = deps.retriever.retrieve(state.query)
    return state.model_copy(update={"selected": tuple(selected)})


def synthesize_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Context -> generation -> citation validation, with the production prompt and parser."""

    check_deadline(state, deps.clock)
    require_llm_call(state)
    recorder = deps.recorder
    safe_record(recorder, "context.started")
    context = deps.context_builder.build(list(state.selected))
    if recorder is not None:
        # Same payload as RAGService: labels, IDs, token counts and the context sha256, never text.
        safe_record(
            recorder,
            "context.completed",
            included_count=context.included_count,
            skipped_count=context.skipped_count,
            total_tokens=context.total_tokens,
            entries=[
                {"source_label": e.source_label, "chunk_id": e.chunk_id, "token_count": e.token_count}
                for e in context.entries
            ],
            context_text_sha256=hashlib.sha256(context.context_text.encode("utf-8")).hexdigest(),
        )
    generated = deps.generator.generate(state.query, context)
    state = charge_llm(state, _llm_calls(deps.generator, context))
    answer = assemble_rag_answer(
        query=state.query,
        generated_answer=generated,
        built_context=context,
        retrieved_chunks=list(state.selected),
    )
    if recorder is not None:
        parsed = parse_citations(answer.answer, context)
        safe_record(
            recorder,
            "citation_validation.completed",
            valid_labels=list(parsed.valid_labels),
            invalid_labels=list(parsed.invalid_labels),
            source_count=len(answer.sources),
        )
    return state.model_copy(update={"context": context, "generated": generated, "answer": answer})


def verify_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Tier-1 check of an answered response (no LLM). A failure is recorded and lowers ``support``;
    it never changes the answer or its status."""

    if state.answer.status != AnswerStatus.ANSWERED:
        return state
    verification = verify_against_context(state.answer.answer, state.query, state.context)
    safe_record(
        deps.recorder,
        "agent.verify.completed",
        checked=verification.checked,
        unsupported_numbers=sum(1 for item in verification.unsupported if item.startswith("number:")),
        unsupported_terms=sum(1 for item in verification.unsupported if item.startswith("term:")),
    )
    return state.model_copy(update={"verification": verification})


def terminate_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """The generator stays the abstention authority: its status decides the termination reason;
    a tier-1 failure turns ``support`` to partial, never the status."""

    if state.answer.status == AnswerStatus.ANSWERED:
        partial = state.verification is not None and not state.verification.passed
        terminal = Terminal(reason="answered", support="partial" if partial else "full")
    else:
        terminal = Terminal(reason="insufficient_context", support="none")
    return state.model_copy(update={"terminal": terminal})


def _lang(text: str) -> str:
    return "zh" if has_cjk(text) else "en"


def _llm_calls(generator: object, context) -> int:
    """Calls the generator made for this answer, from its per-thread diagnostics when it has them."""

    diagnostics = getattr(generator, "last_diagnostics", None)
    if isinstance(diagnostics, dict) and "usages" in diagnostics:
        return len(diagnostics["usages"])
    return 1 if context.entries else 0
