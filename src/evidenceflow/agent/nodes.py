"""Agent nodes: pure ``node(state, deps) -> state`` steps over the existing pipeline components."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from evidenceflow.agent.policy import (
    charge_llm, charge_round, check_deadline, generation_call_count, guard_loop, require_llm_call,
)
from evidenceflow.agent.coverage import Widener, named_papers
from evidenceflow.agent.state import AgentState, Clause, Coverage, Plan, Terminal
from evidenceflow.agent.verify import verify_against_context
from evidenceflow.citation import assemble_rag_answer
from evidenceflow.domain.chunk import RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.rag.service import ContextBuilderProtocol, GeneratorProtocol
from evidenceflow.retrieval.clause_branch import has_cjk, split_clauses
from evidenceflow.tracing import TraceRecorder, record_built_context, safe_record


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
    widener: Widener | None = None


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


def coverage_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """Which named papers have no selected chunk. Without a widener (no paper names) nothing is
    targeted."""

    papers = deps.widener.papers if deps.widener is not None else {}
    targets = tuple(named_papers(state.query, papers))
    coverage = Coverage(targets=targets, uncovered=_uncovered(targets, state.selected, papers))
    safe_record(
        deps.recorder,
        "agent.coverage.completed",
        target_count=len(coverage.targets),
        uncovered_count=len(coverage.uncovered),
    )
    return state.model_copy(update={"coverage": coverage})


def widen_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """At most once: append the best paper-scoped chunk for each uncovered named paper. Chunks that
    would not fit the context budget are dropped again, so the earlier selection always fits."""

    if deps.widener is None or not state.coverage.uncovered or state.widened:
        return state
    check_deadline(state, deps.clock)
    uncovered = state.coverage.uncovered
    state = charge_round(state, deps.widener.max_pairs(len(uncovered)))
    state = guard_loop(state, "widen", list(uncovered))
    targets = [(name, _clause_naming(state, name, deps.widener.papers)) for name in uncovered]
    result = deps.widener.widen(targets, state.selected)
    added = list(result.added)
    while added and deps.context_builder.build([*state.selected, *added]).skipped_count:
        added.pop()
    selected = (*state.selected, *added)
    papers = deps.widener.papers
    coverage = Coverage(
        targets=state.coverage.targets, uncovered=_uncovered(state.coverage.targets, selected, papers)
    )
    safe_record(
        deps.recorder,
        "agent.widen.completed",
        paper_count=min(len(uncovered), deps.widener.max_papers),
        pair_count=result.pair_count,
        added_count=len(added),
        uncovered_after=len(coverage.uncovered),
    )
    return state.model_copy(update={"selected": selected, "coverage": coverage, "widened": True})


def synthesize_node(state: AgentState, deps: AgentDeps) -> AgentState:
    """对照路径沿用生产生成、引用校验和证据 Trace 口径。"""

    check_deadline(state, deps.clock)
    require_llm_call(state)
    recorder = deps.recorder
    safe_record(recorder, "context.started")
    context = deps.context_builder.build(list(state.selected))
    record_built_context(recorder, context)
    generated = deps.generator.generate(state.query, context)
    state = charge_llm(state, generation_call_count(deps.generator, context))
    answer = assemble_rag_answer(
        query=state.query,
        generated_answer=generated,
        built_context=context,
        retrieved_chunks=list(state.selected),
        trace_recorder=recorder,
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
        partial = (state.verification is not None and not state.verification.passed) or bool(
            state.coverage is not None and state.coverage.uncovered
        )
        terminal = Terminal(reason="answered", support="partial" if partial else "full")
    else:
        terminal = Terminal(reason="insufficient_context", support="none")
    return state.model_copy(update={"terminal": terminal})


def _uncovered(targets, selected, papers) -> tuple[str, ...]:
    files = {item.chunk.file_name for item in selected}
    return tuple(name for name in targets if papers[name] not in files)


def _clause_naming(state: AgentState, name: str, papers) -> str:
    """The first clause that names the paper; the whole query when none does on its own."""

    for clause in state.plan.clauses:
        if named_papers(clause.text, {name: papers[name]}):
            return clause.text
    return state.query


def _lang(text: str) -> str:
    return "zh" if has_cjk(text) else "en"
