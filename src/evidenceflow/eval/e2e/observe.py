"""Run one case through the production stage order and keep every stage's result.

This is the single observation implementation shared by ``scripts/e2e_diagnose.py observe`` and the
index comparison entry (PROJECT_SPEC §13.1(c)): the Dense ranking, the Reranker's full ordering, the
final Top-K, the chunks admitted into BuiltContext and, when generation runs, the answer, its actual
citations and the LLM usage of the Generator.

It loads no model and opens no index itself: the retriever / context builder / generator are injected
by the caller, so the same code observes the active index, a read-only staging index, or a test
double without a second evaluation framework.
"""

from __future__ import annotations

import json
import time

from evidenceflow.citation import CitationValidationError, assemble_rag_answer
from evidenceflow.eval.e2e.dataset import EvalCase
from evidenceflow.eval.e2e.diagnosis import (
    E2EObservation,
    GenerationCost,
    ObservedChunk,
)
from evidenceflow.eval.e2e.step0 import CaseRerankScores, score_pool
from evidenceflow.generation import GenerationContractError
from evidenceflow.llm import LLMRequestError
from evidenceflow.retrieval.clause_branch import ClauseBranchRetriever
from evidenceflow.retrieval.reranker import DenseRerankerRetriever
from evidenceflow.tracing import RunTraceRecorder, trace_scope

# These three are generation/citation contract failures, attributed per §13.1(e) rule 1; any
# other exception (retrieval, configuration) aborts the whole run instead.
RECORDED_ERRORS = (LLMRequestError, GenerationContractError, CitationValidationError)


def observe_case(
    case: EvalCase,
    *,
    retriever,
    context_builder,
    generator,
    depth: int,
    retrieval_only: bool,
    collection_name: str,
    chunk_count: int,
    index_signature: str | None = None,
    rerank_depth: int | None = None,
) -> tuple[E2EObservation, CaseRerankScores | None]:
    """Run one case in the ``RAGService.answer`` stage order and keep every stage's result.

    The final Top-K takes the first K items of the production retriever's returned ordering.
    子句检索从同一次调用的 trace 读取实际参与打分的候选，融合结果单独记录。

    With ``rerank_depth`` the Dense Top-``rerank_depth`` is also scored once by the same scorer,
    after the online call, so the online result is never affected by the extra pass; those scores
    come back as the second item (None without ``rerank_depth``).
    """

    query = case.query
    pool_k = retriever.candidate_pool_k
    final_top_k = (
        retriever.top_k_for_query(query)
        if isinstance(retriever, ClauseBranchRetriever)
        else retriever.final_top_k
    )
    dense = retriever.dense_retriever.retrieve(query, top_k=max(depth, pool_k))
    recorder = RunTraceRecorder()
    recorder.start_run(query)
    with trace_scope(recorder):
        reranked = retriever.retrieve(query, top_k=max(final_top_k, retriever.reranker_input_k))
    # 子句路径的各分支均完整打分，保留所有分支候选。基础路径请求的是全部重排结果。
    candidates = None
    for event in recorder.events:
        if event.name == "retrieval.branch.completed":
            by_id = {
                item["chunk_id"]: item
                for branch in event.payload["branches"]
                for item in branch["ranking"]
            }
            candidates = list(by_id.values())
        elif event.name == "rerank.completed" and isinstance(
            retriever, (DenseRerankerRetriever, ClauseBranchRetriever)
        ):
            candidates = [
                ObservedChunk(chunk_id=item.chunk.chunk_id, document_id=item.chunk.document_id)
                for item in reranked
            ]
    final = reranked[:final_top_k]
    context = context_builder.build(final)
    # 子句分支可把全局 Dense Top-K 外的证据送入最终排序；保留这些真实返回项的文档身份，
    # 供归因时区分“全局 Dense 未覆盖”与“生产检索确实找到了”。
    dense_candidates = {item.chunk.chunk_id for item in dense[:pool_k]}
    outside_dense = [item.chunk for item in reranked if item.chunk.chunk_id not in dense_candidates]
    record = {
        "case_id": case.case_id,
        "collection_name": collection_name,
        "index_signature": index_signature,
        "chunk_count": chunk_count,
        "dense_ranking": [
            ObservedChunk(
                chunk_id=item.chunk.chunk_id,
                document_id=item.chunk.document_id,
                heading_path=item.chunk.metadata.get("heading_path"),
            )
            for item in dense
        ],
        "candidate_pool_k": pool_k,
        "reranked_pool": [item.chunk.chunk_id for item in reranked],
        "rerank_candidates": candidates,
        "reranked_outside_dense": [
            ObservedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                heading_path=chunk.metadata.get("heading_path"),
            )
            for chunk in outside_dense
        ],
        "final_top_k": final_top_k,
        "context_chunk_ids": [entry.chunk_id for entry in context.entries],
        "generation_ran": not retrieval_only,
    }
    scores = None if rerank_depth is None else _score_depth(case.case_id, retriever, query, dense, rerank_depth)
    if retrieval_only:
        return E2EObservation(**record), scores
    started = time.perf_counter()
    try:
        generated = generator.generate(query, context)
        answer = assemble_rag_answer(
            query=query, generated_answer=generated, built_context=context, retrieved_chunks=final
        )
    except RECORDED_ERRORS as exc:
        cost = generation_cost(generator, time.perf_counter() - started)
        return E2EObservation(**record, error_type=type(exc).__name__, generation_cost=cost), scores
    observation = E2EObservation(
        **record,
        status=answer.status,
        answer=answer.answer,
        source_chunk_ids=[source.chunk_id for source in answer.sources],
        generation_cost=generation_cost(generator, time.perf_counter() - started),
    )
    return observation, scores


def _score_depth(case_id: str, retriever, query: str, dense, rerank_depth: int) -> CaseRerankScores:
    """Sidecar record: one scoring pass over the Dense Top-``rerank_depth``."""

    if rerank_depth > len(dense):
        raise ValueError(f"--rerank-depth {rerank_depth} exceeds the observed dense depth {len(dense)}")
    reranker = retriever.reranker
    scored = score_pool(reranker.scorer, query, [item.chunk for item in dense[:rerank_depth]],
                        batch_size=reranker.batch_size)
    return CaseRerankScores(
        case_id=case_id,
        dense_scores=[item.similarity_score for item in dense],
        rerank_scores=scored.scores,
        rerank_truncated=scored.truncated,
        rerank_batch_seconds=scored.batch_seconds,
    )


def generation_cost(generator, seconds: float) -> GenerationCost:
    """Wall-clock seconds plus this thread's ``Generator.last_diagnostics``.

    Read right after ``generate`` on the same thread, so the diagnostics belong to this case (also
    when it raised: the usage of every call that returned is already recorded).
    """

    return GenerationCost(seconds=round(seconds, 4), **generation_diagnostics(generator))


def generation_diagnostics(generator) -> dict:
    """从本线程的生成诊断提取评测允许记录的字段，避免写入提供方的额外信息。"""
    diagnostics = generator.last_diagnostics
    return {
        "llm_called": diagnostics["llm_called"],
        "usages": [
            None if usage is None else {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens")}
            for usage in diagnostics["usages"]
        ],
        "repair_attempted": diagnostics["repair_attempted"],
        "repair_succeeded": diagnostics["repair_succeeded"],
    }


def observation_line(observation: E2EObservation) -> str:
    """One JSONL line. An unset ``generation_cost`` (retrieval-only) is left out, so retrieval-only
    lines are the same as before the field existed."""

    exclude = {"generation_cost"} if observation.generation_cost is None else set()
    return json.dumps(observation.model_dump(mode="json", exclude=exclude), ensure_ascii=False, sort_keys=True)
