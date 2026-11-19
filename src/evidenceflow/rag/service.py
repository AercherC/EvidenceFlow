"""Compose the formal Retrieval, Context, Generation, and Citation layers into a single-turn RAG."""

from __future__ import annotations

from typing import Protocol

from evidenceflow.citation import assemble_rag_answer
from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.domain.chunk import (
    RerankedRetrievedChunk,
    RetrievedChunk,
)
from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.generation import GeneratedAnswer
from evidenceflow.tracing import TraceRecorder, record_built_context, resolve_recorder, safe_record


class RetrieverProtocol(Protocol):
    """The minimal formal retrieval interface RAGService needs."""

    def retrieve(
        self, query: str
    ) -> list[RetrievedChunk | RerankedRetrievedChunk]: ...


class ContextBuilderProtocol(Protocol):
    """The minimal Context-building interface RAGService needs."""

    def build(
        self,
        retrieved_chunks: list[RetrievedChunk | RerankedRetrievedChunk],
    ) -> BuiltContext: ...


class GeneratorProtocol(Protocol):
    """The minimal Grounded Generation interface RAGService needs."""

    def generate(self, query: str, context: BuiltContext) -> GeneratedAnswer: ...


class RAGService:
    """Orchestrate the existing components in the frozen order only, and return a Citation-validated RAGAnswer."""

    def __init__(
        self,
        retriever: RetrieverProtocol,
        context_builder: ContextBuilderProtocol,
        generator: GeneratorProtocol,
        *,
        trace_recorder: TraceRecorder | None = None,
    ) -> None:
        """Inject already-created components so the Service never implicitly loads expensive Runtime dependencies.

                ``trace_recorder`` is optional (default None): it records only run.*, context.*, and
                citation_validation.completed, and never changes answers or exceptions (PROJECT_SPEC §12.5).
                Retrieval and generation internal events are recorded by each component's own recorder.
        """

        self._retriever = retriever
        self._context_builder = context_builder
        self._generator = generator
        self.trace_recorder = trace_recorder

    def answer(self, query: str) -> RAGAnswer:
        """Run Retrieve->Context->Generate->Citation, letting errors propagate unchanged."""

        # The blank-query check stays first: the run has not started yet, so no event is recorded.
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must not be blank")
        recorder = resolve_recorder(self.trace_recorder)
        safe_record(recorder, "run.started")
        # Track the current stage name so a failure can write run.failed.stage.
        stage = "retrieval"
        try:
            # Pass no override parameters to the Retriever, so the formal Final Top-K frozen by the
            # Composition Root is used.
            retrieved_chunks = self._retriever.retrieve(query)
            stage = "context"
            safe_record(recorder, "context.started")
            built_context = self._context_builder.build(retrieved_chunks)
            record_built_context(recorder, built_context)
            stage = "generation"
            generated_answer = self._generator.generate(query, built_context)
            stage = "citation_validation"
            answer = assemble_rag_answer(
                query=query,
                generated_answer=generated_answer,
                built_context=built_context,
                retrieved_chunks=retrieved_chunks,
                trace_recorder=recorder,
            )
        except Exception as exc:
            # Record run.failed first (only the exception class name and stage, never the message),
            # then re-raise as is:
            # no wrapping and no chain rewriting, so callers (CLI error-prefix mapping, etc.) see
            # exactly the same exception as without tracing.
            safe_record(recorder, "run.failed", error_type=type(exc).__name__, stage=stage)
            raise
        safe_record(recorder, "run.completed")
        return answer
