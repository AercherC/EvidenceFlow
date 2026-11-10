"""The single composition point that assembles the one official EvidenceFlow RAG runtime."""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evidenceflow.agent import AgentService, Widener, load_paper_names
from evidenceflow.config import Settings
from evidenceflow.context import ContextBuilder
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.generation import Generator
from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
from evidenceflow.knowledge import (
    ActiveCollectionError,
    ActiveIndexProtectedError,
    IndexLifecycle,
    LifecycleState,
)
from evidenceflow.llm import OpenAICompatibleLLMClient
from evidenceflow.rag import RAGService
from evidenceflow.rag.service import GeneratorProtocol, RetrieverProtocol
from evidenceflow.retrieval import (
    BM25Index,
    ClauseBranchRetriever,
    DenseRerankerRetriever,
    DenseRetriever,
    ExactDenseIndex,
    ExactDenseIndexError,
    LearnedReranker,
    SentencePieceTerms,
    TransformersCrossEncoderScorer,
)
from evidenceflow.tracing import TraceRecorder
from evidenceflow.storage import (
    ChromaVectorStore,
    VectorStoreConfigurationError,
    read_collection_metadata,
)

logger = logging.getLogger(__name__)

# BGE-M3 Dense vector dimension; used as the expected value only when the embedding provider does
# not declare a dimension of its own.
EMBEDDING_DIMENSION = 1024


def expected_embedding_dimension(embedding_provider: object) -> int:
    """Dimension the Exact index must match: the provider's declared model dimension wins.

    A mismatch means the collection does not match the current embedding model. Reading the
    dimension from the model instead of hard-coding it keeps the check pointed at the model that
    was actually loaded after ``EMBEDDING_MODEL`` changes.
    """

    dimension = getattr(embedding_provider, "dimension", None)
    if isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
        return dimension
    return EMBEDDING_DIMENSION


def build_llm_client(settings: Settings) -> OpenAICompatibleLLMClient:
    """Map Settings to explicit LLM client arguments inside the single Composition Root."""

    return OpenAICompatibleLLMClient(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout_seconds=settings.llm_timeout_seconds,
        max_attempts=settings.llm_max_attempts,
        thinking=settings.llm_thinking,
        trust_env=settings.llm_trust_env,
    )


def _lifecycle_governed(settings: Settings) -> bool:
    """When the lifecycle state file exists, the production collection is decided by the active
    pointer.

    If ``INDEX_STATE_PATH`` was set explicitly (environment variable, .env or constructor argument)
    but the file does not exist, ``ActiveCollectionError`` is raised: silently falling back to
    ``Settings.collection_name`` (whose default is production v2) when an explicit path is
    mistyped or the file is lost would quietly bypass both the active pointer and the ingest
    protection.
    Only when the path was not set explicitly and no file exists at the default path is it treated
    as a brand-new environment.
    """

    path = Path(settings.index_state_path)
    if path.is_file():
        return True
    if "index_state_path" in settings.model_fields_set:
        raise ActiveCollectionError(
            f"INDEX_STATE_PATH={str(path)!r} is set explicitly but the lifecycle state file "
            "does not exist; fix the path or unset INDEX_STATE_PATH"
        )
    return False


def _load_lifecycle_state(settings: Settings) -> LifecycleState:
    """Read and validate the state file; refuse to start when it is corrupt rather than falling
    back to Settings and reading the wrong index."""

    path = Path(settings.index_state_path)
    try:
        return LifecycleState.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ActiveCollectionError(
            f"Cannot read lifecycle state {path}: {type(exc).__name__}: {exc}"
        ) from exc


def _warn_if_unmanaged_fallback(settings: Settings) -> None:
    """Announce that the collection did NOT come from a lifecycle pointer.

    The state file is deployment state and is not version-controlled, so a fresh clone falls back
    to ``Settings.collection_name`` with no pointer at all. That is legitimate for a brand-new
    environment, but it also silently bypasses the ingest guard and any active/previous/rollback
    bookkeeping. A silent fallback makes "which index am I actually serving?" unanswerable, so it is
    logged at warning level instead.
    """

    logger.warning(
        "no lifecycle state file at %s: serving collection %r from Settings.collection_name. "
        "The active pointer is NOT in effect, so ingest protection and rollback are unavailable. "
        "Run scripts/build_staging_index.py or restore the state file if this environment is "
        "supposed to serve a managed index.",
        settings.index_state_path,
        settings.collection_name,
    )


def resolve_active_collection(settings: Settings) -> str:
    """Decide which collection production reads; the single entry point for the runtime's
    collection choice.

    When the state file exists, ``state.active.collection_name`` wins. If the user explicitly set
    ``COLLECTION_NAME`` and it disagrees with active, ``ActiveCollectionError`` is raised instead of
    silently picking one: otherwise a stale .env after activate/rollback would quietly make
    production read the old index.
    When the state file does not exist (tests, brand-new environments) it falls back to
    ``Settings.collection_name``, and says so loudly (see ``_warn_if_unmanaged_fallback``).
    """

    if not _lifecycle_governed(settings):
        _warn_if_unmanaged_fallback(settings)
        return settings.collection_name
    active = _load_lifecycle_state(settings).active.collection_name
    # Only an "explicitly set" value counts as a conflict: a default differing from active is
    # normal (the default only serves environments without a state file).
    if "collection_name" in settings.model_fields_set and settings.collection_name != active:
        raise ActiveCollectionError(
            f"COLLECTION_NAME={settings.collection_name!r} conflicts with the lifecycle active "
            f"collection {active!r} ({settings.index_state_path}); unset COLLECTION_NAME or "
            "activate/rollback the index through IndexLifecycle"
        )
    return active


def guard_ingest_target(settings: Settings, collection_name: str) -> None:
    """Refuse to ingest directly into the ACTIVE index.

    The ACTIVE index may only change through staging build + validation + activate; a direct
    ingest would take IngestionService's "delete first, then write" path and leave a half-updated
    production index on failure. Two checks:
    1. when the state file exists, a target equal to the active pointer -> refuse;
    2. defense in depth (independent of the state file): read the target collection's metadata and
       refuse if ``lifecycle == "ACTIVE"``. This covers a missing or accidentally deleted state
       file. A non-existent collection is allowed through, so the first ingest in a brand-new
       environment works. That query loads no model and creates no collection.
    """

    if _lifecycle_governed(settings):
        active = _load_lifecycle_state(settings).active.collection_name
        if collection_name == active:
            raise ActiveIndexProtectedError(
                f"Refusing to ingest into the ACTIVE collection {active!r}; "
                "build a staging index with scripts/build_staging_index.py instead"
            )
    metadata = read_collection_metadata(settings.vector_db_path, collection_name)
    if metadata is not None and metadata.get("lifecycle") == "ACTIVE":
        raise ActiveIndexProtectedError(
            f"Refusing to ingest into collection {collection_name!r}: its metadata is "
            "lifecycle=ACTIVE; build a staging index with scripts/build_staging_index.py instead"
        )


def _metadata_cache(settings: Settings) -> Callable[[str], dict[str, Any] | None]:
    """Return a read-only metadata reader that queries each Collection at most once per startup.

    Both the lifecycle consistency audit and the active-collection open need the same metadata; the
    cache keeps startup at one read per Collection name instead of one per consumer, and keeps the
    "validate before loading any model" ordering observable in the event trace.
    """

    cache: dict[str, dict[str, Any] | None] = {}

    def read(name: str) -> dict[str, Any] | None:
        if name not in cache:
            cache[name] = read_collection_metadata(settings.vector_db_path, name)
        return cache[name]

    return read


def _open_active_store(
    settings: Settings,
    collection_name: str,
    *,
    read_metadata: Callable[[str], dict[str, Any] | None] | None = None,
) -> ChromaVectorStore:
    """Open the collection resolved from the active pointer; validation completes before any model
    is loaded.

    The metadata is queried read-only first: a missing collection raises a configuration error
    (instead of leaving an empty collection behind as get_or_create would, then failing later with
    an indirect error such as "size is 0"); and a ``lifecycle`` other than ACTIVE is refused too.
    It is then opened with ``create_if_missing=False``, ensuring the read path never creates a
    collection.
    """

    read = read_metadata or (lambda name: read_collection_metadata(settings.vector_db_path, name))
    metadata = read(collection_name)
    if metadata is None:
        raise ActiveCollectionError(
            f"Lifecycle active collection {collection_name!r} does not exist in "
            f"{settings.vector_db_path}; check {settings.index_state_path} and VECTOR_DB_PATH"
        )
    lifecycle = metadata.get("lifecycle")
    if lifecycle != "ACTIVE":
        raise ActiveCollectionError(
            f"Collection {collection_name!r} has metadata lifecycle={lifecycle!r}, "
            "expected 'ACTIVE'"
        )
    try:
        return ChromaVectorStore(settings.vector_db_path, collection_name, create_if_missing=False)
    except VectorStoreConfigurationError as exc:
        raise ActiveCollectionError(
            f"Cannot open lifecycle active collection {collection_name!r}: {exc}"
        ) from exc


def require_lifecycle_consistency(
    settings: Settings,
    *,
    read_metadata: Callable[[str], dict[str, Any] | None] | None = None,
) -> tuple[str, ...]:
    """Report every pointer whose Collection flag disagrees with it.

    Only the active pointer decides what production reads, so a divergence does not stop serving by
    itself -- but it does mean ``rollback`` would restore a Collection whose flag runtime rejects, or
    that a crash left two Collections claiming ACTIVE. Reporting it at startup is far cheaper than
    discovering it during an incident, and the caller decides whether it is fatal. The documented
    benign case (two ACTIVE Collections after a crash between promote and demote) must not take the
    service down, so ``build_retriever`` logs and continues, while ``repair_lifecycle_flags.py``
    offers the fix.
    """

    if not _lifecycle_governed(settings):
        return ()
    state = _load_lifecycle_state(settings)
    lifecycle = IndexLifecycle(settings.index_state_path, state.knowledge_base_id)
    read = read_metadata or (lambda name: read_collection_metadata(settings.vector_db_path, name))

    def read_flag(name: str) -> str | None:
        metadata = read(name)
        return None if metadata is None else metadata.get("lifecycle")

    return lifecycle.verify_consistency(read_flag)


def build_ingestion_service(settings: Settings) -> IngestionService:
    """Create all runtime dependencies of the official ingestion flow from the unified Settings.

    Protection checks run before any model is loaded: a target that is the active collection is
    refused immediately.
    """

    collection_name = resolve_active_collection(settings)
    guard_ingest_target(settings, collection_name)
    return IngestionService(
        settings.source_docs_path,
        BgeM3Tokenizer(settings.embedding_model),
        BgeM3EmbeddingProvider(settings.embedding_model, device=settings.model_device),
        ChromaVectorStore(settings.vector_db_path, collection_name),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        chunk_strategy=settings.chunk_strategy,
    )


def build_retriever(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> DenseRerankerRetriever | ClauseBranchRetriever:
    """Assemble production retrieval: Exact Dense Top50 -> Learned Reranker -> Top6 (Settings defaults).

    With ``multi_doc_strategy="clause_branch"`` the same components are wrapped in
    ``ClauseBranchRetriever``: queries the R2 rule splits get clause branches, every other query
    still goes through the DenseRerankerRetriever built here, unchanged.

    ``trace_recorder`` is optional (default None) and is passed through to
    ``DenseRerankerRetriever`` unchanged; it only observes and does not change behavior.

    Dense candidates use the in-process ``ExactDenseIndex`` (numpy.exact) rather than Chroma HNSW:
    HNSW's approximate results are not guaranteed to be identical across processes, which would
    make A/B experiments non-reproducible question by question; while the full inner-product cost
    of 817 vectors is negligible. Chroma remains the single persistent source.
    """

    collection_name = resolve_active_collection(settings)
    # Collection existence and lifecycle validation come before the embedding model is loaded: a
    # configuration error should surface immediately instead of waiting for model loading to
    # finish. A collection resolved from the active pointer must carry lifecycle=ACTIVE, so the
    # pointer cannot target an unactivated or STAGING-marked collection; a brand-new environment
    # without a state file has no such marker, so it is not validated and keeps the original open
    # order/mode (get_or_create).
    governed = _lifecycle_governed(settings)
    read_metadata = _metadata_cache(settings)
    if governed:
        # A pointer/flag divergence is reported, not fatal to reads: the active pointer alone
        # decides what is served, so an over-privileged previous Collection (the documented
        # post-crash state) must not take the service down. ``verify_active_index.py`` and
        # ``repair_lifecycle_flags.py`` act on the report.
        problems = require_lifecycle_consistency(settings, read_metadata=read_metadata)
        if problems:
            logger.warning("lifecycle pointer/flag divergence: %s", "; ".join(problems))
    vector_store = (
        _open_active_store(settings, collection_name, read_metadata=read_metadata)
        if governed
        else None
    )
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    if vector_store is None:
        vector_store = ChromaVectorStore(settings.vector_db_path, collection_name)
    # At startup read every vector out of Chroma and build the index; no second persistent vector
    # file is ever written.
    pairs = list(vector_store.iter_all_chunks_with_embeddings())
    exact_index = ExactDenseIndex(
        [chunk for chunk, _ in pairs], [embedding for _, embedding in pairs]
    )
    # Startup validation: the index size must equal the collection record count and the dimension
    # must match the embedding model; otherwise an incomplete or mismatched index was read, so
    # refuse to start instead of silently degrading.
    if exact_index.size != vector_store.count:
        raise ExactDenseIndexError(
            f"Exact index size {exact_index.size} does not match "
            f"collection count {vector_store.count}"
        )
    expected_dimension = expected_embedding_dimension(embedding_provider)
    if exact_index.dimension != expected_dimension:
        raise ExactDenseIndexError(
            f"Exact index dimension {exact_index.dimension} != embedding model "
            f"dimension {expected_dimension}"
        )
    dense_retriever = DenseRetriever(
        embedding_provider,
        exact_index,
        top_k=settings.reranker_input_k,
    )
    scorer = TransformersCrossEncoderScorer(
        settings.reranker_model,
        device=settings.model_device,
        dtype=settings.reranker_dtype,
        max_length=settings.reranker_max_length,
    )
    retriever = DenseRerankerRetriever(
        dense_retriever,
        LearnedReranker(
            scorer,
            batch_size=settings.reranker_batch_size,
            max_candidates=settings.reranker_input_k,
        ),
        candidate_pool_k=settings.reranker_input_k,
        reranker_input_k=settings.reranker_input_k,
        final_top_k=settings.reranker_final_k,
        trace_recorder=trace_recorder,
    )
    if settings.multi_doc_strategy == "off":
        return retriever
    # BM25 covers the same chunks as the Exact index, read from the same Chroma snapshot.
    bm25 = (
        BM25Index(
            {chunk.chunk_id: chunk.content for chunk, _ in pairs},
            SentencePieceTerms(settings.embedding_model),
        )
        if settings.clause_branch_bm25_k
        else None
    )
    return ClauseBranchRetriever(
        retriever,
        bm25,
        dense_depth=exact_index.size,
        clause_pool_k=settings.clause_branch_pool_k,
        bm25_k=settings.clause_branch_bm25_k,
        pair_budget=settings.clause_branch_pair_budget,
        final_top_k=settings.clause_branch_final_k,
        trace_recorder=trace_recorder,
    )


def build_rag_service(
    settings: Settings,
    *,
    retriever: RetrieverProtocol | None = None,
    generator: GeneratorProtocol | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> RAGService:
    """Create the single official Exact Dense Top50 -> Learned Reranker -> Top6 pipeline (Settings defaults).

    ``retriever`` and ``generator`` are only for evaluation scripts to inject (to record the final Top-K, read
    Generator.last_diagnostics, or wrap the LLM client for rate limiting and raw-reply retention);
    when not provided, the official assembly is used. Every other component is always created by
    this function from Settings.

    ``trace_recorder`` is passed to the retriever, generator and RAGService that this function
    assembles itself; a retriever / generator injected by the caller is not automatically given
    the recorder (PROJECT_SPEC §12.5).
    """

    retriever = (
        retriever
        if retriever is not None
        else build_retriever(settings, trace_recorder=trace_recorder)
    )
    context_builder = build_context_builder(settings)
    generator = (
        generator
        if generator is not None
        else build_generator(settings, trace_recorder=trace_recorder)
    )
    return RAGService(retriever, context_builder, generator, trace_recorder=trace_recorder)


def build_agent_service(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> AgentService:
    """Assemble the agent over the production components (``ask --agent``).

    The agent's multi-clause retrieval is always the clause-branch retriever, whatever
    ``multi_doc_strategy`` says; single questions still go through Dense Top50 -> Reranker -> Top6.
    The fallback is a plain RAGService over the same components, built without its own recorder so
    a fallback run does not open a second run in the trace.
    """

    agent_settings = (
        settings
        if settings.multi_doc_strategy == "clause_branch"
        else settings.model_copy(update={"multi_doc_strategy": "clause_branch"})
    )
    retriever = build_retriever(agent_settings, trace_recorder=trace_recorder)
    context_builder = build_context_builder(settings)
    generator = build_generator(settings, trace_recorder=trace_recorder)
    fallback = RAGService(retriever, context_builder, generator)
    papers = load_paper_names(settings.agent_paper_manifest)
    widener = (
        Widener(
            retriever.base.dense_retriever,
            retriever.base.reranker,
            papers,
            dense_depth=retriever.dense_depth,
        )
        if papers
        else None
    )
    return AgentService(
        retriever,
        context_builder,
        generator,
        fallback=fallback.answer,
        single_round_pairs=settings.reranker_input_k,
        multi_round_pairs=settings.clause_branch_pair_budget,
        llm_calls_max=settings.agent_llm_calls_max,
        rerank_pairs_max=settings.agent_rerank_pairs_max,
        rounds_max=settings.agent_rounds_max,
        deadline_seconds=settings.agent_deadline_seconds,
        widener=widener,
        trace_recorder=trace_recorder,
    )


def build_context_builder(settings: Settings) -> ContextBuilder:
    """Production Context assembly: the same Tokenizer as Embedding, budget taken from Settings.

    The evaluation diagnosis scripts observe retrieval, context and generation stage by stage;
    reusing this function keeps them on exactly the production wiring.
    """

    return ContextBuilder(BgeM3Tokenizer(settings.embedding_model), settings.max_context_tokens)


def build_generator(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> Generator:
    """Production Generator assembly; like ``build_context_builder``, reused by stage diagnosis."""

    return Generator(build_llm_client(settings), trace_recorder=trace_recorder)
