"""The single composition point that assembles the one official EvidenceFlow RAG runtime."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from functools import partial
from collections.abc import Callable, Collection
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from evidenceflow.agent import AgentService, EvidenceTools, Widener, load_paper_names
from evidenceflow.agent.adaptive import AdaptiveAgentService

if TYPE_CHECKING:
    from evidenceflow.agent.experimental import AdaptiveAgentService as HybridAgentService
from evidenceflow.config import Settings
from evidenceflow.context import ContextBuilder
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.evidence import EvidenceIndexIdentity
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.generation import Generator
from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
from evidenceflow.knowledge import (
    ActiveCollectionError,
    ActiveIndexProtectedError,
    IndexLifecycle,
    IndexPointer,
    LifecycleState,
)
from evidenceflow.llm import OpenAICompatibleLLMClient
from evidenceflow.rag import RAGService
from evidenceflow.rag.service import GeneratorProtocol, RetrieverProtocol
from evidenceflow.retrieval.clause_branch import paper_passage
from evidenceflow.retrieval import (
    BM25Index,
    ClauseBranchRetriever,
    DenseRerankerRetriever,
    DenseRetriever,
    EvidenceNavigator,
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
    """Assemble production retrieval: Exact Dense Top50 -> Learned Reranker -> Top8 (Settings defaults).

    With ``multi_doc_strategy="clause_branch"`` the same components are wrapped in
    ``ClauseBranchRetriever``: queries the R2 rule splits get clause branches, every other query
    still goes through the DenseRerankerRetriever built here, unchanged.

    ``trace_recorder`` is optional (default None) and is passed through to
    ``DenseRerankerRetriever`` unchanged; it only observes and does not change behavior.

    Dense candidates use the in-process ``ExactDenseIndex`` (numpy.exact) rather than Chroma HNSW:
    HNSW's approximate results are not guaranteed to be identical across processes, which would
    make A/B experiments non-reproducible question by question; while the full inner-product cost
    of 817 vectors is negligible. Chroma remains the single persistent source.

    This is a thin wrapper over ``_read_index_snapshot`` + ``_build_retriever_from_snapshot``: the
    public contract (returned type, validation order, errors) is unchanged, and
    ``build_evidence_tools`` reuses the same two steps over its own single read.
    """

    snapshot = _read_index_snapshot(settings)
    return _build_retriever_from_snapshot(settings, snapshot, trace_recorder=trace_recorder)


@dataclass(frozen=True)
class _IndexSnapshot:
    """检索、BM25 和邻块阅读共用同一次集合读取；向量只由 Exact 矩阵持有。"""

    chunks: tuple[DocumentChunk, ...]
    embedding_provider: BgeM3EmbeddingProvider
    exact_index: ExactDenseIndex
    identity: EvidenceIndexIdentity | None = None


def _read_exact_index(
    store: ChromaVectorStore, *, validation_error: type[Exception] = ExactDenseIndexError,
) -> tuple[tuple[DocumentChunk, ...], ExactDenseIndex]:
    """读取同一批块和向量构建 Exact 索引，返回后释放集合返回的原始向量。"""
    pairs = list(store.iter_all_chunks_with_embeddings())
    chunks = tuple(chunk for chunk, _ in pairs)
    index = ExactDenseIndex(chunks, [embedding for _, embedding in pairs])
    count = store.count
    if index.size != count:
        raise validation_error(f"Exact index size {index.size} does not match collection count {count}")
    return chunks, index


def _validate_exact_dimension(
    index: ExactDenseIndex,
    embedding_provider: BgeM3EmbeddingProvider,
    *,
    validation_error: type[Exception] = ExactDenseIndexError,
) -> None:
    """按实际加载的模型维度校验索引，错误类型沿用调用入口的合同。"""
    dimension = expected_embedding_dimension(embedding_provider)
    if index.dimension != dimension:
        raise validation_error(f"Exact index dimension {index.dimension} != embedding model dimension {dimension}")


def _read_index_snapshot(
    settings: Settings,
    *,
    read_only: bool = False,
    bind_identity: bool = False,
) -> _IndexSnapshot:
    """Open the serving Collection once and read every chunk/vector into memory.

    ``read_only=True`` is the evidence-tool path: when no lifecycle state file governs the
    environment, the configured collection must already exist and be a serving index, and it is
    opened with ``create_if_missing=False``. Otherwise a mistyped ``COLLECTION_NAME`` would create an
    empty collection as a side effect of a read. The retriever path keeps its historical
    get_or_create behaviour.

    ``bind_identity=True`` captures the lifecycle pointer *before* the read and re-checks it
    afterwards, so the returned identity describes exactly the snapshot that was read; a pointer
    that moved at any point in between is refused (PROJECT_SPEC §10.2).
    """

    collection_name = resolve_active_collection(settings)
    # Collection existence and lifecycle validation come before the embedding model is loaded: a
    # configuration error should surface immediately instead of waiting for model loading to
    # finish. A collection resolved from the active pointer must carry lifecycle=ACTIVE, so the
    # pointer cannot target an unactivated or STAGING-marked collection; a brand-new environment
    # without a state file has no such marker, so it is not validated and keeps the original open
    # order/mode (get_or_create).
    governed = _lifecycle_governed(settings)
    pointer_before = None
    if governed and bind_identity:
        # Only the identity-bound path pays for the second read of the pointer; ``build_retriever``
        # keeps its historical state-file access pattern.
        pointer_before = _load_lifecycle_state(settings).active
        if collection_name != pointer_before.collection_name:
            raise ActiveCollectionError(
                "active index changed before its snapshot was read; retry the run"
            )
    read_metadata = _metadata_cache(settings)
    if governed:
        # A pointer/flag divergence is reported, not fatal to reads: the active pointer alone
        # decides what is served, so an over-privileged previous Collection (the documented
        # post-crash state) must not take the service down. ``verify_active_index.py`` and
        # ``repair_lifecycle_flags.py`` act on the report.
        problems = require_lifecycle_consistency(settings, read_metadata=read_metadata)
        if problems:
            logger.warning("lifecycle pointer/flag divergence: %s", "; ".join(problems))
    if read_only and not governed:
        metadata = read_metadata(collection_name)
        if metadata is None or metadata.get("lifecycle") not in {None, "ACTIVE"}:
            raise ActiveCollectionError(
                f"Collection {collection_name!r} is missing or is not a serving index"
            )
    vector_store = (
        _open_active_store(settings, collection_name, read_metadata=read_metadata)
        if governed
        else None
    )
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    if vector_store is None:
        # The read-only call shape is kept distinct on purpose: the retriever path must keep calling
        # the store exactly as before (no new keyword), so existing stores and test doubles are
        # unaffected.
        vector_store = (
            ChromaVectorStore(
                settings.vector_db_path, collection_name, create_if_missing=False
            )
            if read_only
            else ChromaVectorStore(settings.vector_db_path, collection_name)
        )
    chunks, exact_index = _read_exact_index(vector_store)
    _validate_exact_dimension(exact_index, embedding_provider)
    identity = (
        _bind_identity_to_snapshot(settings, collection_name, exact_index.size, pointer_before)
        if bind_identity
        else None
    )
    return _IndexSnapshot(
        chunks=chunks,
        embedding_provider=embedding_provider,
        exact_index=exact_index,
        identity=identity,
    )


def _bind_identity_to_snapshot(
    settings: Settings,
    collection_name: str,
    chunk_count: int,
    pointer_before: IndexPointer | None = None,
) -> EvidenceIndexIdentity:
    """Describe the read snapshot as one index, refusing a count that contradicts the pointer.

    Managed: the identity is the lifecycle pointer, so two indexes sharing a collection name and
    chunk count are still distinguishable. The pointer is re-read here (after the chunks), so a
    pointer that moved during the read is detected by comparing the full pointers, including the
    signature even when collection name and chunk count stayed the same. Unmanaged: there is no pointer and
    therefore no signature, so the identity says ``settings_fallback`` rather than inventing one.
    """

    if not _lifecycle_governed(settings):
        return EvidenceIndexIdentity(
            collection_name=collection_name,
            index_signature=None,
            chunk_count=chunk_count,
            governance="settings_fallback",
        )
    active = _load_lifecycle_state(settings).active
    if pointer_before is not None and pointer_before != active:
        raise ActiveCollectionError(
            "active index changed while its snapshot was being read; retry the run"
        )
    if collection_name != active.collection_name:
        raise ActiveCollectionError(
            "active index changed while its snapshot was being read; retry the run"
        )
    if chunk_count != active.chunk_count:
        raise ActiveCollectionError(
            f"index {collection_name!r} changed while its snapshot was being read: the snapshot "
            f"has {chunk_count} chunks but the active pointer declares {active.chunk_count}"
        )
    return EvidenceIndexIdentity(
        collection_name=collection_name,
        index_signature=active.index_signature,
        chunk_count=active.chunk_count,
        governance="lifecycle_active",
    )


def _build_retriever_from_snapshot(
    settings: Settings,
    snapshot: _IndexSnapshot,
    *,
    trace_recorder: TraceRecorder | None = None,
) -> DenseRerankerRetriever | ClauseBranchRetriever:
    """Wire retrieval over an already-read snapshot; opens no collection and reads no vector twice."""

    exact_index = snapshot.exact_index
    dense_retriever = DenseRetriever(
        snapshot.embedding_provider,
        exact_index,
        top_k=settings.reranker_input_k,
    )

    def build_reranker(model: str, max_length: int, max_candidates: int) -> LearnedReranker:
        """两种重排器共用设备与批大小，候选容量由对应检索路径的配额决定。"""
        return LearnedReranker(
            TransformersCrossEncoderScorer(
                model,
                device=settings.model_device,
                dtype=settings.reranker_dtype,
                max_length=max_length,
            ),
            batch_size=settings.reranker_batch_size,
            max_candidates=max_candidates,
        )

    base_depth = settings.reranker_input_k
    extra_pairs = 0
    if settings.multi_doc_strategy == "clause_branch" and settings.clause_branch_single_query_bm25:
        # 给单句的 BM25 追加留容量，基础 Dense 深度和 Agent 每次搜索上界不变。
        extra_pairs = min(settings.clause_branch_bm25_k, max(0, settings.clause_branch_pair_budget - base_depth))
    retriever = DenseRerankerRetriever(
        dense_retriever,
        build_reranker(settings.reranker_model, settings.reranker_max_length, base_depth + extra_pairs),
        candidate_pool_k=settings.reranker_input_k,
        reranker_input_k=settings.reranker_input_k,
        final_top_k=settings.reranker_final_k,
        trace_recorder=trace_recorder,
        passage_text=paper_passage if settings.reranker_use_document_context else None,
        dense_rrf_weight=settings.reranker_dense_rrf_weight,
    )
    if settings.multi_doc_strategy == "off":
        return retriever
    # BM25 covers the same chunks as the Exact index, read from the same Chroma snapshot.
    bm25 = (
        BM25Index(
            {chunk.chunk_id: chunk.content for chunk in snapshot.chunks},
            SentencePieceTerms(settings.embedding_model),
        )
        if settings.clause_branch_bm25_k
        else None
    )
    paper_names = load_paper_names(settings.clause_branch_paper_manifest)
    file_names, explicit_names = _resolve_paper_aliases(snapshot, paper_names)
    paper_aliases = {**file_names, **explicit_names} if settings.clause_branch_file_aliases else explicit_names
    paper_reranker = (
        build_reranker("cross-encoder/ms-marco-MiniLM-L6-v2", min(settings.reranker_max_length, 512), base_depth)
        if paper_names and paper_aliases else None
    )
    return ClauseBranchRetriever(
        retriever,
        bm25,
        dense_depth=exact_index.size,
        clause_pool_k=settings.clause_branch_pool_k,
        bm25_k=settings.clause_branch_bm25_k,
        pair_budget=settings.clause_branch_pair_budget,
        final_top_k=settings.clause_branch_final_k,
        fusion_order=settings.clause_branch_fusion_order,
        trace_recorder=trace_recorder,
        paper_aliases=paper_aliases,
        exact_paper_names=file_names,
        paper_reranker=paper_reranker,
        single_query_bm25=settings.clause_branch_single_query_bm25,
        parallel_papers=settings.clause_branch_parallel_papers,
        balance_two_part_queries=settings.clause_branch_balance_two_part_queries,
    )


def _resolve_paper_aliases(
    snapshot: _IndexSnapshot, paper_names: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    """从同一快照解析唯一 PDF 名和显式别名，保留两类命名的启用范围。"""
    documents_by_file: dict[str, set[str]] = defaultdict(set)
    documents_by_stem: dict[str, set[str]] = defaultdict(set)
    for chunk in snapshot.chunks:
        documents_by_file[chunk.file_name].add(chunk.document_id)
        if Path(chunk.file_name).suffix.casefold() == ".pdf":
            stem = Path(chunk.file_name).stem.replace("_", " ").strip().casefold()
            if stem:
                documents_by_stem[stem].add(chunk.document_id)
    file_aliases = {stem: next(iter(ids)) for stem, ids in documents_by_stem.items() if len(ids) == 1}
    explicit_aliases = {
        alias.casefold(): next(iter(documents_by_file[file_name]))
        for alias, file_name in paper_names.items()
        if len(documents_by_file[file_name]) == 1
    }
    return file_aliases, explicit_aliases


class IndexEvaluationError(RuntimeError):
    """只读评测指定的索引无法被证明可测：指针缺失、flag 不符、读取期间指针移动或数量不一致。"""


@dataclass(frozen=True)
class EvaluationIndex:
    """一次只读读取得到的评测索引，以及固定其身份的指针签名。

    评测入口（active/staging 对照、三路径同题比较）都从这里取检索器与 Context 构建器，
    保证同一次评测只用一份快照、同一套模型与配置；``staging`` 目标只在内存中读取，
    从不改动 active 指针，也不把 staging 标记为 ACTIVE。
    """

    identity: EvidenceIndexIdentity
    snapshot: _IndexSnapshot
    retriever: DenseRerankerRetriever | ClauseBranchRetriever
    context_builder: ContextBuilder

    @property
    def chunks(self) -> tuple[DocumentChunk, ...]:
        """返回这份快照里的全部 Chunk（与检索器同源，可直接复用做 gold 映射）。"""

        return self.snapshot.chunks


def _evaluation_pointer(settings: Settings, target: Literal["active", "staging"]) -> IndexPointer:
    """读取评测目标的 lifecycle 指针；没有状态文件时拒绝，因为无法给出真实 index_signature。"""

    if not _lifecycle_governed(settings):
        raise IndexEvaluationError(
            f"no lifecycle state file at {settings.index_state_path}: the {target} index_signature "
            "is unavailable, so an evaluation cannot be bound to a real index identity"
        )
    state = _load_lifecycle_state(settings)
    pointer = state.active if target == "active" else state.staging
    if pointer is None:
        raise IndexEvaluationError(
            f"no staging pointer is registered in {settings.index_state_path}; build a staging index "
            "before evaluating it"
        )
    return pointer


def describe_evaluation_index(
    settings: Settings, *, target: Literal["active", "staging"] = "active"
) -> EvidenceIndexIdentity:
    """只读指针给出了哪个索引的身份；不打开 Collection、不加载模型（用于计划与预检输出）。"""

    pointer = _evaluation_pointer(settings, target)
    return EvidenceIndexIdentity(
        collection_name=pointer.collection_name,
        index_signature=pointer.index_signature,
        chunk_count=pointer.chunk_count,
        governance="lifecycle_active" if target == "active" else "lifecycle_staging",
    )


def build_evaluation_index(
    settings: Settings,
    *,
    target: Literal["active", "staging"] = "active",
    trace_recorder: TraceRecorder | None = None,
) -> EvaluationIndex:
    """只读装配指定 active / staging 索引，供评测使用；active 指针绝不临时切换。

    与线上共用 ``Settings``（Embedding / Reranker / 候选深度 / Context 预算完全一致），
    因此 staging 结果不是另一套模型配置的产物。集合只以 ``create_if_missing=False`` 打开，
    读取前后各读一次目标指针并要求完全相等：读取期间指针移动、签名变化或数量不符一律拒绝，
    不把旧快照标成新签名。staging 集合必须带 ``lifecycle=STAGING``，active 集合必须带
    ``lifecycle=ACTIVE``，否则同样拒绝。
    """

    pointer_before = _evaluation_pointer(settings, target)
    collection_name = pointer_before.collection_name
    metadata = read_collection_metadata(settings.vector_db_path, collection_name)
    if metadata is None:
        raise IndexEvaluationError(
            f"{target} collection {collection_name!r} does not exist in {settings.vector_db_path}"
        )
    expected_flag = pointer_before.status
    if metadata.get("lifecycle") != expected_flag:
        raise IndexEvaluationError(
            f"collection {collection_name!r} has lifecycle={metadata.get('lifecycle')!r}, expected "
            f"{expected_flag!r} for the {target} pointer"
        )
    try:
        store = ChromaVectorStore(
            settings.vector_db_path, collection_name, create_if_missing=False
        )
    except VectorStoreConfigurationError as exc:
        raise IndexEvaluationError(
            f"cannot open {target} collection {collection_name!r}: {exc}"
        ) from exc
    chunks, exact_index = _read_exact_index(store, validation_error=IndexEvaluationError)
    embedding_provider = BgeM3EmbeddingProvider(
        settings.embedding_model, device=settings.model_device
    )
    _validate_exact_dimension(exact_index, embedding_provider, validation_error=IndexEvaluationError)
    identity = EvidenceIndexIdentity(
        collection_name=collection_name,
        index_signature=pointer_before.index_signature,
        chunk_count=pointer_before.chunk_count,
        governance="lifecycle_active" if target == "active" else "lifecycle_staging",
    )
    snapshot = _IndexSnapshot(
        chunks=chunks,
        embedding_provider=embedding_provider,
        exact_index=exact_index,
        identity=identity,
    )
    retriever = _build_retriever_from_snapshot(settings, snapshot, trace_recorder=trace_recorder)
    context_builder = build_context_builder(settings)
    # 模型加载后再读一次指针：评测是长流程，读取期间发生过激活/切换就必须整体作废。
    pointer_after = _evaluation_pointer(settings, target)
    if pointer_after != pointer_before:
        raise IndexEvaluationError(
            f"{target} index pointer changed while its snapshot was being read; retry the run"
        )
    if len(chunks) != pointer_before.chunk_count:
        raise IndexEvaluationError(
            f"{target} snapshot has {len(chunks)} chunks but the pointer declares "
            f"{pointer_before.chunk_count}"
        )
    return EvaluationIndex(
        identity=identity,
        snapshot=snapshot,
        retriever=retriever,
        context_builder=context_builder,
    )


def _evaluation_agent_retriever(
    settings: Settings,
    index: EvaluationIndex,
    *,
    trace_recorder: TraceRecorder | None = None,
) -> DenseRerankerRetriever | ClauseBranchRetriever:
    """固定 Agent 对照沿用 clause_branch，在同一评测快照上装配检索器。

    自适应 Agent 不走这里：它与线上 ``build_agent_service`` 一样沿用
    ``Settings.multi_doc_strategy``，因此直接使用 ``index.retriever``。
    """

    if settings.multi_doc_strategy == "clause_branch":
        return index.retriever
    agent_settings = settings.model_copy(update={"multi_doc_strategy": "clause_branch"})
    return _build_retriever_from_snapshot(
        agent_settings,
        index.snapshot,
        trace_recorder=trace_recorder,
    )


@dataclass(frozen=True)
class EvaluationPaths:
    """同题对照的基线和候选，全部固定在同一个 EvaluationIndex 快照上。

    ``include_agents=False`` 时只保留索引检索组件，所有回答服务均为 None；
    未 opt-in 的检索评测不能构造 LLM 客户端。
    """

    index: EvaluationIndex
    rag: RAGService | None = None
    rag_generator: Generator | None = None
    fixed_agent: AgentService | None = None
    adaptive_agent: AdaptiveAgentService | None = None
    hybrid_agent: HybridAgentService | None = None


def build_evaluation_paths(
    settings: Settings,
    index: EvaluationIndex,
    *,
    include_agents: bool = True,
    selected_paths: Collection[str] = ("rag", "fixed_agent", "adaptive_agent"),
    trace_recorder: TraceRecorder | None = None,
) -> EvaluationPaths:
    """在同一个只读快照上装配普通 RAG、固定 Agent、自适应 Agent，复用现有服务。

    - 普通 RAG 用 ``Settings.multi_doc_strategy`` 对应的检索器（``index.retriever``）；
    - 固定 Agent 对照保留 clause_branch，并在同一快照上装配检索器；
    - 自适应 Agent 与线上 ``build_agent_service`` 共用服务装配和检索策略。

    各路径共用同一份索引快照。``selected_paths`` 限定本轮需要装配的回答服务，
    ``include_agents=False`` 用于纯检索评测，此时只保留索引和检索组件。
    """

    # 纯检索对照在此提前返回，避免即使不调用 answer 也初始化付费 LLM 客户端。
    if not include_agents:
        return EvaluationPaths(index=index)
    rag_generator = None
    rag = None
    fixed_agent = None
    adaptive_agent = None
    hybrid_agent = None
    if "rag" in selected_paths:
        rag_generator = build_generator(settings, trace_recorder=trace_recorder)
        rag = RAGService(index.retriever, index.context_builder, rag_generator, trace_recorder=trace_recorder)
    if "fixed_agent" in selected_paths:
        agent_retriever = _evaluation_agent_retriever(settings, index, trace_recorder=trace_recorder)
        fixed_agent = _assemble_agent_service(
            settings, agent_retriever, index.context_builder, trace_recorder=trace_recorder,
        )
    if {"adaptive_agent", "hybrid_agent"}.intersection(selected_paths):
        tools = EvidenceTools(index.retriever, EvidenceNavigator(index.chunks, index.identity))
        if "adaptive_agent" in selected_paths:
            adaptive_agent = _assemble_adaptive_service(
                settings, tools, index.context_builder, trace_recorder=trace_recorder,
            )
        if "hybrid_agent" in selected_paths:
            # 候选由显式评测路径加载，生产问答入口仍按其原有配置装配。
            from evidenceflow.agent.experimental import AdaptiveAgentService as HybridAgent

            recovery_search = None
            if isinstance(index.retriever, ClauseBranchRetriever):
                recovery_search = partial(tools.search, retrieve=index.retriever.retrieve_for_recovery)
            hybrid_agent = HybridAgent(
                tools, index.context_builder, build_generator(settings, trace_recorder=trace_recorder),
                decision_llm=build_llm_client(settings), trace_recorder=trace_recorder,
                allow_direct_answer=True, recovery_search=recovery_search, **_adaptive_budget_options(settings),
            )
    return EvaluationPaths(
        index=index,
        rag=rag,
        rag_generator=rag_generator,
        fixed_agent=fixed_agent,
        adaptive_agent=adaptive_agent,
        hybrid_agent=hybrid_agent,
    )


def build_rag_service(
    settings: Settings,
    *,
    retriever: RetrieverProtocol | None = None,
    generator: GeneratorProtocol | None = None,
    trace_recorder: TraceRecorder | None = None,
) -> RAGService:
    """Create the single official Exact Dense Top50 -> Learned Reranker -> Top8 pipeline (Settings defaults).

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


def build_evidence_navigator(settings: Settings) -> EvidenceNavigator:
    """Read chunks from the serving collection for index-bound evidence navigation.

    This separate read-only entry point leaves the existing ask and agent assembly unchanged.
    It never creates a collection or loads a model.

    It is a *standalone* navigator: it reads the collection itself, so a caller must not combine its
    result with a retriever built by ``build_retriever`` (two reads can straddle an activation).
    Use ``build_evidence_tools`` when ``search`` and ``open`` must share one snapshot
    (PROJECT_SPEC §10.2).
    """

    collection_name = resolve_active_collection(settings)
    if _lifecycle_governed(settings):
        # Bind the read to the active pointer before opening Chroma; staging and previous
        # collections must never be substituted for the serving index.
        active = _load_lifecycle_state(settings).active
        if collection_name != active.collection_name:
            raise ActiveCollectionError("active index changed while building evidence navigation")
        store = _open_active_store(settings, collection_name)
        identity = EvidenceIndexIdentity(
            collection_name=collection_name,
            index_signature=active.index_signature,
            chunk_count=active.chunk_count,
            governance="lifecycle_active",
        )
    else:
        metadata = read_collection_metadata(settings.vector_db_path, collection_name)
        if metadata is None or metadata.get("lifecycle") not in {None, "ACTIVE"}:
            raise ActiveCollectionError(
                f"Collection {collection_name!r} is missing or is not a serving index"
            )
        # Even without a lifecycle state file, navigation is read-only and does not create an
        # empty collection when the configured name is wrong.
        store = ChromaVectorStore(settings.vector_db_path, collection_name, create_if_missing=False)
        identity = EvidenceIndexIdentity(
            collection_name=collection_name,
            index_signature=None,
            chunk_count=store.count,
            governance="settings_fallback",
        )

    return EvidenceNavigator(list(store.iter_all_chunks()), identity)


def build_evidence_tools(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> EvidenceTools:
    """Assemble the read-only ``search`` / ``open`` tools over ONE read of the serving collection.

    ``search`` and ``open`` must answer from the same index, otherwise a chunk found by ``search``
    could be unopenable, or ``open`` could return a neighbour from a different index that happens to
    share the collection name. So this function performs a single
    ``iter_all_chunks_with_embeddings()`` read and builds *both* the production retriever and the
    navigator from that one snapshot: the Exact index, BM25 and the navigator all see the same
    chunks. The collection is opened once; the lifecycle pointer is read before and after that single
    read only to prove the snapshot still belongs to the claimed index (a pointer that moved in
    between is refused rather than mislabelled).

    Unlike ``build_evidence_navigator`` this loads the embedding and reranker models, because
    ``search`` is production retrieval. Like it, it never creates a collection, never writes, and
    never falls back to another document or index. The Agent uses it through ``EvidenceTools``.
    """

    snapshot = _read_index_snapshot(settings, read_only=True, bind_identity=True)
    identity = snapshot.identity
    if identity is None:  # pragma: no cover - bind_identity=True always sets it
        raise ActiveCollectionError("evidence tools require an index identity")
    retriever = _build_retriever_from_snapshot(
        settings, snapshot, trace_recorder=trace_recorder
    )
    navigator = EvidenceNavigator(snapshot.chunks, identity)
    return EvidenceTools(retriever, navigator)


def _assemble_agent_service(
    settings: Settings,
    retriever: DenseRerankerRetriever | ClauseBranchRetriever,
    context_builder: ContextBuilder,
    *,
    trace_recorder: TraceRecorder | None = None,
) -> AgentService:
    """用评测快照的 clause_branch 检索器装配固定 Agent 对照组。"""

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


def build_agent_service(
    settings: Settings, *, trace_recorder: TraceRecorder | None = None
) -> AdaptiveAgentService:
    """为 CLI、WebUI 和 HTTP API 装配证据 Agent，检索与邻块阅读绑定同一份只读索引快照。"""

    tools = build_evidence_tools(settings, trace_recorder=trace_recorder)
    return _assemble_adaptive_service(
        settings, tools, build_context_builder(settings), trace_recorder=trace_recorder,
    )


def _assemble_adaptive_service(
    settings: Settings,
    tools: EvidenceTools,
    context_builder: ContextBuilder,
    *,
    trace_recorder: TraceRecorder | None = None,
) -> AdaptiveAgentService:
    """生产入口和同题评测共用预算映射与生成组件装配。"""

    generator = build_generator(settings, trace_recorder=trace_recorder)
    decision_llm = build_llm_client(settings)
    return AdaptiveAgentService(
        tools,
        context_builder,
        generator,
        decision_llm=decision_llm,
        trace_recorder=trace_recorder,
        targeted_evidence=settings.agent_adaptive_targeted_evidence,
        **_adaptive_budget_options(settings),
    )


def _adaptive_budget_options(settings: Settings) -> dict[str, Any]:
    """同题比较沿用线上额度，避免候选通过额外调用获得优势。"""
    pair_limit = settings.reranker_input_k
    if settings.multi_doc_strategy == "clause_branch":
        pair_limit = max(pair_limit, settings.clause_branch_pair_budget)
    return {
        "llm_calls_max": settings.agent_adaptive_llm_calls_max,
        "decision_calls_max": settings.agent_adaptive_decision_calls_max,
        "searches_max": settings.agent_adaptive_searches_max,
        "opens_max": settings.agent_adaptive_opens_max,
        "rerank_pairs_max": settings.agent_adaptive_rerank_pairs_max,
        "search_pair_upper_bound": pair_limit,
        "deadline_seconds": settings.agent_adaptive_deadline_seconds,
        "clock": time.monotonic,
    }


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
