"""CLI 和 WebUI 共用的候选索引构建、结构校验与质量门禁发布流程。"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

from evidenceflow.config import Settings
from evidenceflow.domain.index_quality import QualityGateResult, verify_quality_report
from evidenceflow.knowledge import (
    BuildJobStore,
    ChromaFlagWriter,
    IndexLifecycle,
    IndexPointer,
    LifecycleState,
    PipelineIdentity,
    SourceRecord,
    StagingBuildCoordinator,
    build_corpus_signature,
    build_index_signature,
    build_manifest,
    build_staging_collection_name,
    current_pipeline_from_settings,
    load_manifest,
    manifest_sources_from_documents,
    save_manifest,
    validate_collection,
)
from evidenceflow.knowledge.job import BuildJobStatus
from evidenceflow.knowledge.lifecycle import CollectionReader

# BGE-M3 Dense vector dimension; the same value as runtime.EMBEDDING_DIMENSION (runtime is not
# imported here, to avoid pulling in the model dependencies).
EMBEDDING_DIMENSION = 1024
SMOKE_QUERY = "PoinTr point cloud representation for Transformer"
# Build stages may only move forward; ingestion loops per file so the same stage recurs, and it is
# reported only on first entry.
_STAGE_ORDER: tuple[BuildJobStatus, ...] = ("PARSING", "CHUNKING", "EMBEDDING", "INDEXING")


class StagingBuildRefused(RuntimeError):
    """构建或激活的前置状态不满足，当前索引保持服务。"""


class ProtectedCollectionError(RuntimeError):
    """Raised when something tries to delete the active or previous collection; those two are the
    indexes that serving and rollback depend on.

    Scenarios that can be detected up front (the generated staging name collides with them) are
    refused before the build via ``StagingBuildRefused``; this exception still being raised after the
    build means the state file changed during the build, and maps to exit code 3.
    """


@contextmanager
def index_update(state_path: Path) -> Iterator[None]:
    """串行化 CLI 与 WebUI 对同一生命周期状态的构建、发布和清理。"""

    state_path = state_path.resolve()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(state_path) + ".write.lock", timeout=0)
    try:
        with lock:
            yield
    except Timeout:
        raise StagingBuildRefused("another index update is in progress") from None


class StageReporter:
    """Report the "first entry into a stage" facts from ingestion one-way to the Coordinator."""

    def __init__(self, report: Callable[[BuildJobStatus], None]) -> None:
        self._report = report
        self._reached = -1

    def reach(self, stage: BuildJobStatus) -> None:
        """Advance to stage; skipped intermediate stages are reported one by one, and repeated or backward events are ignored."""

        target = _STAGE_ORDER.index(stage)
        while self._reached < target:
            self._reached += 1
            self._report(_STAGE_ORDER[self._reached])


class Observed:
    """Transparent proxy: the configured hooks fire when the named method is first called, and every other attribute or method is forwarded unchanged."""

    def __init__(self, inner: Any, hooks: dict[str, Callable[[], None]]) -> None:
        self._inner = inner
        self._hooks = hooks

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._inner, name)
        hook = self._hooks.get(name)
        if hook is None or not callable(attribute):
            return attribute

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            hook()
            return attribute(*args, **kwargs)

        return wrapped


@dataclass
class StagingPorts:
    """Every contact point between the script and the outside world (disk, Chroma, models); tests replace the whole set with Fakes."""

    scan_sources: Callable[[], list[SourceRecord]]
    list_collections: Callable[[], set[str]]
    active_chunk_ids: Callable[[str], list[str]]
    build: Callable[[str, Callable[[BuildJobStatus], None]], CollectionReader]
    smoke_embedding: Callable[[], list[float]]
    delete_collection: Callable[[str], None]
    set_collection_lifecycle: Callable[[str, str], None]
    # 只读打开一个已存在的 collection：`--activate-staging` 用它重跑独立的结构校验。
    open_collection: Callable[[str], CollectionReader] | None = None


def _protected_names(state: LifecycleState) -> set[str]:
    """Active and previous are the collections serving and rollback depend on, so any deletion or reuse must avoid them."""

    names = {state.active.collection_name}
    if state.previous is not None:
        names.add(state.previous.collection_name)
    return names


def discard_staging_collection(
    lifecycle: IndexLifecycle,
    delete_collection: Callable[[str], None],
    list_collections: Callable[[], set[str]],
) -> LifecycleState:
    """Delete the registered staging collection, then clear the staging pointer; safe to repeat (idempotent).

    The target is confirmed not to be active or previous first: deleting either of those would break
    serving or rollback. The order is collection first, pointer second, so a failed deletion leaves the
    pointer unchanged and the state still matches reality.
    If a previous run already deleted the collection but failed to clear the pointer, the collection no
    longer exists on the next run, so only the pointer is cleared and the call succeeds; otherwise that
    half-finished state would make every later build fail with "a staging pointer is already registered".
    """

    state = lifecycle.load_state()
    if state.staging is None:
        raise StagingBuildRefused("no staging pointer is registered; nothing to discard")
    name = state.staging.collection_name
    if name in _protected_names(state):
        raise ProtectedCollectionError(
            f"refusing to delete collection {name!r}: it is the active or previous index"
        )
    # When the collection no longer exists (last deletion succeeded, clearing the pointer failed),
    # skip
    # the deletion and only finish the pointer cleanup.
    if name in list_collections():
        delete_collection(name)
    return lifecycle.discard_staging()


def mark_collection_lifecycle(client: Any, collection_name: str, lifecycle: str) -> None:
    """Rewrite the lifecycle flag in the collection metadata through the Chroma API, keeping every other
    business metadata field unchanged.

    Thin adapter over ``evidenceflow.knowledge.lifecycle.make_chroma_flag_writer`` so the metadata
    rewrite lives in one place: ``IndexLifecycle`` owns the flags for every transition (staging,
    activate, rollback), and this helper only exists for the legacy ``--activate`` path of this
    script and its tests.
    """

    from evidenceflow.knowledge.lifecycle import ChromaFlagWriter

    ChromaFlagWriter(client)(collection_name, lifecycle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Match the project's write_json: UTF-8, non-ASCII characters left unescaped, keys sorted."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative(path: Path, repo_root: Path | None) -> str:
    """Use a repo-relative path for manifest_path in the state file; paths outside the repo stay as they are."""

    try:
        return path.resolve().relative_to((repo_root or Path.cwd()).resolve()).as_posix()
    except ValueError:
        return str(path)


def run_staging_build(
    settings: Settings,
    ports: StagingPorts,
    *,
    job_dir: Path,
    expected_chunk_ids_from_active: bool,
    activate: bool,
    discard_after_validate: bool,
    quality_report_path: Path | None = None,
    repo_root: Path | None = None,
    expected_active: IndexPointer | None = None,
    manifest_name: str | None = None,
    activate_after_write: bool = False,
) -> tuple[int, dict[str, Any]]:
    """构建候选索引并将结果写入 job_dir，返回退出码及摘要。

    ``activate=True`` 必须先通过一份有效质量报告：结构校验（``validate_collection``）只证明
    collection 可读、数量与身份一致，不证明检索/Context 质量；因此结构校验通过后还会复核
    ``quality_report_path`` 提到的 active/staging ``index_signature`` 是否与当前指针一致、结论
    是否为 ``passed``。未提供报告或报告不合格时拒绝激活，staging collection 与指针原样保留。
    ``quality_report_path`` 在给了 ``--activate`` 时必填；不给 ``--activate`` 时忽略。
    activate_after_write 是本地 WebUI 的增量入库模式，以同一任务写入的完整快照直接提交。
    """

    if (activate or activate_after_write) and discard_after_validate:
        raise StagingBuildRefused("--activate and --discard-after-validate are mutually exclusive")
    if activate_after_write and (activate or expected_active is None):
        raise StagingBuildRefused("incremental publication requires its base snapshot")
    if activate and quality_report_path is None:
        raise StagingBuildRefused("--activate requires a valid --quality-report; structural validation alone is not a quality gate")
    # The state file is the single source of truth; fail here when the knowledge-base identity cannot
    # be read instead of guessing it.
    state = LifecycleState.model_validate_json(Path(settings.index_state_path).read_text(encoding="utf-8"))
    if expected_active is not None and state.active != expected_active:
        raise StagingBuildRefused("the active index changed before this build")
    # The lifecycle owns the Collection lifecycle flags for every transition, so activate() flips
    # the incoming index to ACTIVE and the outgoing one to PREVIOUS itself; the script no longer
    # has to remember a separate mark-collection-active call before switching the pointer.
    lifecycle = IndexLifecycle(
        settings.index_state_path,
        state.knowledge_base_id,
        flag_writer=ports.set_collection_lifecycle,
    )
    if state.staging is not None:
        raise StagingBuildRefused(
            f"a staging pointer is already registered ({state.staging.collection_name!r}); "
            "inspect it and discard it before starting another build"
        )

    sources = ports.scan_sources()
    pipeline: PipelineIdentity = current_pipeline_from_settings(settings)
    if activate_after_write:
        # 新 PDF 使用当前解析器，旧片段绑定其真实来源，即使新旧解析器版本不同也可追溯。
        pipeline = PipelineIdentity.model_validate({
            **pipeline.model_dump(),
            "chunker_config": {
                **pipeline.chunker_config,
                "reused_from_index_signature": state.active.index_signature,
            },
            "chunker_config_hash": None,
        })
    index_signature = build_index_signature(build_corpus_signature(sources), pipeline)
    protected = _protected_names(state)
    # active/previous are passed in as reserved so the generator steers around them (for example
    # when
    # rebuilding with the same signature after a rollback, the preferred name is exactly previous).
    staging_name = build_staging_collection_name(
        state.active.collection_name, index_signature, reserved=protected
    )
    # Defense in depth: whatever the naming rule, the staging name must never equal the collection
    # that
    # serving or rollback depends on, otherwise the build would write into the ACTIVE/PREVIOUS
    # index and
    # discard would even delete it.
    # A3.2 note: the current generator already avoids active/previous through reserved, so this
    # check
    # cannot fire with the current implementation; it is kept on purpose as a backstop against
    # future
    # generator regressions (covered in tests by monkeypatching the generator).
    if staging_name in protected:
        raise StagingBuildRefused(
            f"generated staging name {staging_name!r} equals the active or previous collection; "
            "refusing to build. Do not delete or modify that collection: it is a live or rollback "
            "index. This indicates a staging naming bug"
        )
    if staging_name in ports.list_collections():
        raise StagingBuildRefused(
            f"collection {staging_name!r} already exists and is neither active nor previous "
            "(likely left over from an earlier build); inspect it before deciding what to do. "
            "This script never reuses or deletes an existing collection"
        )
    expected_ids = (
        ports.active_chunk_ids(state.active.collection_name) if expected_chunk_ids_from_active else None
    )
    # The staging corpus may differ from the serving one, so its size is not known before the build.
    # These counts are provisional (they only feed the manifest's shape) and are replaced by the
    # measured values after validation; inheriting active's chunk_count here used to make every
    # corpus growth fail validation, because the new index could never equal the old one.
    manifest = build_manifest(
        knowledge_base_id=state.knowledge_base_id,
        sources=sources,
        pipeline=pipeline,
        collection_name=staging_name,
        document_count=len(sources),
        chunk_count=0,
        status="STAGING",
        parent_index_signature=state.active.index_signature,
    )

    job_store = BuildJobStore(job_dir / "staging_build_job.json")
    coordinator = StagingBuildCoordinator(lifecycle, job_store)
    job, report = coordinator.build(
        manifest,
        ports.build,
        expected_chunk_ids=expected_ids,
        smoke_query_embedding=ports.smoke_embedding(),
        expected_embedding_dimension=EMBEDDING_DIMENSION,
        # A same-corpus rebuild must reproduce the previous size exactly; a corpus change is not
        # expected to, and expected_chunk_ids already pins the identity in that case.
        expected_chunk_count=len(expected_ids) if expected_ids is not None else None,
    )
    summary: dict[str, Any] = {
        "staging_collection": staging_name,
        "index_signature": index_signature,
        "active_collection": state.active.collection_name,
        "expected_chunk_ids_from_active": expected_chunk_ids_from_active,
        "job": json.loads(job.model_dump_json()),
        "validation": report.model_dump(mode="json") if report is not None else None,
        "actions": [],
    }
    passed = job.status == "READY" and report is not None and report.passed
    if not passed:
        # On failure keep the staging collection and the pointer: they are diagnostic evidence and
        # the
        # user decides how to handle them.
        summary["outcome"] = "FAILED_KEPT_STAGING"
        _write_json(job_dir / "staging_build_summary.json", summary)
        return 1, summary

    # Persist the measured counts on the staging pointer and write its manifest before returning.
    # ``prepare_staging`` only knew provisional counts, but read-only staging evaluation
    # (``build_evaluation_index``) refuses a snapshot whose size disagrees with the pointer, and a
    # quality report must bind the true chunk_count. Active/previous pointers stay untouched.
    # --discard-after-validate skips this: the collection is about to be deleted, and a manifest for
    # it would only be a stale file on disk.
    measured = coordinator.measured_manifest or manifest
    manifest_path = job_dir / (manifest_name or f"{staging_name}.manifest.json")
    if not discard_after_validate:
        save_manifest(manifest_path, measured)
        lifecycle.record_staging_measurement(
            measured, manifest_path=_repo_relative(manifest_path, repo_root)
        )
        summary["measured_chunk_count"] = report.actual_chunk_count
        summary["staging_manifest"] = _repo_relative(manifest_path, repo_root)

    if activate:
        # 结构校验（validate_collection）只保证 collection 可读、身份与数量一致；
        # 它不衡量检索与 Context 质量。因此激活前必须复核一份绑定了真实索引签名的质量报告。
        gate = verify_quality_report_for_activation(
            quality_report_path,
            active_pointer=(
                state.active.collection_name,
                state.active.index_signature,
                state.active.chunk_count,
            ),
            staging_pointer=(
                staging_name,
                index_signature,
                report.actual_chunk_count,
            ),
        )
        summary["quality_gate"] = gate.model_dump(mode="json")
        if not gate.passed:
            # 拒绝激活时保留 staging collection 与指针，方便修报告后重试；不动 active。
            summary["outcome"] = "QUALITY_GATE_REFUSED"
            summary["error"] = "; ".join(gate.errors)
            _write_json(job_dir / "staging_build_summary.json", summary)
            return 4, summary
    if activate or activate_after_write:
        # 使用本次构建的实测结果提交，只切一次指针，旧索引成为回滚版本。
        activated = measured
        save_manifest(manifest_path, activated.with_index_state(status="ACTIVE"))
        # activate() writes the Collection flags itself (staging -> ACTIVE, old active -> PREVIOUS)
        # before switching the pointer, so the pointer never names a collection that runtime rejects.
        lifecycle.activate(activated, report, manifest_path=_repo_relative(manifest_path, repo_root))
        job_store.save(job.transition("ACTIVATED"))
        summary["actions"].append("activated")
        summary["outcome"] = "ACTIVATED"
    elif discard_after_validate:
        try:
            discard_staging_collection(lifecycle, ports.delete_collection, ports.list_collections)
        except ProtectedCollectionError as exc:
            # The staging name was confirmed not to be active/previous before the build; reaching
            # this
            # point means the state file changed during the build. The build has already happened,
            # so
            # "2 = refused to start" no longer applies: use exit code 3 and keep staging and the
            # pointer.
            summary["outcome"] = "VALIDATED_DISCARD_REFUSED"
            summary["error"] = str(exc)
            _write_json(job_dir / "staging_build_summary.json", summary)
            return 3, summary
        summary["actions"].append("staging_collection_deleted")
        summary["actions"].append("staging_pointer_cleared")
        summary["outcome"] = "VALIDATED_AND_DISCARDED"
    else:
        summary["outcome"] = "VALIDATED_STAGING_KEPT"
    _write_json(job_dir / "staging_build_summary.json", summary)
    return 0, summary


def _open_collection_port(settings: Settings):
    """懒加载只读 collection 端口，避免脚本导入时就加载 Chroma 依赖。"""

    from evidenceflow.storage import ChromaVectorStore

    return lambda name: ChromaVectorStore(
        settings.vector_db_path, name, create_if_missing=False
    )


def activate_validated_staging(
    settings: Settings,
    ports: StagingPorts,
    *,
    job_dir: Path,
    quality_report_path: Path | None,
    repo_root: Path | None = None,
    expected_staging: IndexPointer | None = None,
    require_quality_report: bool = True,
) -> tuple[int, dict[str, Any]]:
    """`--activate-staging`：激活已经构建并通过评测的 staging；不重建、不改 staging 内容。

    顺序固定为：读指针 → 独立重跑结构校验（独立于构建时的快照）→ 复核质量报告身份与结论 →
    才调用 lifecycle.activate。任何一步失败都返回非 0，且 active 指针保持不变。
    这正是 staging → 评测 → 人工复核 → 激活的正常流程：评测脚本只读 staging，
    激活入口单独存在，不需要在评测期间临时切换 active。

    CLI 缺少质量报告时返回退出码 4。工作台的异常恢复可关闭报告复核，
    仅适用于本次增量构建生成、且仍与当前正式索引一致的候选。
    """

    state = LifecycleState.model_validate_json(Path(settings.index_state_path).read_text(encoding="utf-8"))
    if state.staging is None:
        raise StagingBuildRefused("no staging pointer is registered; nothing to activate")
    staging = state.staging
    if expected_staging is not None and staging != expected_staging:
        raise StagingBuildRefused("the staging index changed before this activation")
    lifecycle = IndexLifecycle(
        settings.index_state_path,
        state.knowledge_base_id,
        flag_writer=ports.set_collection_lifecycle,
    )
    open_collection = ports.open_collection or _open_collection_port(settings)
    store = open_collection(staging.collection_name)
    manifest_path = Path(staging.manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = (repo_root or Path.cwd()) / manifest_path
    if not manifest_path.is_file():
        raise StagingBuildRefused(
            f"staging manifest {manifest_path} is missing; rebuild staging to regenerate it"
        )
    staging_manifest = load_manifest(manifest_path).with_index_state(
        collection_name=staging.collection_name, status="STAGING"
    )
    if staging_manifest.index.index_signature != staging.index_signature:
        raise StagingBuildRefused(
            "staging manifest index_signature does not match the staging pointer"
        )
    if not require_quality_report and (
        expected_staging is None
        or staging_manifest.index.parent_index_signature != state.active.index_signature
        or staging_manifest.pipeline.chunker_config.get("reused_from_index_signature") != state.active.index_signature
    ):
        raise StagingBuildRefused("候选索引需要按已有发布流程处理，或放弃后重新上传 PDF。")
    # validate_collection 会用 manifest 的 document_count 校验实际文档数，所以用指针里的实测值
    # 覆盖 manifest 上的同一字段（两者在正常流程里本就相等）。
    staging_manifest = staging_manifest.with_index_state(
        document_count=staging.document_count, chunk_count=staging.chunk_count
    )
    # 结构校验独立于构建时快照重跑：文件被替换或 collection 被改动都会在这里暴露。
    # ``staging.chunk_count`` 是构建后写回的实测值，用它做精确数量校验。
    report = validate_collection(
        staging_manifest,
        store,
        smoke_query_embedding=ports.smoke_embedding(),
        expected_embedding_dimension=EMBEDDING_DIMENSION,
        expected_chunk_count=staging.chunk_count,
    )
    summary: dict[str, Any] = {
        "outcome": "ACTIVATE_STAGING_VALIDATION_FAILED" if not report.passed else "ACTIVATE_STAGING_READY",
        "staging_collection": staging.collection_name,
        "index_signature": staging.index_signature,
        "active_collection": state.active.collection_name,
        "validation": report.model_dump(mode="json"),
        "actions": [],
    }
    if not report.passed:
        _write_json(job_dir / "staging_activation_summary.json", summary)
        return 1, summary
    gate = verify_quality_report_for_activation(
        quality_report_path,
        active_pointer=(
            state.active.collection_name,
            state.active.index_signature,
            state.active.chunk_count,
        ),
        staging_pointer=(
            staging.collection_name,
            staging.index_signature,
            staging.chunk_count,
        ),
    ) if require_quality_report else None
    if gate is not None:
        summary["quality_gate"] = gate.model_dump(mode="json")
    if gate is not None and not gate.passed:
        summary["outcome"] = "QUALITY_GATE_REFUSED"
        summary["error"] = "; ".join(gate.errors)
        _write_json(job_dir / "staging_activation_summary.json", summary)
        return 4, summary
    activated = staging_manifest.with_index_state(
        document_count=report.actual_document_count, chunk_count=report.actual_chunk_count
    )
    save_manifest(manifest_path, activated.with_index_state(status="ACTIVE"))
    lifecycle.activate(activated, report, manifest_path=_repo_relative(manifest_path, repo_root))
    summary["actions"].append("activated")
    summary["outcome"] = "ACTIVATED"
    _write_json(job_dir / "staging_activation_summary.json", summary)
    return 0, summary


def verify_quality_report_for_activation(
    path: Path | None,
    *,
    active_pointer: tuple[str, str, int],
    staging_pointer: tuple[str, str, int],
) -> QualityGateResult:
    """读取并复核质量报告；缺文件、坏 JSON、身份不符、结论非 passed 都算未通过门禁。

    门禁本身不发明阈值：它只要求报告由 ``scripts/e2e_index_quality.py`` 依据既有 gold 映射生成，
    并且明确写出 ``status=passed``、无 blocker、无 Context 覆盖下降。
    """

    if path is None:
        return QualityGateResult(
            passed=False,
            errors=("no --quality-report was provided; structural validation alone is not a quality gate",),
        )
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return QualityGateResult(
            passed=False,
            errors=(f"quality report {path} does not exist",),
        )
    except Exception as exc:  # noqa: BLE001 - 读不了报告就等同于没有报告
        return QualityGateResult(
            passed=False,
            errors=(f"cannot read quality report {path}: {type(exc).__name__}: {exc}",),
        )
    if not isinstance(payload, dict):
        return QualityGateResult(
            passed=False,
            errors=(f"quality report {path} is not a JSON object",),
        )
    return verify_quality_report(
        payload, active_pointer=active_pointer, staging_pointer=staging_pointer
    )


def make_real_ports(
    settings: Settings, *, source_files: Sequence[Path] | None = None,
    on_file: Callable[[int, int], None] | None = None,
    copy_from: IndexPointer | None = None,
) -> StagingPorts:
    """装配构建端口。copy_from 的向量原样拷贝，只对 source_files 指定的新文件运行摄取。"""

    import chromadb

    from evidenceflow.embedding import BgeM3EmbeddingProvider
    from evidenceflow.ingestion import BgeM3Tokenizer, IngestionService
    from evidenceflow.ingestion.loader import load_document
    from evidenceflow.ingestion.service import discover_source_files
    from evidenceflow.storage import ChromaVectorStore

    root = Path(settings.source_docs_path)
    client = chromadb.PersistentClient(path=str(settings.vector_db_path))
    provider = None

    def embedding_provider() -> BgeM3EmbeddingProvider:
        """扫描文档成功后再加载模型，构建与 smoke 共用此实例。"""
        nonlocal provider
        if provider is None:
            provider = BgeM3EmbeddingProvider(settings.embedding_model, device=settings.model_device)
        return provider

    def scan_sources() -> list[SourceRecord]:
        files = source_files if source_files is not None else discover_source_files(root, root)
        documents = [load_document(path, root) for path in files]
        return manifest_sources_from_documents(documents)

    def active_chunk_ids(collection_name: str) -> list[str]:
        # Open active read-only: ChromaVectorStore does not modify the metadata of an existing
        # collection.
        return [chunk.chunk_id for chunk in ChromaVectorStore(settings.vector_db_path, collection_name).iter_all_chunks()]

    def build(collection_name: str, report_stage: Callable[[BuildJobStatus], None]) -> ChromaVectorStore:
        reporter = StageReporter(report_stage)
        reporter.reach("PARSING")
        store = ChromaVectorStore(
            settings.vector_db_path,
            collection_name,
            # Do not write the legacy "hnsw:space" key: nothing in the source reads it and cosine
            # is
            # guaranteed by ChromaVectorStore's configuration. (Collections carrying that key can
            # still be
            # rewritten for lifecycle, see mark_collection_lifecycle.)
            collection_metadata={
                "chunk_strategy": settings.chunk_strategy,
                "storage_contract": "domain_chunk_metadata_v2",
                "lifecycle": "STAGING",
            },
        )
        if copy_from is not None:
            original = ChromaVectorStore(settings.vector_db_path, copy_from.collection_name, create_if_missing=False)
            rows = original.iter_all_chunks_with_embeddings()
            copied = 0
            while batch := list(islice(rows, 256)):
                chunks, vectors = zip(*batch)
                store.add_chunks(chunks, vectors)
                copied += len(batch)
            if copied != copy_from.chunk_count:
                raise RuntimeError("active chunk count changed while copying the snapshot")
        service = IngestionService(
            root,
            Observed(BgeM3Tokenizer(settings.embedding_model), {"encode": lambda: reporter.reach("CHUNKING")}),
            Observed(embedding_provider(), {"embed_documents": lambda: reporter.reach("EMBEDDING")}),
            Observed(store, {"add_chunks": lambda: reporter.reach("INDEXING")}),
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            chunk_strategy=settings.chunk_strategy,
        )
        if source_files is None:
            summaries = [service.ingest()]
        else:
            summaries = []
            for offset, path in enumerate(source_files, 1):
                summaries.append(service.ingest(path))
                if on_file is not None:
                    on_file(offset, len(source_files))
        files_discovered = sum(summary.files_discovered for summary in summaries)
        failures = [
            f"{item.source_path}: {item.failed_stage}: {item.error_message}"
            for summary in summaries for item in summary.results if item.status == "failed"
        ]
        if not files_discovered or failures:
            raise RuntimeError(f"staging ingestion incomplete (files={files_discovered}): {failures}")
        reporter.reach("INDEXING")
        return store

    return StagingPorts(
        scan_sources=scan_sources,
        list_collections=lambda: {item.name for item in client.list_collections()},
        active_chunk_ids=active_chunk_ids,
        build=build,
        smoke_embedding=lambda: embedding_provider().embed_query(SMOKE_QUERY),
        delete_collection=lambda name: client.delete_collection(name),
        set_collection_lifecycle=lambda name, status, *, missing_ok=False: ChromaFlagWriter(client)(
            name, status, missing_ok=missing_ok
        ),
        # 只读打开；collection 不存在时直接失败，不留下空集合。
        open_collection=lambda name: ChromaVectorStore(
            settings.vector_db_path, name, create_if_missing=False
        ),
    )
