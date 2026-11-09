"""Phase 15A staging validation, active pointer, activation and rollback."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator

from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.knowledge.job import BuildJob, BuildJobStatus, BuildJobStore, BuildJobTracker
from evidenceflow.knowledge.manifest import KnowledgeBaseManifest

LifecyclePointerStatus = Literal["ACTIVE", "STAGING", "PREVIOUS"]


class ActiveCollectionError(RuntimeError):
    """Resolving the serving collection or verifying its identity failed: configuration conflict, corrupt state file or mismatching lifecycle flag."""


class ActiveIndexProtectedError(RuntimeError):
    """The target collection is the index the lifecycle active pointer references; ingesting into it directly is forbidden."""


class CollectionReader(Protocol):
    """The read-only Vector Store capabilities that lifecycle validation needs."""

    def iter_all_chunks(self) -> Iterator[DocumentChunk]:
        """Return every recoverable Chunk in the current Collection."""

        ...

    def search(self, query_embedding: Sequence[float], top_k: int) -> list[Any]:
        """Run one read-only Dense smoke query."""

        ...


class IndexPointer(BaseModel):
    """Immutable index identity summary of the active/staging/previous pointers."""

    knowledge_base_id: str = Field(min_length=1, description="所属知识库 ID。")
    collection_name: str = Field(min_length=1, description="绑定的 Chroma Collection 名称。")
    index_signature: str = Field(min_length=1, description="绑定的 indexed representation 身份。")
    manifest_path: str = Field(min_length=1, description="对应 manifest 的相对或绝对路径。")
    document_count: int = Field(ge=0, description="该索引的逻辑文档数量。")
    chunk_count: int = Field(ge=0, description="该索引的 Chunk 数量。")
    status: LifecyclePointerStatus = Field(description="ACTIVE、STAGING 或 PREVIOUS。")

    @classmethod
    def from_manifest(
        cls,
        manifest: KnowledgeBaseManifest,
        *,
        status: LifecyclePointerStatus | None = None,
        manifest_path: str | None = None,
    ) -> "IndexPointer":
        """Build a pointer from a manifest, defaulting to the manifest's index status."""

        pointer_status = status or manifest.index.status
        if pointer_status not in {"ACTIVE", "STAGING", "PREVIOUS"}:
            raise ValueError(f"manifest status cannot become lifecycle pointer: {pointer_status}")
        return cls(
            knowledge_base_id=manifest.knowledge_base_id,
            collection_name=manifest.index.collection_name,
            index_signature=manifest.index.index_signature,
            manifest_path=manifest_path or f"manifests/{manifest.index.index_signature}.json",
            document_count=manifest.index.document_count,
            chunk_count=manifest.index.chunk_count,
            status=pointer_status,
        )


class LifecycleState(BaseModel):
    """The single active/previous/staging state record of a single-machine knowledge base."""

    schema_version: str = Field(default="index-lifecycle-v1", min_length=1)
    knowledge_base_id: str = Field(min_length=1, description="状态所属知识库 ID。")
    active: IndexPointer = Field(description="当前唯一 active index。")
    previous: IndexPointer | None = Field(default=None, description="最近一次已知良好旧 active。")
    staging: IndexPointer | None = Field(default=None, description="尚未 activation 的新 index。")

    @model_validator(mode="after")
    def validate_pointer_ownership(self) -> "LifecycleState":
        """Ensure pointers never cross knowledge bases or carry the wrong lifecycle status."""

        if self.active.status != "ACTIVE":
            raise ValueError("active pointer must have status ACTIVE")
        pointers = [("active", self.active, "ACTIVE"), ("previous", self.previous, "PREVIOUS"), ("staging", self.staging, "STAGING")]
        for name, pointer, expected_status in pointers:
            if pointer is None:
                continue
            if pointer.knowledge_base_id != self.knowledge_base_id:
                raise ValueError(f"{name} pointer belongs to another knowledge base")
            if pointer.status != expected_status:
                raise ValueError(f"{name} pointer must have status {expected_status}")
        return self


class CollectionValidationReport(BaseModel):
    """Result of validating a staging Collection before activation."""

    passed: bool = Field(description="所有启用的 validation checks 是否通过。")
    manifest_document_count: int = Field(ge=0)
    actual_document_count: int = Field(ge=0)
    manifest_chunk_count: int = Field(ge=0)
    actual_chunk_count: int = Field(ge=0)
    missing_document_ids: tuple[str, ...] = Field(default=())
    unexpected_document_ids: tuple[str, ...] = Field(default=())
    duplicate_chunk_ids: tuple[str, ...] = Field(default=())
    missing_chunk_ids: tuple[str, ...] = Field(default=())
    unexpected_chunk_ids: tuple[str, ...] = Field(default=())
    missing_metadata: tuple[str, ...] = Field(default=())
    embedding_dimension: int | None = Field(default=None, ge=1)
    expected_embedding_dimension: int | None = Field(default=None, ge=1)
    embedding_dimension_consistent: bool = Field(default=False)
    collection_readable: bool = Field(default=False)
    dense_retrieval_smoke: bool = Field(default=False)
    errors: tuple[str, ...] = Field(default=())


class StagingBuildCoordinator:
    """Drive one synchronous staging build and write the real stages into the BuildJob."""

    def __init__(self, lifecycle: "IndexLifecycle", job_store: BuildJobStore) -> None:
        """Inject the lifecycle pointer and the Job persistence backend; creates no background task."""

        self.lifecycle = lifecycle
        self.job_store = job_store
        # The manifest with the build's measured counts filled in; None until a build validates.
        # Callers that activate after build() must pass this one, not their provisional manifest.
        self.measured_manifest: KnowledgeBaseManifest | None = None

    def build(
        self,
        manifest: KnowledgeBaseManifest,
        build: Callable[[str, Callable[[BuildJobStatus], None]], CollectionReader],
        *,
        expected_chunk_ids: Sequence[str] | None = None,
        smoke_query_embedding: Sequence[float] | None = None,
        expected_embedding_dimension: int | None = None,
        expected_chunk_count: int | None = None,
    ) -> tuple[BuildJob, CollectionValidationReport | None]:
        """Run a build on the new Collection under the callback-driven stage contract, then validate.

        `build` is responsible for reusing the existing IngestionService against the Collection name it
        is given, and it must invoke the stage callback when it actually enters
        PARSING/CHUNKING/EMBEDDING/INDEXING. The Coordinator does not duplicate the ingestion pipeline,
        and it never calls activation when validation fails.

        The build's measured size is written back into the returned manifest before reporting READY,
        so a caller that activates afterwards publishes the real counts rather than the provisional
        ones it registered (``expected_chunk_count`` is only for callers that truly know the size).
        """

        tracker = BuildJobTracker.create(
            knowledge_base_id=manifest.knowledge_base_id,
            source_corpus_signature=manifest.corpus_signature,
            target_index_signature=manifest.index.index_signature,
            target_collection_name=manifest.index.collection_name,
            store=self.job_store,
        )
        try:
            self.lifecycle.prepare_staging(manifest)
            tracker.advance("SCANNING")

            def report_stage(status: BuildJobStatus) -> None:
                """Advance the same persisted Job to the real stage reported by the builder."""

                if status not in {"PARSING", "CHUNKING", "EMBEDDING", "INDEXING"}:
                    raise ValueError(f"invalid staging build stage: {status}")
                tracker.advance(status)

            store = build(manifest.index.collection_name, report_stage)
            if tracker.job.status != "INDEXING":
                raise ValueError("staging builder returned before INDEXING stage")
            tracker.advance("VALIDATING")
            report = validate_collection(
                manifest,
                store,
                expected_chunk_ids=expected_chunk_ids,
                smoke_query_embedding=smoke_query_embedding,
                expected_embedding_dimension=expected_embedding_dimension,
                expected_chunk_count=expected_chunk_count,
            )
            if not report.passed:
                tracker.fail("VALIDATION_ERROR", "; ".join(report.errors) or "staging validation failed")
                return tracker.job, report
            self.measured_manifest = manifest.with_index_state(
                document_count=report.actual_document_count,
                chunk_count=report.actual_chunk_count,
            )
            tracker.advance(
                "READY",
                document_count=report.actual_document_count,
                chunk_count=report.actual_chunk_count,
            )
            return tracker.job, report
        except Exception as exc:
            # A failure only writes Job diagnostics and never touches the active pointer; staging
            # can be
            # cleaned up by the caller per the lifecycle policy.
            if tracker.job.status not in {"FAILED", "BLOCKED", "ACTIVATED"}:
                tracker.fail("BUILD_ERROR", f"{type(exc).__name__}: {exc}")
            return tracker.job, None


# Total length of a staging name: base prefix 24 + "-" + base short hash 8 + "__staging__" 11 +
# signature prefix 16 = 60, below Chroma's 63-character name limit.
_STAGING_BASE_PREFIX = 24
_STAGING_BASE_HASH = 8
_STAGING_SIGNATURE_PREFIX = 16
# Maximum number of candidates tried when avoiding reserved names; going beyond it means the
# input is anomalous, so raise instead of retrying forever.
_STAGING_NAME_ATTEMPTS = 8


def build_staging_collection_name(
    base_collection: str,
    index_signature: str,
    *,
    reserved: Collection[str] = (),
) -> str:
    """Generate a deterministic, readable Collection name for staging that does not rely on a random UUID.

    Name = truncated base prefix + short hash of the complete base + ``__staging__`` + signature
    prefix. The short hash is computed from the **complete** base: with only the truncated prefix,
    generating a name from a generated name would return that name itself (found in the A3 review),
    so a staging name rebuilt after activation would equal active. With the hash of the complete base
    included, generating from a generated result necessarily differs from itself (barring a 32-bit
    hash collision).

    Names in ``reserved`` (usually active and previous) are steered around: when rebuilding with the
    same base + signature after a rollback, the preferred name happens to equal previous (it was the
    last staging name), so the code falls back to candidates salted with an index in a fixed order,
    and the result is still decided only by the input. Raises ``ValueError`` when every candidate is taken.
    """

    if not base_collection.strip() or not index_signature.strip():
        raise ValueError("base_collection and index_signature must not be empty")
    # Keep only the characters Chroma allows and truncate to the name-length limit; leading and
    # trailing . and - are not allowed, so strip them.
    safe_base = (
        re.sub(r"[^A-Za-z0-9._-]", "-", base_collection)[:_STAGING_BASE_PREFIX].strip(".-")
        or "evidenceflow"
    )
    taken = set(reserved)
    for attempt in range(_STAGING_NAME_ATTEMPTS):
        # Candidate 0 hashes the base itself, making the first build's name a pure function of
        # (base, signature); later candidates append an index to the hash input only to avoid
        # reserved
        # names.
        salt = "" if attempt == 0 else f"#{attempt}"
        digest = hashlib.sha256(f"{base_collection}{salt}".encode("utf-8")).hexdigest()
        name = (
            f"{safe_base}-{digest[:_STAGING_BASE_HASH]}"
            f"__staging__{index_signature[:_STAGING_SIGNATURE_PREFIX]}"
        )
        if name not in taken:
            return name
    raise ValueError(
        f"cannot derive a staging collection name for {base_collection!r} that avoids {sorted(taken)}"
    )


def _read_embedding_dimension(store: Any) -> int | None:
    """Read the vector dimension from the concrete Store's read-only inspection capability."""

    value = getattr(store, "embedding_dimension", None)
    if callable(value):
        value = value()
    return int(value) if value is not None else None


def validate_collection(
    manifest: KnowledgeBaseManifest,
    store: CollectionReader,
    *,
    expected_chunk_ids: Sequence[str] | None = None,
    smoke_query_embedding: Sequence[float] | None = None,
    expected_embedding_dimension: int | None = None,
    required_metadata_keys: Sequence[str] = ("chunk_strategy", "element_ids"),
    expected_chunk_count: int | None = None,
) -> CollectionValidationReport:
    """Validate the staging document/Chunk/metadata/dimension/readability and the Dense smoke query.

    ``expected_chunk_count`` is the only way to demand an exact Chunk total. The manifest's own
    ``chunk_count`` is deliberately **not** enforced: a staging rebuild of a changed corpus cannot
    know its size in advance, so the build registers a provisional value and writes the measured one
    back afterwards. Enforcing the provisional number made any corpus growth fail validation (the
    new index could never equal the old one), which is the opposite of what a staging build is for.
    Callers that genuinely expect a fixed size -- a same-corpus rebuild -- pass the number here and
    get the equality check back, on top of ``expected_chunk_ids`` which already pins identity.
    """

    errors: list[str] = []
    try:
        chunks = list(store.iter_all_chunks())
        collection_readable = True
    except Exception as exc:
        chunks = []
        collection_readable = False
        errors.append(f"collection read failed: {type(exc).__name__}: {exc}")

    actual_document_ids = {chunk.document_id for chunk in chunks}
    expected_document_ids = {source.document_id for source in manifest.sources}
    chunk_ids = [chunk.chunk_id for chunk in chunks]
    duplicate_chunk_ids = tuple(sorted(chunk_id for chunk_id, count in Counter(chunk_ids).items() if count > 1))
    expected_ids = set(expected_chunk_ids) if expected_chunk_ids is not None else None
    missing_chunk_ids = tuple(sorted(expected_ids - set(chunk_ids))) if expected_ids is not None else ()
    unexpected_chunk_ids = tuple(sorted(set(chunk_ids) - expected_ids)) if expected_ids is not None else ()
    missing_metadata = tuple(
        f"{chunk.chunk_id}:{key}"
        for chunk in chunks
        for key in required_metadata_keys
        if key not in chunk.metadata or (key == "element_ids" and not chunk.metadata[key])
    )
    dimension = _read_embedding_dimension(store)
    dimension_consistent = (
        expected_embedding_dimension is None
        or dimension == expected_embedding_dimension
    ) and (expected_embedding_dimension is not None or dimension is not None)
    dense_smoke = False
    if smoke_query_embedding is not None and collection_readable:
        try:
            dense_smoke = bool(store.search(smoke_query_embedding, top_k=1))
        except Exception as exc:
            errors.append(f"dense retrieval smoke failed: {type(exc).__name__}: {exc}")

    if duplicate_chunk_ids:
        errors.append("duplicate chunk IDs")
    if missing_metadata:
        errors.append("required chunk metadata is missing")
    if not collection_readable:
        errors.append("collection is not readable")
    if not dimension_consistent:
        errors.append("embedding dimensionality is inconsistent or unavailable")
    if not dense_smoke:
        errors.append("dense retrieval smoke did not pass")
    if expected_chunk_count is not None and len(chunks) != expected_chunk_count:
        errors.append(f"chunk count is {len(chunks)}, expected {expected_chunk_count}")

    passed = (
        collection_readable
        and len(actual_document_ids) == manifest.index.document_count
        and (expected_chunk_count is None or len(chunks) == expected_chunk_count)
        and not (expected_document_ids - actual_document_ids)
        and not (actual_document_ids - expected_document_ids)
        and not duplicate_chunk_ids
        and not missing_chunk_ids
        and not unexpected_chunk_ids
        and not missing_metadata
        and dimension_consistent
        and dense_smoke
    )
    return CollectionValidationReport(
        passed=passed,
        manifest_document_count=manifest.index.document_count,
        actual_document_count=len(actual_document_ids),
        manifest_chunk_count=manifest.index.chunk_count,
        actual_chunk_count=len(chunks),
        missing_document_ids=tuple(sorted(expected_document_ids - actual_document_ids)),
        unexpected_document_ids=tuple(sorted(actual_document_ids - expected_document_ids)),
        duplicate_chunk_ids=duplicate_chunk_ids,
        missing_chunk_ids=missing_chunk_ids,
        unexpected_chunk_ids=unexpected_chunk_ids,
        missing_metadata=missing_metadata,
        embedding_dimension=dimension,
        expected_embedding_dimension=expected_embedding_dimension,
        embedding_dimension_consistent=dimension_consistent,
        collection_readable=collection_readable,
        dense_retrieval_smoke=dense_smoke,
        errors=tuple(dict.fromkeys(errors)),
    )


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace the lifecycle JSON, preserving the original active pointer on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


LifecycleFlagWriter = Callable[..., bool | None]
"""Set the Chroma ``lifecycle`` flag of one Collection (collection name, target status, ``missing_ok``).

Returns whether the flag was written; a writer that meets a missing Collection with
``missing_ok=True`` returns False instead of raising, so a caller can mark a Collection that is
created later in the same flow.
"""


class LifecycleMetadataMismatch(RuntimeError):
    """A pointer and its Collection disagree about the lifecycle status: the state file was edited,
    the flag was changed behind the lifecycle's back, or an activation happened without the writer.

    Raised only by ``require_consistency``; ``verify_consistency`` keeps returning the problems so an
    audit can report them without turning a serviceable state into an outage.
    """


class ChromaFlagWriter:
    """Rewrite one Collection's ``lifecycle`` metadata through an existing Chroma client.

    Chroma's ``modify`` replaces a Collection's metadata wholesale and rejects metadata containing
    the legacy ``hnsw:space`` key, so the existing fields are read back, that obsolete key is
    dropped, and everything else is preserved. The distance metric is fixed by the Collection
    configuration and is unaffected by a metadata rewrite (measured on Chroma 1.5.9). A collection
    already at the requested status is left untouched, which keeps every transition idempotent.

    ``missing_ok`` exists for exactly one call site. ``prepare_staging`` runs *before* the builder
    creates the staging Collection, so a strict ``get_collection`` there raises ``NotFoundError`` on
    every real build. Passing ``missing_ok=True`` returns False and leaves the flag to whoever
    creates the Collection (the real builder already sets ``lifecycle=STAGING`` on creation).
    Activation and rollback stay strict: a missing serving collection is a real error.
    """

    def __init__(self, client: Any) -> None:
        self.client = client

    def __call__(
        self,
        collection_name: str,
        status: LifecyclePointerStatus,
        *,
        missing_ok: bool = False,
    ) -> bool:
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception as exc:  # noqa: BLE001 - only the "not found" shape may be swallowed
            if missing_ok and _is_missing_collection(exc):
                return False
            raise
        merged = {
            key: value
            for key, value in (collection.metadata or {}).items()
            if key != "hnsw:space"
        }
        if merged.get("lifecycle") == status:
            return True
        collection.modify(metadata={**merged, "lifecycle": status})
        return True


def _is_missing_collection(exc: BaseException) -> bool:
    """Recognise Chroma's "collection does not exist" error without importing it at module scope."""

    if type(exc).__name__ == "NotFoundError":
        return True
    message = str(exc).lower()
    return "does not exist" in message and "collection" in message


def make_chroma_flag_writer(persist_path: str | Path) -> LifecycleFlagWriter:
    """Build the production ``flag_writer`` against a Chroma persist directory."""

    import chromadb

    return ChromaFlagWriter(chromadb.PersistentClient(path=str(persist_path)))


class IndexLifecycle:
    """The single authority over which index is serving, and the one place that keeps the
    ``index_state.json`` pointers and the per-Collection ``lifecycle`` flags in agreement.

    Two records describe the same reality: the pointer file decides what production reads, while
    each Collection carries a ``lifecycle`` flag that runtime validates before it will open the
    collection (and that the ingest guard reads to protect a serving index). Keeping them in step by
    hand leaked that responsibility to scripts/ and left ``activate()``/``rollback()`` able to return
    a state whose Collections runtime refuses to open, so the writer is injected here instead and
    every state transition owns both records.

    ``flag_writer`` stays optional so pure pointer tests can run without Chroma. When it is absent
    the transitions keep their historical pointer-only behaviour, and ``verify_consistency`` reports
    the resulting disagreement instead of hiding it."""

    def __init__(
        self,
        state_path: str | Path,
        knowledge_base_id: str,
        *,
        flag_writer: LifecycleFlagWriter | None = None,
    ) -> None:
        """Inject the state file path, the knowledge base identity and the optional flag writer."""

        if not knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        self.state_path = Path(state_path)
        self.knowledge_base_id = knowledge_base_id
        self.flag_writer = flag_writer

    def load_state(self) -> LifecycleState:
        """Read and validate the current active pointer state."""

        state = LifecycleState.model_validate_json(self.state_path.read_text(encoding="utf-8"))
        if state.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("lifecycle state belongs to another knowledge base")
        return state

    def _apply_flags(
        self,
        changes: Mapping[str, LifecyclePointerStatus],
        *,
        missing_ok: bool = False,
    ) -> tuple[str, ...]:
        """Set the Collection flags named by ``changes`` (collection -> target status), idempotently.

        Callers must order their writes so that the collection the pointer currently names is never
        demoted before the pointer moves off it; see ``_commit_transition``.

        ``missing_ok`` is only for a transition that runs before its Collection exists; it returns the
        names that were skipped so the caller can report them instead of silently claiming success.
        """

        if self.flag_writer is None:
            return ()
        skipped: list[str] = []
        for collection_name, status in changes.items():
            written = self.flag_writer(collection_name, status, missing_ok=missing_ok)
            if written is False:
                skipped.append(collection_name)
        return tuple(skipped)

    def _commit_transition(
        self,
        next_state: LifecycleState,
        *,
        promote: str | None = None,
        demote: str | None = None,
    ) -> LifecycleState:
        """Publish a new state, then retire the collection it replaced.

        The obvious order -- write every flag, then the pointer -- is wrong in the direction that
        matters. If the pointer write fails after the old collection has already been demoted, the
        pointer still names a collection whose flag is now PREVIOUS, so ``runtime`` refuses to open
        *any* index and the system is unusable. That is strictly worse than the pointer-only
        behaviour this class used to have.

        So the two flag writes are split around the pointer:

          1. promote the incoming collection to ACTIVE -- the outgoing one is still ACTIVE, so the
             current pointer remains serviceable if this fails;
          2. write the pointer -- the commit point; the incoming collection is already ACTIVE, so it
             is serviceable the moment the pointer names it;
          3. demote the outgoing collection to PREVIOUS -- now the state converges.

        A crash between 1 and 2 leaves two collections flagged ACTIVE, which serving ignores (the
        pointer decides) and which only makes the ingest guard more conservative. Re-running the
        transition is idempotent. A crash between 2 and 3 leaves the old collection over-privileged
        but the system serving correctly; ``verify_consistency`` reports it and the next transition
        repairs it.
        """

        if promote is not None:
            self._apply_flags({promote: "ACTIVE"})
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        if demote is not None and demote != promote:
            self._apply_flags({demote: "PREVIOUS"})
        return next_state

    def verify_consistency(self, read_flag: Callable[[str], str | None]) -> tuple[str, ...]:
        """Return one message per pointer whose Collection flag disagrees; empty means consistent.

        ``read_flag`` returns the Collection's current ``lifecycle`` value (None when the collection
        or the flag is missing). This is a read-only audit, used by startup checks and tests.
        """

        state = self.load_state()
        problems: list[str] = []
        for name, pointer in (
            ("active", state.active),
            ("previous", state.previous),
            ("staging", state.staging),
        ):
            if pointer is None:
                continue
            actual = read_flag(pointer.collection_name)
            if actual != pointer.status:
                problems.append(
                    f"{name} pointer {pointer.collection_name!r} says {pointer.status} "
                    f"but the collection flag is {actual!r}"
                )
        return tuple(problems)

    def initialize_active(
        self,
        manifest: KnowledgeBaseManifest,
        *,
        manifest_path: str | None = None,
    ) -> LifecycleState:
        """Establish the initial active state from one known-good manifest."""

        if manifest.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("manifest belongs to another knowledge base")
        if manifest.index.status != "ACTIVE":
            raise ValueError("initial active manifest must have status ACTIVE")
        state = LifecycleState(
            knowledge_base_id=self.knowledge_base_id,
            active=IndexPointer.from_manifest(
                manifest,
                status="ACTIVE",
                manifest_path=manifest_path,
            ),
        )
        self._apply_flags({manifest.index.collection_name: "ACTIVE"})
        _write_json_atomic(self.state_path, state.model_dump(mode="json"))
        return state

    def prepare_staging(self, manifest: KnowledgeBaseManifest) -> LifecycleState:
        """Register the staging identity; the serving index is untouched.

        The staging Collection does **not** exist yet at this point: the caller registers the
        pointer first and the builder creates the Collection afterwards. Writing the flag here is
        therefore best-effort -- it only succeeds when a Collection with that name is already there
        (a leftover being re-registered), and the real builder sets ``lifecycle=STAGING`` when it
        creates the Collection. Before this distinction existed, a strict write here raised
        ``NotFoundError`` on every real build and no staging rebuild could start.
        """

        if manifest.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("manifest belongs to another knowledge base")
        if manifest.index.status != "STAGING":
            raise ValueError("staging manifest must have status STAGING")
        state = self.load_state()
        next_state = state.model_copy(
            update={"staging": IndexPointer.from_manifest(manifest, status="STAGING")}
        )
        # Best effort: the Collection is created later, so a missing name is expected rather than
        # an error. The pointer is the commit point either way.
        self._apply_flags({manifest.index.collection_name: "STAGING"}, missing_ok=True)
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        return next_state

    def activate(
        self,
        manifest: KnowledgeBaseManifest,
        validation: CollectionValidationReport,
        *,
        manifest_path: str | None = None,
    ) -> LifecycleState:
        """Turn staging into active only after validation PASSES, keeping the old active index.

        The manifest that names the new active index is not trusted for its size fields: the build
        registers provisional counts before the corpus is known, so the pointer is written from the
        validated measurement. Otherwise every activation would publish a pointer whose
        ``chunk_count`` describes the previous index.
        """

        if manifest.index.status != "STAGING":
            raise ValueError("activation requires a staging manifest")
        if not validation.passed:
            raise ValueError("cannot activate a staging collection that failed validation")
        state = self.load_state()
        if state.staging is None:
            raise ValueError("no staging pointer is registered")
        if state.staging.index_signature != manifest.index.index_signature or state.staging.collection_name != manifest.index.collection_name:
            raise ValueError("staging pointer does not match manifest")
        if manifest.knowledge_base_id != self.knowledge_base_id:
            raise ValueError("manifest belongs to another knowledge base")
        measured = manifest.with_index_state(
            document_count=validation.actual_document_count,
            chunk_count=validation.actual_chunk_count,
        )
        next_state = LifecycleState(
            knowledge_base_id=self.knowledge_base_id,
            active=IndexPointer.from_manifest(measured, status="ACTIVE", manifest_path=manifest_path),
            previous=state.active.model_copy(update={"status": "PREVIOUS"}),
            staging=None,
        )
        # Both sides move: the incoming index becomes ACTIVE and the outgoing one is demoted to
        # PREVIOUS. The commit helper promotes first, moves the pointer, and only then demotes, so a
        # failure at any point leaves a serviceable index rather than a pointer to a demoted one.
        return self._commit_transition(
            next_state,
            promote=manifest.index.collection_name,
            demote=state.active.collection_name,
        )

    def rollback(self) -> LifecycleState:
        """Restore the previous pointer directly, without re-parsing, re-Embedding or rebuilding the old index."""

        state = self.load_state()
        if state.previous is None:
            raise ValueError("no previous known-good index is available")
        next_state = LifecycleState(
            knowledge_base_id=self.knowledge_base_id,
            active=state.previous.model_copy(update={"status": "ACTIVE"}),
            previous=state.active.model_copy(update={"status": "PREVIOUS"}),
            staging=None,
        )
        # Symmetric to activate: the collection being restored must carry lifecycle=ACTIVE or runtime
        # refuses to open it, and the one it replaces must stop claiming ACTIVE -- but only after the
        # pointer has moved, so the outgoing index is never demoted while still being served.
        return self._commit_transition(
            next_state,
            promote=state.previous.collection_name,
            demote=state.active.collection_name,
        )

    def discard_staging(self) -> LifecycleState:
        """Clear only the staging pointer; active/previous are untouched and no Chroma collection is deleted.

        The caller is responsible for deleting the staging collection through the Chroma API first and
        for confirming beforehand that it is neither active nor previous; this method only maintains the
        pointer, avoiding hand-edited JSON. When staging is already null it returns immediately without
        rewriting the state file.

        The discarded Collection's flag is deliberately untouched. It is being deleted by the caller;
        rewriting its flag to some "orphan" value would invent a fourth lifecycle state that nothing
        else understands, and a failed deletion would then leave a Collection claiming a status no
        reader handles. ``scripts/prune_chroma_collections.py`` protects only names referenced by the
        pointer file, so an undeleted staging Collection is unreferenced and prunable -- which is the
        intended recovery path. ``verify_consistency`` reports it while the pointer still exists.
        """

        state = self.load_state()
        if state.staging is None:
            return state
        next_state = state.model_copy(update={"staging": None})
        _write_json_atomic(self.state_path, next_state.model_dump(mode="json"))
        return next_state

    def require_consistency(self, read_flag: Callable[[str], str | None]) -> None:
        """Raise ``LifecycleMetadataMismatch`` when any pointer disagrees with its Collection flag.

        This is the startup gate: ``verify_consistency`` remains the reporting form for audits and
        tests, while the runtime path calls this so a divergence surfaces as a load error instead of
        a mysterious "runtime refuses to open the collection" much later.
        """

        problems = self.verify_consistency(read_flag)
        if problems:
            raise LifecycleMetadataMismatch("; ".join(problems))
