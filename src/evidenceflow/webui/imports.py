"""文档库的 PDF 增量入库任务，写入成功后切换正式快照。"""

from __future__ import annotations

import logging
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path, PureWindowsPath
from typing import BinaryIO, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from evidenceflow.config import Settings
from evidenceflow.domain.ids import build_document_id
from evidenceflow.knowledge import (
    BuildJob, BuildJobStore, IndexLifecycle, IndexPointer, LifecycleState,
    current_pipeline_from_settings, load_manifest,
)
from evidenceflow.knowledge.lifecycle import _write_json_atomic
from evidenceflow.knowledge.staging import (
    StagingPorts,
    activate_validated_staging,
    discard_staging_collection,
    index_update,
    make_real_ports,
    run_staging_build,
)

logger = logging.getLogger(__name__)

PDF_LIMIT = 100 * 1024 * 1024
_COPY_BLOCK = 1024 * 1024
_JOB_FILE = "staging_build_job.json"
_STAGING_OCCUPIED = "已有候选索引。请先完成它的发布或清理，再添加 PDF。"


class ImportProblem(RuntimeError):
    """可直接展示给用户的入库或发布错误。"""

    def __init__(self, message: str, status_code: int = 409) -> None:
        super().__init__(message)
        self.status_code = status_code


class _ImportRecord(BaseModel):
    """在浏览器刷新或服务重启后定位同一份上传文件与 BuildJob。"""

    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    file_name: str
    status: Literal["BUILDING", "READY", "PUBLISHING", "PUBLISHED", "FAILED", "DISCARDED"]
    message: str | None = None

    @property
    def source_path(self) -> Path:
        """上传文件按任务隔离，文件名仅用于展示和语料内寻址。"""
        return Path("webui_uploads") / self.id / self.file_name


def _pdf_name(filename: str) -> str:
    """接受可作为单个本地文件名的 PDF，禁止路径成分和 Windows 保留名。"""

    name = unicodedata.normalize("NFC", filename)
    if (
        Path(name).suffix.lower() != ".pdf"
        or any(char in '<>:"/\\|?*' or ord(char) < 32 for char in name)
        or PureWindowsPath(name).is_reserved()
    ):
        raise ImportProblem("请选择文件名有效的 PDF。", 422)
    return name


def _copy_limited(source: BinaryIO, target: Path, limit: int, *, prefix: bytes = b"") -> None:
    """上传按块写入，并以实际读取字节数执行上限。"""

    remaining = limit - len(prefix)
    writer = target.open("xb")
    try:
        with writer:
            writer.write(prefix)
            while block := source.read(_COPY_BLOCK):
                remaining -= len(block)
                if remaining < 0:
                    raise ImportProblem(f"文件过大，上限为 {limit // (1024 * 1024)} MiB。", 413)
                writer.write(block)
    except (OSError, ImportProblem):
        target.unlink(missing_ok=True)
        raise


class PDFImports:
    """每个生命周期索引只有一个候选位，任务状态可持久化读取。"""

    def __init__(self, settings: Settings, *, ports_factory: Callable[..., StagingPorts] = make_real_ports) -> None:
        self.settings = settings
        self._ports = ports_factory
        self._base = settings.index_state_path.resolve().with_suffix(".webui")
        self._current = self._base / "current.json"
        self._working: str | None = None
        self._files: tuple[int, int] | None = None

    def _state(self) -> LifecycleState:
        """候选构建以现有 active 为基线，沿用相同的受治理生命周期。"""

        try:
            return LifecycleState.model_validate_json(self.settings.index_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ImportProblem("添加 PDF 需要可读取的已发布索引。请先检查知识库的生命周期配置。") from None

    def _record(self) -> _ImportRecord | None:
        try:
            return _ImportRecord.model_validate_json(self._current.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def _update(self, record: _ImportRecord, **changes: str | None) -> _ImportRecord:
        record = record.model_copy(update=changes)
        _write_json_atomic(self._current, record.model_dump(mode="json"))
        return record

    def _job(self, record: _ImportRecord) -> BuildJob | None:
        try:
            return BuildJobStore(self._base / record.id / _JOB_FILE).load()
        except FileNotFoundError:
            return None

    @staticmethod
    def _matches(pointer: IndexPointer | None, job: BuildJob | None) -> bool:
        return bool(
            pointer is not None and job is not None
            and pointer.collection_name == job.target_collection_name
            and pointer.index_signature == job.target_index_signature
        )

    def _source(self, record: _ImportRecord) -> Path:
        """构造语料内路径，拒绝指向语料之外的本地链接。"""

        root = self.settings.source_docs_path.resolve()
        path = root / record.source_path
        if not path.resolve().is_relative_to(root):
            raise ImportProblem("上传目录需位于已配置的语料根目录内。")
        return path

    def _publication_setting(self) -> str | None:
        if "collection_name" in self.settings.model_fields_set:
            return "发布后由生命周期选择新索引。请移除显式 COLLECTION_NAME 并重启 WebUI。"
        return None

    def _view(self, record: _ImportRecord, state: LifecycleState) -> dict:
        """区分构建进度、可恢复的待提交状态和已完成的指针切换。"""

        job = self._job(record)
        status, message = record.status, record.message
        if self._matches(state.active, job) or self._matches(state.previous, job):
            status, message = "PUBLISHED", None
        elif self._working != record.id and status not in {"PUBLISHED", "DISCARDED"}:
            ready = (
                self._matches(state.staging, job) and job.status == "READY"
                and job.chunk_count == state.staging.chunk_count
                and Path(state.staging.manifest_path).is_file()
            )
            if ready:
                status = "READY"
            elif status != "FAILED":
                status, message = "FAILED", "候选构建未完成或已变更，可以放弃此次任务后重试。"
        phase = job.status if job is not None and status == "BUILDING" else None
        if phase == "READY":
            phase = "VALIDATING"
        return {
            "id": record.id,
            "file_name": record.file_name,
            "document_id": build_document_id(record.source_path.as_posix()),
            "status": status,
            "phase": phase or ("SCANNING" if status == "BUILDING" else None),
            "message": message,
            "document_count": job.document_count if job is not None else None,
            "chunk_count": job.chunk_count if job is not None else None,
            "file_progress": self._files if status == "BUILDING" else None,
        }

    def status(self) -> dict:
        """读取当前任务和上传可用性；只读取状态文件与现有 BuildJob。"""

        try:
            state = self._state()
        except ImportProblem as exc:
            return {"job": None, "blocked": str(exc), "can_add": False}
        record = self._record()
        job = self._view(record, state) if record else None
        busy = job is not None and job["status"] not in {"PUBLISHED", "DISCARDED"}
        blocked = self._publication_setting()
        if state.staging is not None and (record is None or not self._matches(state.staging, self._job(record))):
            blocked = _STAGING_OCCUPIED
        return {"job": job, "blocked": blocked, "can_add": not (busy or blocked)}

    def submit(self, filename: str, source: BinaryIO) -> dict:
        """接收一份本地 PDF，保存为未参与语料扫描的待构建文件。"""

        name = _pdf_name(filename)
        prefix = source.read(1024)
        if b"%PDF-" not in prefix:
            raise ImportProblem("文件内容不是可识别的 PDF。", 422)
        with index_update(self.settings.index_state_path):
            status = self.status()
            if not status.get("can_add"):
                raise ImportProblem(status.get("blocked") or "请先完成当前入库任务。")
            record = _ImportRecord(id=uuid4().hex, file_name=name, status="BUILDING")
            target = self._source(record).with_name(".incoming")
            target.parent.mkdir(parents=True, exist_ok=False)
            try:
                _copy_limited(source, target, PDF_LIMIT, prefix=prefix)
                self._update(record)
            except Exception:
                target.unlink(missing_ok=True)
                target.parent.rmdir()
                raise
            self._working = record.id
            return self._view(record, self._state())

    def _require_record(self, job_id: str) -> _ImportRecord:
        record = self._record()
        if record is None or record.id != job_id:
            raise ImportProblem("找不到这次入库任务，请刷新文档库。", 404)
        return record

    def build(self, job_id: str) -> None:
        """后台追加新 PDF，并提交完整的新索引版本。"""

        record = self._require_record(job_id)
        try:
            with index_update(self.settings.index_state_path):
                self._build(record)
        except Exception as exc:
            logger.warning("PDF staging build failed (%s)", type(exc).__name__)
            hint = str(exc) if isinstance(exc, ImportProblem) else "构建失败。请检查本地文件和模型，放弃本次任务后可重试。"
            self._update(record, status="FAILED", message=hint)
        finally:
            self._files = None
            self._working = None

    def _build(self, record: _ImportRecord) -> None:
        state = self._state()
        if state.staging is not None:
            raise ImportProblem(_STAGING_OCCUPIED)
        manifest = load_manifest(state.active.manifest_path)
        if manifest.knowledge_base_id != state.knowledge_base_id or (
            manifest.index.collection_name, manifest.index.index_signature, manifest.index.document_count
        ) != (state.active.collection_name, state.active.index_signature, state.active.document_count):
            raise ImportProblem("当前索引的语料清单与发布版本不匹配。")
        pipeline = current_pipeline_from_settings(self.settings)
        if (manifest.pipeline.embedding_model, manifest.pipeline.embedding_version) != (
            pipeline.embedding_model, pipeline.embedding_version
        ):
            raise ImportProblem("向量模型或向量协议已变化，请先构建同一向量空间的完整索引。")

        source = self._source(record)
        incoming = source.with_name(".incoming")
        incoming.rename(source)
        try:
            self._files = (0, 1)

            def report_file(completed: int, total: int) -> None:
                """记录新 PDF 的处理进度，旧片段的复制不计入文件进度。"""
                self._files = (completed, total)

            ports = self._ports(self.settings, source_files=[source], on_file=report_file, copy_from=state.active)

            def scan():
                """解析上传文件，旧扫描记录直接取自实际复用的正式索引。"""

                try:
                    (new,) = ports.scan_sources()
                except (OSError, ValueError, RuntimeError):
                    raise ImportProblem("PDF 无法读取。请确认文件未加密且包含可提取文本。", 422) from None
                if any(old.content_hash == new.content_hash for old in manifest.sources):
                    raise ImportProblem("这份 PDF 已在当前文档库中。")
                return [*manifest.sources, new]

            code, _ = run_staging_build(
                self.settings, replace(ports, scan_sources=scan), job_dir=self._base / record.id,
                expected_chunk_ids_from_active=False, activate=False, discard_after_validate=False,
                # 每个任务已有独立目录，短文件名避免 Windows 临时清单名超过路径长度。
                expected_active=state.active, manifest_name="manifest.json", activate_after_write=True,
            )
            if code:
                raise ImportProblem("候选索引构建或结构校验失败。可放弃此次任务后重试。")
            self._update(record, status="PUBLISHED", message=None)
        except Exception:
            # 只有已写完且可恢复的候选保留正式文件名，失败文件不参与后续语料扫描。
            state, job = self._state(), self._job(record)
            ready = self._matches(state.staging, job) and job.status == "READY"
            if not (ready or self._matches(state.active, job)):
                source.rename(incoming)
            raise

    def publish(self, job_id: str) -> dict:
        """恢复工作台已写完但尚未提交的候选任务。"""

        with index_update(self.settings.index_state_path):
            record = self._require_record(job_id)
            state = self._state()
            status = self._view(record, state)
            if status["status"] == "PUBLISHED":
                return status
            if status["status"] != "READY":
                raise ImportProblem("候选索引通过结构校验后才可发布。")
            if problem := self._publication_setting():
                raise ImportProblem(problem)
            job_dir = self._base / record.id
            self._working = record.id
            try:
                record = self._update(record, status="PUBLISHING", message=None)
                code, _ = activate_validated_staging(
                    self.settings, self._ports(self.settings),
                    job_dir=job_dir, quality_report_path=None, expected_staging=state.staging,
                    require_quality_report=False,
                )
                if code:
                    raise ImportProblem("重新核对候选索引时发现结构校验失败。", 422)
            except Exception as exc:
                actual = self._state()
                if not self._matches(actual.active, self._job(record)):
                    hint = str(exc) if isinstance(exc, ImportProblem) else "发布失败，候选索引仍可检查后重试。"
                    self._update(record, status="READY", message=hint)
                    if isinstance(exc, ImportProblem):
                        raise
                    logger.warning("PDF staging publication failed (%s)", type(exc).__name__)
                    raise ImportProblem(hint, 503) from None
            finally:
                self._working = None
            return self._view(self._update(record, status="PUBLISHED", message=None), self._state())

    def discard(self, job_id: str) -> dict:
        """只放弃本次未发布任务，不处理其他入口创建的候选索引。"""

        with index_update(self.settings.index_state_path):
            record = self._require_record(job_id)
            state = self._state()
            status = self._view(record, state)
            if status["status"] == "DISCARDED":
                return status
            if self._working or status["status"] not in {"FAILED", "READY"}:
                raise ImportProblem("只能放弃已经停止且尚未发布的任务。")
            source = self._source(record)
            if source.exists():
                for pointer in (state.active, state.previous):
                    if pointer is not None and any(
                        item.relative_path == record.source_path.as_posix()
                        for item in load_manifest(pointer.manifest_path).sources
                    ):
                        raise ImportProblem("文件正由已发布或可回滚的索引使用，请保留原文件。")
            if state.staging is not None:
                if not self._matches(state.staging, self._job(record)):
                    raise ImportProblem("当前候选索引来自其他任务，请先处理该候选。")
                ports = self._ports(self.settings)
                lifecycle = IndexLifecycle(self.settings.index_state_path, state.knowledge_base_id)
                discard_staging_collection(lifecycle, ports.delete_collection, ports.list_collections)
            source.unlink(missing_ok=True)
            source.with_name(".incoming").unlink(missing_ok=True)
            if source.parent.is_dir():
                source.parent.rmdir()
            return self._view(self._update(record, status="DISCARDED", message=None), self._state())
