"""用同源 HTTP 和 SSE 提供本地论文工作台，候选入库与已发布问答隔离。"""

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Event
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, StringConstraints

from evidenceflow.config import Settings
from evidenceflow.domain.answer import SourceReference
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.trace import TraceEvent
from evidenceflow.knowledge.staging import StagingBuildRefused
from evidenceflow.retrieval.evidence_navigation import UnknownChunkError
from evidenceflow.webui.imports import ImportProblem, PDFImports, PDF_LIMIT
from evidenceflow.webui.memory import ConversationMemory
from evidenceflow.webui.preferences import AgentPreferences
from evidenceflow.webui.service import (
    RuntimeServices,
    RuntimeSnapshot,
    WebUIDependencyError,
    WebUIResult,
    ask_question,
    describe_error,
    evidence_actions,
    progress_steps,
    render_evidence,
    render_trace,
)

_STATIC = Path(__file__).with_name("static")
_LOOPBACK = ("127.0.0.1", "localhost")
_MISSING = '请安装 WebUI 依赖后重试。运行 pip install -e ".[web]"'


class _Question(BaseModel):
    """会话标识用于加载服务端检查点，浏览器只提交本轮输入。"""

    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
    document_id: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)] | None = None
    thread_id: UUID | None = None
    turn_id: UUID | None = None
    settings: AgentPreferences | None = None
    force_search: bool = False


class _Citation(BaseModel):
    """按回答使用的快照和完整来源身份定位引文。"""

    snapshot: str
    source: SourceReference
    context: bool = False


@dataclass
class _Run:
    """记录当前运行的停止信号，由其工作线程退出后释放 WebUI 名额。"""

    id: str = field(default_factory=lambda: uuid4().hex)
    stop: Event = field(default_factory=Event)


def _passage(chunk: DocumentChunk) -> dict[str, Any]:
    """发送阅读需要的原文、位置和章节，内部索引元数据留在服务端。"""

    return {
        "content": chunk.content,
        "heading": chunk.metadata.get("heading_path"),
        "chunk_index": chunk.chunk_index,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
    }


def _sse(name: str, data: object) -> str:
    """将一帧发送为单行 JSON，换行或引号由 JSON 编码。"""

    return f"event: {name}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _answer_payload(result: WebUIResult) -> dict[str, Any]:
    """仅提交已通过引用校验的最终答案和公开运行明细。"""

    answer = result.answer
    status = "cancelled" if result.cancelled else "failed"
    if answer is not None:
        status = answer.status.value
    evidence = render_evidence(result)
    trace = render_trace(result) if result.events else ""
    return {
        "status": status,
        "answer": answer.answer if answer is not None else "",
        "sources": [source.model_dump(mode="json") for source in answer.sources] if answer is not None else [],
        "snapshot": result.snapshot,
        "error": result.error,
        "steps": [asdict(step) for step in progress_steps(result.events, status=result.trace_status)],
        "evidence_actions": evidence_actions(result.events, status=result.trace_status),
        "settings": dict(result.limits),
        "trace": f"### 证据去向\n\n{evidence}\n\n{trace}" if evidence else trace,
        "searches": sum(event.name == "agent.search.completed" for event in result.events),
        "elapsed_ms": result.elapsed_ms,
    }


def build_app(services: RuntimeServices, *, imports: PDFImports | None = None) -> Any:
    """组装本地工作台，问答与入库端口均可在离线测试中替换。"""

    try:
        import python_multipart  # noqa: F401 - 提前检查 FastAPI 文件表单依赖
        from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request, UploadFile
        from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
        from fastapi.staticfiles import StaticFiles
        from starlette.middleware.trustedhost import TrustedHostMiddleware
    except ImportError as exc:  # pragma: no cover - 取决于是否安装可选依赖
        raise WebUIDependencyError(_MISSING) from exc

    app = FastAPI(title="EvidenceFlow", openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(_LOOPBACK))
    app.mount("/assets", StaticFiles(directory=_STATIC), name="workspace-assets")
    active_run: _Run | None = None
    pdf_imports = imports if imports is not None else PDFImports(services.settings)
    memory = ConversationMemory(services.settings.index_state_path.resolve().with_suffix(".webui") / "conversations.sqlite3")

    def active_snapshot(token: str | None = None) -> RuntimeSnapshot:
        """同一请求固定读取一个已发布版本，并拒绝误读旧引文。"""

        try:
            snapshot = services.snapshot()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=describe_error(exc)) from None
        if token is not None and token != snapshot.token:
            raise HTTPException(status_code=409, detail="索引版本已变更，请重新提问以定位当前原文。")
        return snapshot

    def workspace_write(request: Request) -> None:
        """写入请求需要工作台脚本标头，第三方网页的普通表单不能触发。"""

        if request.headers.get("X-EvidenceFlow-Workspace") != "1":
            raise HTTPException(status_code=403, detail="请从本地 EvidenceFlow 工作台提交。")

    @app.exception_handler(ImportProblem)
    async def import_problem(request: Request, exc: ImportProblem) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=exc.status_code)

    @app.exception_handler(StagingBuildRefused)
    async def staging_refusal(request: Request, exc: StagingBuildRefused) -> JSONResponse:
        return JSONResponse({"detail": "索引正在更新或候选已变更，请刷新文档库。"}, status_code=409)

    @app.get("/")
    async def workspace() -> Any:
        """工作台脚本、字体和样式均从当前进程加载。"""

        return FileResponse(
            _STATIC / "index.html",
            headers={
                "Cache-Control": "no-cache",
                "Content-Security-Policy": "default-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            },
        )

    @app.get("/api/library")
    def library() -> dict[str, Any]:
        """列出索引内真实文档与覆盖页码，供列表和本地文件名筛选。"""

        snapshot = active_snapshot()
        tools = snapshot.agent.tools
        return {
            "index": tools.identity.model_dump(mode="json"),
            "snapshot": snapshot.token,
            "documents": [doc.model_dump(mode="json") for doc in tools.list_documents()],
        }

    @app.get("/api/agent/settings")
    def agent_settings() -> dict[str, Any]:
        """读取默认运行预算以及可编辑字段的范围，不装配检索模型。"""

        return {
            "defaults": services.budget_limits(),
            "bounds": {
                "searches": [1, 3], "opens": [0, 20], "deadline_seconds": [0.1, 600],
            },
        }

    @app.post("/api/agent/settings")
    def validate_agent_settings(body: AgentPreferences) -> dict[str, Any]:
        """提前校验会话设置。浏览器保存覆盖项，本轮请求仍会再校验一次。"""
        return body.effective(services.settings)

    @app.delete("/api/conversations/{thread_id}", status_code=204, response_model=None,
                dependencies=[Depends(workspace_write)])
    async def delete_conversation(thread_id: UUID) -> Any:
        """清理会话证据时避免与同一工作台的在途写入冲突。"""
        if active_run is not None:
            raise HTTPException(status_code=409, detail="请在当前问答完成后删除对话。")
        memory.delete(str(thread_id))
        return Response(status_code=204)

    @app.get("/api/imports/current")
    def current_import() -> dict[str, Any]:
        """提供当前上传任务的真实阶段，供文档库轮询。"""

        return pdf_imports.status()

    @app.post("/api/imports", status_code=202, dependencies=[Depends(workspace_write)])
    def upload_pdf(file: UploadFile, background_tasks: BackgroundTasks) -> dict[str, Any]:
        """接受一个 PDF，后台写入并自动切换知识库索引。"""

        if file.size is not None and file.size > PDF_LIMIT:
            raise ImportProblem("PDF 大小超过 100 MiB。", 413)
        job = pdf_imports.submit(file.filename or "", file.file)
        background_tasks.add_task(pdf_imports.build, job["id"])
        return job

    @app.post("/api/imports/{job_id}/publish", dependencies=[Depends(workspace_write)])
    def publish_pdf(job_id: str) -> dict[str, Any]:
        """恢复本地工作台一项写入已完成的入库任务。"""

        return pdf_imports.publish(job_id)

    @app.delete("/api/imports/{job_id}", dependencies=[Depends(workspace_write)])
    def discard_import(job_id: str) -> dict[str, Any]:
        """清理工作台创建的未发布任务。"""

        return pdf_imports.discard(job_id)

    @app.get("/api/passage")
    def passage(document_id: str, snapshot: str, offset: int = Query(0, ge=0)) -> dict[str, Any]:
        """按文档身份和已索引顺序读取一个片段，供翻页阅读。"""

        tools = active_snapshot(snapshot).agent.tools
        try:
            return _passage(tools.document_chunk(document_id, offset))
        except UnknownChunkError:
            raise HTTPException(status_code=404, detail="当前索引中找不到这个文档片段。") from None

    @app.post("/api/citation")
    def citation(body: _Citation) -> dict[str, Any]:
        """读取同版本、同页码的引文，按需附带前后各一个邻块。"""

        snapshot = active_snapshot(body.snapshot)
        match = snapshot.citation_context(body.source, radius=int(body.context))
        if match is None:
            raise HTTPException(status_code=404, detail="当前索引中找不到与该引文匹配的原文。")
        return {
            "anchor": _passage(match.anchor),
            "previous": [_passage(chunk) for chunk in match.previous],
            "following": [_passage(chunk) for chunk in match.following],
        }

    @app.post("/api/ask")
    async def ask(body: _Question) -> Any:
        """在一个工作线程执行整轮问答，以 SSE 实时发送公开步骤与终态。"""

        nonlocal active_run
        if active_run is not None:
            raise HTTPException(status_code=503, detail="已有一轮问答在运行，请等待结束后重试。")

        loop = asyncio.get_running_loop()
        messages: asyncio.Queue[TraceEvent | WebUIResult] = asyncio.Queue()
        run = _Run()

        def on_event(event: TraceEvent) -> None:
            loop.call_soon_threadsafe(messages.put_nowait, event)

        def work() -> WebUIResult:
            def execute(recent) -> WebUIResult:
                """在同一个 SSE 运行内执行问答及可选的会话恢复。"""
                return ask_question(
                    services, body.query, on_event=on_event, cancelled=run.stop.is_set, document_id=body.document_id,
                    recent=recent, preferences=body.settings, force_search=body.force_search,
                )

            if body.thread_id is None:
                return execute(())
            return memory.run(str(body.thread_id), str(body.turn_id or run.id), execute)

        def finish(task: asyncio.Future[WebUIResult]) -> None:
            nonlocal active_run
            active_run = None
            try:
                result = task.result()
            except Exception as exc:
                result = WebUIResult(question=body.query, trace_status="failed", error=describe_error(exc))
            messages.put_nowait(result)

        pending = loop.run_in_executor(None, work)
        active_run = run
        pending.add_done_callback(finish)

        async def frames() -> AsyncIterator[str]:
            events: list[TraceEvent] = []
            last_steps = []
            last_actions = []
            try:
                yield _sse("start", {"run_id": run.id})
                while True:
                    try:
                        item = await asyncio.wait_for(messages.get(), timeout=10)
                    except TimeoutError:
                        yield ": pulse\n\n"
                        continue
                    if isinstance(item, WebUIResult):
                        yield _sse("result", _answer_payload(item))
                        return
                    events.append(item)
                    steps = progress_steps(events)
                    if steps != last_steps:
                        last_steps = steps
                        yield _sse("progress", [asdict(step) for step in steps])
                    actions = evidence_actions(events)
                    if actions != last_actions:
                        last_actions = actions
                        yield _sse("evidence", actions)
            finally:
                # 浏览器关闭页面时继续保留名额，直到在途同步步骤真正退出。
                run.stop.set()

        return StreamingResponse(
            frames(), media_type="text/event-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/api/runs/{run_id}/stop", status_code=204, response_model=None)
    async def stop(run_id: str) -> Any:
        """只通知对应运行停止，SSE 在工作线程退出后报告终态。"""

        if active_run is not None and active_run.id == run_id:
            active_run.stop.set()
        return Response(status_code=204)

    return app


def launch(
    settings: Settings | None = None,
    *,
    server_name: str = "127.0.0.1",
    server_port: int = 7860,
) -> Any:
    """启动复用当前项目设置的本地 WebUI，监听地址限制为回环。"""

    if server_name not in _LOOPBACK:
        raise ValueError("WebUI only accepts a loopback host")
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver  # noqa: F401 - 检查旧环境是否装齐记忆依赖
    except ImportError as exc:
        raise WebUIDependencyError(_MISSING) from exc
    app = build_app(RuntimeServices(settings or Settings()))
    try:
        from uvicorn import run
    except ImportError as exc:  # pragma: no cover - 取决于是否安装可选依赖
        raise WebUIDependencyError(_MISSING) from exc
    run(app, host=server_name, port=server_port)
    return app


def main(argv: Sequence[str] | None = None, *, settings: Settings | None = None) -> int:
    """启动本地工作台，并在首次装配模型前关闭后台权重格式转换。"""

    parser = argparse.ArgumentParser(prog="python -m evidenceflow.webui", description="本地论文证据工作台")
    parser.add_argument("--host", choices=_LOOPBACK, default="127.0.0.1", help="本机回环地址")
    parser.add_argument("--port", type=int, default=7860, help="监听端口，默认 7860")
    args = parser.parse_args(argv)
    # BGE-M3 已可使用原版 PyTorch 权重。禁止后台转换下载，避免退出时被下载线程拖住。
    os.environ["DISABLE_SAFETENSORS_CONVERSION"] = "true"
    try:
        launch(settings, server_name=args.host, server_port=args.port)
    except WebUIDependencyError as exc:
        print(exc)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - 进程入口
    raise SystemExit(main())
