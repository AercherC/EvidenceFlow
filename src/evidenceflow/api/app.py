"""把现有同步证据 Agent 暴露为只读 HTTP 服务，限制每进程问答并发。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from evidenceflow.agent.adaptive import AdaptiveActionError, AdaptiveAgentAnswer, AdaptiveAgentService
from evidenceflow.agent.policy import DeadlineExceeded
from evidenceflow.agent.result import TerminationReason
from evidenceflow.citation import CitationValidationError
from evidenceflow.config import Settings
from evidenceflow.domain.answer import SourceReference
from evidenceflow.domain.evidence import EvidenceIndexIdentity
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.generation import GenerationContractError
from evidenceflow.knowledge import ActiveCollectionError
from evidenceflow.llm import LLMRequestError
from evidenceflow.retrieval.exact_dense import ExactDenseIndexError

logger = logging.getLogger(__name__)


class AskRequest(BaseModel):
    """问答接口接收单个非空问题，去除首尾空白。"""

    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class CitedAnswer(BaseModel):
    """复用 RAGAnswer 的最终文本和实际引用，省略仅供调试的检索原文。"""

    model_config = ConfigDict(from_attributes=True)
    query: str
    status: AnswerStatus
    answer: str
    sources: list[SourceReference]


class AskUsage(BaseModel):
    """复用 Agent 的运行计数，重排对数按其预算上界计。"""

    model_config = ConfigDict(from_attributes=True)
    llm_calls: int = Field(description="Agent 记账的模型调用次数，传输重试和计费 token 不在此字段内。")
    searches: int
    opens: int
    rerank_pairs: int = Field(description="按每次搜索的重排对数上限累计。")


class AskResponse(BaseModel):
    """最终答案、终止原因及本次使用的索引和预算计数。"""

    answer: CitedAnswer
    index: EvidenceIndexIdentity
    termination_reason: TerminationReason
    usage: AskUsage


class ReadyResponse(BaseModel):
    """就绪探针返回该进程已加载的非空索引身份。"""

    status: Literal["ready"] = "ready"
    index: EvidenceIndexIdentity


class _AskPool:
    """在事件循环内管理准入，把整次同步问答交给同一工作线程。"""

    def __init__(self, settings: Settings, service: AdaptiveAgentService) -> None:
        self.service = service
        self.index = service.tools.identity
        self.executor = ThreadPoolExecutor(max_workers=settings.api_max_concurrent_requests)
        self.slots = asyncio.Semaphore(settings.api_max_concurrent_requests)
        self.capacity = settings.api_max_concurrent_requests + settings.api_max_queued_requests
        self.queue_timeout = settings.api_queue_timeout_seconds
        self.admitted = 0

    async def run(self, query: str) -> AdaptiveAgentAnswer:
        """超出容量立即拒绝，排队超时无需启动 Agent。"""

        if self.admitted >= self.capacity:
            raise HTTPException(status_code=503, detail="capacity_exceeded")
        # 检查和记账之间不让出事件循环，后续请求看到的是完整的在途与排队数。
        self.admitted += 1
        try:
            async with asyncio.timeout(self.queue_timeout):
                await self.slots.acquire()
        except TimeoutError:
            self.admitted -= 1
            raise HTTPException(status_code=503, detail="queue_timeout") from None
        except BaseException:
            self.admitted -= 1
            raise

        try:
            work = asyncio.get_running_loop().run_in_executor(self.executor, self.service.run, query)
        except BaseException:
            self.slots.release()
            self.admitted -= 1
            raise
        # HTTP 任务取消时，同步推理仍可能在运行。只在实际退出后释放名额。
        work.add_done_callback(self._finish)
        return await asyncio.shield(work)

    def _finish(self, work: asyncio.Future[AdaptiveAgentAnswer]) -> None:
        """归还名额，同时接收客户端离开后产生的异常。"""

        self.slots.release()
        self.admitted -= 1
        if not work.cancelled():
            work.exception()


def create_app(
    settings: Settings | None = None,
    *,
    builder: Callable[[Settings], AdaptiveAgentService] | None = None,
) -> FastAPI:
    """创建 ASGI 应用；可注入装配函数用于离线测试，生产走 CLI 共用的组合根。"""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        active_settings = settings if settings is not None else Settings()
        load_service = builder
        if load_service is None:
            from evidenceflow.runtime import build_agent_service

            load_service = build_agent_service
        try:
            service = load_service(active_settings)
        except (ActiveCollectionError, ExactDenseIndexError) as exc:
            logger.error("Active index unavailable during API startup (%s)", type(exc).__name__)
        else:
            if service.tools.identity.chunk_count:
                app.state.ask_pool = _AskPool(active_settings, service)
            else:
                logger.error("Active index is empty during API startup")
        try:
            yield
        finally:
            pool = app.state.ask_pool
            app.state.ask_pool = None
            if pool is not None:
                await asyncio.to_thread(pool.executor.shutdown)

    app = FastAPI(title="EvidenceFlow API", version="1", lifespan=lifespan)
    app.state.ask_pool = None

    def ready_pool(request: Request) -> _AskPool:
        """只有完成装配且绑定非空索引时才接收问答。"""

        pool = request.app.state.ask_pool
        if pool is None:
            raise HTTPException(status_code=503, detail="index_not_ready")
        return pool

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        """确认进程能响应 HTTP。"""

        return {"status": "ok"}

    @app.get("/readyz", responses={503: {"description": "索引未就绪"}})
    async def ready(request: Request) -> ReadyResponse:
        """返回当前进程服务的索引快照。"""

        return ReadyResponse(index=ready_pool(request).index)

    @app.post(
        "/v1/ask",
        responses={
            502: {"description": "模型请求或输出失败"},
            503: {"description": "索引未就绪、已满载或排队超时"},
            504: {"description": "Agent 执行超时"},
        },
    )
    async def ask(question: AskRequest, request: Request) -> AskResponse:
        """运行现有引用校验，证据不足作为正常问答结果返回。"""

        pool = ready_pool(request)
        try:
            result = await pool.run(question.query)
        except DeadlineExceeded:
            raise HTTPException(status_code=504, detail="agent_deadline_exceeded") from None
        except LLMRequestError:
            raise HTTPException(status_code=502, detail="model_request_failed") from None
        except (AdaptiveActionError, GenerationContractError, CitationValidationError):
            raise HTTPException(status_code=502, detail="model_output_invalid") from None
        return AskResponse(
            answer=CitedAnswer.model_validate(result.answer),
            index=pool.index,
            termination_reason=result.termination_reason,
            usage=AskUsage.model_validate(result),
        )

    return app
