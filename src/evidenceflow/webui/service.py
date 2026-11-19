"""WebUI 共用的问答、公开检索轨迹和同快照引文读取逻辑。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from uuid import uuid4

from evidenceflow.agent.adaptive import AdaptiveActionError, AdaptiveAgentAnswer, AdaptiveAgentService
from evidenceflow.agent.policy import AgentPolicyError, RunCancelled
from evidenceflow.citation import CitationValidationError
from evidenceflow.config import Settings
from evidenceflow.domain.answer import RAGAnswer, SourceReference
from evidenceflow.domain.evidence import EvidenceNeighborhood
from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.domain.trace import RunTrace, TraceEvent
from evidenceflow.generation import GenerationContractError
from evidenceflow.knowledge import ActiveCollectionError, IndexPointer, LifecycleState
from evidenceflow.llm import LLMConfigurationError, LLMRequestError
from evidenceflow.retrieval.evidence_navigation import UnknownChunkError
from evidenceflow.retrieval.exact_dense import ExactDenseIndexError
from evidenceflow.tracing import RunTraceRecorder, passage_identity, safe_record, trace_scope
from evidenceflow.webui.memory import RememberedTurn, follow_up_query, normalized_question
from evidenceflow.webui.preferences import AgentPreferences

_TERMINATION_LABELS = {
    "answered": "正常回答",
    "insufficient_context": "生成判定证据不足",
    "insufficient_evidence": "自适应判定证据不足",
    "budget_exhausted": "预算耗尽",
}

_TRACE_STATUS_LABELS = {
    "completed": "完成",
    "failed": "失败",
    "incomplete": "未完成（运行中断）",
    "not_started": "未开始",
}

_BUDGET_LABELS = {
    "llm_calls": "LLM 调用",
    "rerank_pairs": "重排 pair",
    "searches": "检索次数",
    "opens": "打开片段",
}

# 这些完整问句询问的是运行配置，交给知识库检索会把系统信息误当成论文事实。
_MODEL_QUESTIONS = frozenset({
    "你是什么模型", "你用的是什么模型", "当前用的是什么模型",
    "当前使用的是什么模型", "what model are you", "which model are you using",
})


class WebUIDependencyError(RuntimeError):
    """Web 服务依赖缺失，提示安装可选的 web extra。"""


Builder = Callable[[Settings], AdaptiveAgentService]


def describe_error(exc: Exception) -> str:
    """将边界异常映射到可操作的 UI 提示，省略可能含敏感值的异常消息。"""

    prefixes = (
        ((ActiveCollectionError, ExactDenseIndexError), "索引配置错误，请检查当前索引后重试。"),
        (LLMConfigurationError, "LLM 配置错误，请检查服务端模型设置。"),
        (LLMRequestError, "LLM 请求失败，请检查模型服务后重试。"),
        (AdaptiveActionError, "Agent 动作校验失败，请重试或调整问题。"),
        (AgentPolicyError, "Agent 达到运行预算或时限，请缩小问题范围后重试。"),
        (GenerationContractError, "生成格式错误，请重试。"),
        (CitationValidationError, "引用校验失败，请重试。"),
    )
    for error_type, message in prefixes:
        if isinstance(exc, error_type):
            return message
    return f"Agent 处理失败（{type(exc).__name__}），请重试。"


@dataclass(frozen=True)
class WebUIResult:
    """一次提问的页面数据：最终答案、Agent 元数据、本次 RunTrace 与错误提示。"""

    question: str
    answer: RAGAnswer | None = None
    agent: AdaptiveAgentAnswer | None = None
    trace: RunTrace | None = None
    events: tuple[TraceEvent, ...] = ()
    trace_status: str = "not_started"
    error: str | None = None
    elapsed_ms: float = 0.0
    limits: Mapping[str, int | float | bool] = field(default_factory=dict)
    cancelled: bool = False
    snapshot: str | None = None
    resolved_query: str | None = None
    document_id: str | None = None


@dataclass(frozen=True)
class RuntimeSnapshot:
    """单次问答或读取固定使用的 Agent、工具和索引版本。"""

    agent: AdaptiveAgentService
    active: IndexPointer | None
    token: str

    def citation_context(self, source: SourceReference, *, radius: int = 0) -> EvidenceNeighborhood | None:
        """校验引文身份后返回原文与同文档邻块，radius 控制两侧块数。"""

        try:
            neighborhood = self.agent.tools.open(source.chunk_id, before=radius, after=radius)
        except UnknownChunkError:
            return None
        chunk = neighborhood.anchor
        expected = (source.document_id, source.file_name, source.page_start, source.page_end)
        actual = (chunk.document_id, chunk.file_name, chunk.page_start, chunk.page_end)
        return neighborhood if actual == expected else None


class RuntimeServices:
    """首次读取或 active 切换后装配 Agent，单个请求固定使用装配时的快照。"""

    def __init__(
        self,
        settings: Settings,
        *,
        builder: Builder | None = None,
    ) -> None:
        self.settings = settings
        self._builder = builder
        self._snapshot: RuntimeSnapshot | None = None
        # 并发首次读取或发布后的读取只装配一次。
        self._lock = threading.Lock()

    def snapshot(self) -> RuntimeSnapshot:
        """读取已发布指针，候选构建和失败发布不触发重装配。"""

        with self._lock:
            try:
                state = Path(self.settings.index_state_path).read_text(encoding="utf-8")
                active = LifecycleState.model_validate_json(state).active
            except FileNotFoundError:
                active = None
            except (OSError, ValueError) as exc:
                raise ActiveCollectionError("无法读取生命周期状态") from exc
            if self._snapshot is None or self._snapshot.active != active:
                builder = self._builder
                if builder is None:
                    from evidenceflow.runtime import build_agent_service

                    builder = build_agent_service
                # 旧快照由尚在执行的请求持有；空闲时释放缓存引用再加载模型。
                self._snapshot = None
                agent = builder(self.settings)
                identity = getattr(agent.tools, "identity", None)
                signature = identity.index_signature if identity is not None else None
                token = f"{identity.collection_name}:{signature}" if signature is not None else uuid4().hex
                self._snapshot = RuntimeSnapshot(agent, active, token)
            return self._snapshot

    def get(self) -> AdaptiveAgentService:
        """返回当前版本的 Agent。"""

        return self.snapshot().agent

    def snapshot_token(self) -> str:
        """为对话历史绑定索引版本，无签名的旧索引仅在当前进程内定位。"""
        return self.snapshot().token

    def budget_limits(self) -> dict:
        """读取服务默认预算，无需装配模型。"""

        return AgentPreferences().effective(self.settings)

    def read_citation(self, source: SourceReference) -> str | None:
        """读取当前快照内与引文完整身份匹配的片段正文。"""

        match = self.citation_context(source)
        return match.anchor.content if match is not None else None

    def citation_context(self, source: SourceReference, *, radius: int = 0) -> EvidenceNeighborhood | None:
        """从当前版本校验并读取引用及邻块。"""

        return self.snapshot().citation_context(source, radius=radius)


def ask_question(
    services: RuntimeServices, question: str, *, on_event: Callable[[TraceEvent], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
    document_id: str | None = None,
    recent: Sequence[RememberedTurn] = (),
    force_search: bool = False,
    preferences: AgentPreferences | None = None,
) -> WebUIResult:
    """绑定执行和引文所用的快照，可指定 document_id 限定整轮召回。"""

    text = question.strip() if isinstance(question, str) else ""
    options = preferences or AgentPreferences()
    limits = options.effective(services.settings)
    if not text:
        return WebUIResult(
            question="",
            trace_status="not_started",
            error="请输入非空问题。",
            limits=limits,
        )

    if document_id is None and text.rstrip("？?。.!！").rstrip().casefold() in _MODEL_QUESTIONS:
        return WebUIResult(
            question=text,
            answer=RAGAnswer(
                query=text, status=AnswerStatus.ANSWERED,
                answer=f"当前工作台配置的问答模型是 {services.settings.llm_model}。",
                sources=[], retrieved_chunks=[],
            ),
            trace_status="completed",
            limits=limits,
        )

    recorder = RunTraceRecorder(on_event=on_event, tool_details=True)
    recorder.start_run(text)
    started = time.perf_counter()
    answer: RAGAnswer | None = None
    agent: AdaptiveAgentAnswer | None = None
    error: str | None = None
    stopped = False
    snapshot_token: str | None = None
    resolved = text
    try:
        snapshot = services.snapshot()
        service = snapshot.agent
        snapshot_token = snapshot.token
        previous = recent[-1] if recent else None
        sources = [SourceReference.model_validate(item) for item in previous["sources"]] if previous else []
        inferred = None
        if previous and (document_id is None or all(source.document_id == document_id for source in sources)):
            resolved = follow_up_query(text, sources, previous["resolved_query"])
            if resolved != text and len({source.document_id for source in sources}) == 1:
                inferred = sources[0].document_id
        if document_id is None and inferred is not None:
            try:
                service.tools.document_chunk(inferred, 0)
            except UnknownChunkError:
                pass  # 文档已从当前索引移除，仍可按题目名称搜索。
            else:
                document_id = inferred
        if document_id is not None:
            first = service.tools.document_chunk(document_id, 0)
            safe_record(recorder, "agent.scope.selected", document_id=document_id, file_name=first.file_name)
        # trace_scope 用 ContextVar 绑定当前线程/协程的 recorder；共享组件保持无 recorder 状态，
        # 因此并发请求不会串 trace。
        with trace_scope(recorder):
            if cancelled is not None and cancelled():
                raise RunCancelled()
            kwargs = {}
            if cancelled is not None:
                kwargs["cancelled"] = cancelled
            if document_id is not None:
                kwargs["document_id"] = document_id
            if preferences and any(value is not None for value in
                                   (preferences.searches, preferences.opens, preferences.deadline_seconds)):
                kwargs["run_limits"] = options.run_limits(limits)
            if options.reuse_evidence and not force_search:
                for turn in reversed(recent):
                    if (turn["snapshot"], turn["document_id"], turn["settings"]) != (snapshot.token, document_id, limits):
                        continue
                    if normalized_question(turn["resolved_query"]) != normalized_question(resolved):
                        continue
                    refs = [SourceReference.model_validate(item) for item in turn["sources"]]
                    matches = [snapshot.citation_context(source) for source in refs]
                    if matches and all(matches):
                        safe_record(recorder, "run.started")
                        safe_record(recorder, "agent.memory.reused", passages=[passage_identity(match.anchor) for match in matches])
                        safe_record(recorder, "run.completed")
                        answer = RAGAnswer(query=text, status=AnswerStatus.ANSWERED, answer=turn["answer"],
                                           sources=refs, retrieved_chunks=[])
                        break
                if answer is None and previous and resolved != text and previous["snapshot"] == snapshot.token:
                    matches = [snapshot.citation_context(source) for source in sources[:3]
                               if document_id is None or source.document_id == document_id]
                    remembered = [match.anchor for match in matches if match is not None]
                    if remembered:
                        kwargs["remembered"] = remembered
            if answer is None:
                agent = service.run(resolved, **kwargs)
                answer = agent.answer
    except RunCancelled:
        stopped = True
    except UnknownChunkError:
        error = "选中的文档当前不可检索，请返回文档库重新选择。"
    except Exception as exc:  # noqa: BLE001 - 页面需要稳定提示，而不是把 traceback 抛给用户
        error = describe_error(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    trace = _finish_trace(recorder)
    events = tuple(recorder.events)
    if trace is not None:
        trace_status = trace.status
    else:
        trace_status = "failed" if error is not None else "incomplete"
    return WebUIResult(
        question=text,
        answer=answer,
        agent=agent,
        trace=trace,
        events=events,
        trace_status=trace_status,
        error=error,
        elapsed_ms=elapsed_ms,
        limits=limits,
        cancelled=stopped,
        snapshot=snapshot_token,
        resolved_query=resolved,
        document_id=document_id,
    )


def _finish_trace(recorder: RunTraceRecorder) -> RunTrace | None:
    """尽力产出 RunTrace；装配失败等中断场景返回 None，页面仍展示已有事件。"""

    try:
        return recorder.finish()
    except ValueError:
        return None


# ---------- Markdown 渲染 ----------


def _md_cell(value: object) -> str:
    """单元格内容压成单行并转义竖线，避免破坏 Markdown 表格。"""

    return str(value).replace("|", "\\|").replace("\n", " ")


def _number(value: object, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _page_text(page_start: int | None, page_end: int | None) -> str:
    if page_start is None:
        return "-"
    if page_start == page_end:
        return str(page_start)
    return f"{page_start}-{page_end}"


def render_answer(result: WebUIResult) -> str:
    """对话正文只显示可读答案，有拒答或故障时补充对应状态。"""

    if result.error is not None:
        return f"> **处理失败**：{_md_cell(result.error)}"
    if result.answer is None:
        return "_没有可展示的回答。_"
    if result.answer.status is AnswerStatus.INSUFFICIENT_CONTEXT:
        return f"**证据不足，已拒答**\n\n{result.answer.answer}"
    return result.answer.answer


def render_sources(result: WebUIResult) -> str:
    """引用面板：只列回答中真实出现的标签，不把候选或内部 ID 当作引用。"""

    lines = ["### 引用来源", ""]
    if result.answer is None:
        lines.append("没有可用的引用信息。")
        return "\n".join(lines)
    sources = result.answer.sources
    if not sources:
        lines.append("本次回答没有实际引用来源。")
        return "\n".join(lines)
    lines += ["| 标签 | 文件 | 页码 |", "| --- | --- | --- |"]
    for source in sources:
        lines.append(
            f"| {_md_cell(source.source_label)} | {_md_cell(source.file_name)} | "
            f"{_page_text(source.page_start, source.page_end)} |"
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class ProgressStep:
    """实时面板中的一步，时长仅在相应公开事件记录了时长时提供。"""

    title: str
    phase: str
    detail: str
    duration_ms: float | None


def evidence_actions(events: Sequence[TraceEvent], *, status: str = "running") -> list[dict]:
    """从工具的公开输入与结果生成时间线，不读取模型的自由文本理由。"""

    rows: list[dict] = []
    searches: dict[int, dict] = {}
    opens: dict[int, dict] = {}
    scope_names: dict[str, str] = {}
    for event in events:
        name, data = event.name, event.payload
        if name == "agent.scope.selected":
            scope_names[data["document_id"]] = data["file_name"]
        elif name == "agent.search.started":
            index = data.get("search_index", 1)
            row = {"kind": "search", "index": index, "query": data.get("query"),
                   "document_id": data.get("document_id"),
                   "file_name": scope_names.get(data.get("document_id")), "phase": "started"}
            searches[index] = row
            rows.append(row)
        elif name == "retrieval.branch.planned" and searches:
            searches[next(reversed(searches))]["branches"] = data.get("queries", [])
        elif name == "agent.search.completed" and (row := searches.get(data.get("search_index"))):
            row.update(phase="completed", duration_ms=event.duration_ms,
                       hit_count=data.get("hit_count", 0), passages=data.get("passages", []))
        elif name == "agent.open.started":
            row = {"kind": "open", "phase": "started"}
            opens[data.get("open_index", 1)] = row
            rows.append(row)
        elif name == "agent.open.completed" and (row := opens.get(data.get("open_index"))):
            row.update(phase="completed", duration_ms=event.duration_ms,
                       anchor=data.get("anchor"), passages=data.get("passages", []))
        elif name == "agent.memory.reused":
            rows.append({"kind": "reuse", "phase": "completed", "passages": data.get("passages", [])})
        elif name == "context.completed":
            rows.append({"kind": "context", "phase": "completed", "passages": data.get("entries", []),
                         "tokens": data.get("total_tokens"), "skipped": data.get("skipped_count")})
    if status != "running":
        for row in rows:
            if row["phase"] == "started":
                row["phase"] = "interrupted"
    return rows


def progress_steps(events: Sequence[TraceEvent], *, status: str = "running") -> list[ProgressStep]:
    """由同一轮公开事件生成有序步骤，终态将未完成的步骤标为中断。"""

    steps: dict[tuple[str, int], ProgressStep] = {}
    titles = {
        "agent.search": "定向补搜",
        "agent.decision": "判断下一步",
        "agent.assessment": "检查要点覆盖",
        "agent.open": "打开相邻原文",
        "context": "选入回答证据",
        "generation": "生成回答",
        "citation_validation": "核对引文",
    }
    indices = {"agent.search": "search_index", "agent.decision": "decision_index", "agent.open": "open_index"}
    latest_decision = None
    for event in events:
        stage, _, phase = event.name.rpartition(".")
        if event.name == "agent.adaptive.action_rejected" and latest_decision in steps:
            detail = "动作修复失败" if event.payload.get("repair") else "动作需要重试"
            steps[latest_decision] = replace(steps[latest_decision], phase="interrupted", detail=detail)
        if stage not in titles or phase not in {"started", "completed"}:
            continue
        index = event.payload.get(indices.get(stage, ""), 1)
        key = (stage, index)
        title = titles[stage]
        detail = ""
        if stage == "agent.search" and index == 1:
            title = "首轮检索"
        if stage == "agent.decision":
            latest_decision = key
        if "hit_count" in event.payload:
            detail = f"{event.payload['hit_count']} 个片段"
        if "included_count" in event.payload:
            detail = f"采用 {event.payload['included_count']} 段"
        if stage == "citation_validation":
            detail = f"{event.payload.get('source_count', 0)} 条引文"
        if stage == "agent.assessment":
            covered = event.payload.get("covered_count", 0)
            required = event.payload.get("required_count", 0)
            detail = f"覆盖 {covered}/{required} 个要点"
            index = sum(step[0] == stage for step in steps) + 1
            key = (stage, index)
        steps[key] = ProgressStep(title, phase, detail, event.duration_ms)
    return [
        replace(step, phase="interrupted", detail="中断")
        if status != "running" and step.phase == "started" else step
        for step in steps.values()
    ]


def render_evidence(result: WebUIResult) -> str:
    """用最终结果标明证据去向；候选被检索到并不代表实际用于生成或引用。"""
    if result.agent is None or not result.agent.evidence:
        return ""
    lines = ["| 证据块 | 来源 | 进入上下文 | 被引用 |", "| --- | --- | --- | --- |"]
    for entry in result.agent.evidence:
        origin = f"检索 {entry.search_index}" if entry.search_index is not None else "相邻原文"
        lines.append(
            f"| `{_md_cell(entry.chunk_id)}` | {origin} | "
            f"{'是' if entry.in_generation_context else '—'} | {'是' if entry.cited else '—'} |"
        )
    return "\n".join(lines)


def render_trace(result: WebUIResult) -> str:
    """Trace 面板：阶段时间线、候选/重排明细、Agent 动作与预算；不含提示词或私有推理。"""

    events = result.events
    lines = [
        "### 运行 Trace（只读）",
        "",
        "> 仅展示 RunTrace 已记录的事件、计数与分数；不包含提示词、模型原文或私有推理。",
        "",
    ]
    if result.trace is not None:
        lines += [
            f"- Run ID：`{result.trace.run_id}`",
            f"- 状态：{_TRACE_STATUS_LABELS.get(result.trace.status, result.trace.status)}"
            f"（`{result.trace.status}`）",
        ]
    else:
        lines.append(
            f"- 状态：{_TRACE_STATUS_LABELS.get(result.trace_status, result.trace_status)}"
            f"（`{result.trace_status}`）"
        )
    lines += [
        f"- 事件数：{len(events)}",
        f"- 请求耗时：{result.elapsed_ms:.0f} ms",
    ]
    if not events:
        lines += ["", "_本次请求没有产生 trace 事件。_"]
        return "\n".join(lines)

    lines += [
        "",
        "#### 阶段时间线",
        "",
        "| # | 事件 | 相对耗时 (ms) | 时长 (ms) | 摘要 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for event in events:
        duration = "-" if event.duration_ms is None else f"{event.duration_ms:.0f}"
        lines.append(
            f"| {event.seq} | `{event.name}` | {event.elapsed_ms:.0f} | {duration} | "
            f"{_md_cell(_stage_summary(event))} |"
        )
    lines += _candidate_sections(events)
    lines += _agent_section(result)
    return "\n".join(lines)


def _stage_summary(event: TraceEvent) -> str:
    """每个事件压成一行摘要；只读取 §12.5 白名单字段，不推断模型内部过程。"""

    payload = event.payload
    name = event.name
    if name == "retrieval.dense.completed":
        return f"Dense 候选 {payload.get('candidate_count', 0)} 条"
    if name == "rerank.completed":
        return (
            f"重排 {payload.get('input_count')} → {payload.get('output_count')}，"
            f"截断 pair {payload.get('truncated_pair_count')}"
        )
    if name == "retrieval.branch.planned":
        return (
            f"子句 {payload.get('clause_count')} 个，BM25 子句 "
            f"{payload.get('bm25_clause_count')} 个，原查询深度 {payload.get('original_depth')}"
        )
    if name == "retrieval.branch.completed":
        return (
            f"重排 pair {payload.get('pair_count')}，输出 {payload.get('output_count')}，"
            f"分支 {len(payload.get('branches', []))} 个"
        )
    if name == "context.completed":
        return (
            f"纳入 {payload.get('included_count')} / 跳过 {payload.get('skipped_count')}，"
            f"tokens {payload.get('total_tokens')}"
        )
    if name == "generation.completed":
        repaired = "是" if payload.get("repair_attempted") else "否"
        return (
            f"status={payload.get('status')}，调用 LLM="
            f"{'是' if payload.get('llm_called') else '否'}，格式修复={repaired}"
        )
    if name == "citation_validation.completed":
        return (
            f"有效标签 {payload.get('valid_labels')}，无效 {payload.get('invalid_labels')}，"
            f"来源 {payload.get('source_count')}"
        )
    if name == "run.failed":
        return f"error_type={payload.get('error_type')}，stage={payload.get('stage')}"
    if name == "agent.adaptive.terminated":
        return (
            f"reason={payload.get('reason')}，support={payload.get('support')}，"
            f"searches={payload.get('searches')}，opens={payload.get('opens')}，"
            f"llm_calls={payload.get('llm_calls')}，rerank_pairs={payload.get('rerank_pairs')}"
        )
    if name == "agent.adaptive.action_rejected":
        return f"code={payload.get('code')}"
    if name == "agent.search.started":
        return f"search_index={payload.get('search_index')}"
    # 兜底只显示标量字段，列表/字典折叠成计数，避免把大对象塞进表格。
    parts = []
    for key, value in payload.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            parts.append(f"{key}={value}")
        elif isinstance(value, (list, tuple, dict)):
            parts.append(f"{key}[{len(value)}]")
    return "，".join(parts) if parts else "-"


def _candidate_sections(events: Sequence[TraceEvent]) -> list[str]:
    """按检索顺序展示每轮候选和重排，子句计划与同轮结果一起显示。"""

    lines: list[str] = []
    dense = planned = None
    search = 0
    for event in events:
        if event.name in {"retrieval.dense.started", "retrieval.branch.planned"}:
            search += 1
            lines += ["", f"#### 第 {search} 轮检索"]
        if event.name == "retrieval.dense.completed":
            dense = event
        elif event.name == "rerank.completed":
            lines += _dense_details(dense, event)
            dense = None
        elif event.name == "retrieval.branch.planned":
            planned = event
        elif event.name == "retrieval.branch.completed":
            lines += _branch_details(planned, event)
            planned = None
    # 失败可能发生在候选召回或分支计划之后，仍展示已经记录的部分。
    if dense is not None:
        lines += _dense_details(dense, None)
    if planned is not None:
        lines += _branch_details(planned, None)
    return lines


def _dense_details(dense: TraceEvent | None, rerank: TraceEvent | None) -> list[str]:
    """展示一轮全局 Dense 与重排结果。"""

    lines: list[str] = []
    if dense is not None:
        lines += [
            "",
            "##### Dense 候选（Top-K）",
            "",
            "| 名次 | chunk_id | 相似度 |",
            "| --- | --- | --- |",
        ]
        for item in dense.payload.get("candidates", []):
            lines.append(
                f"| {item.get('rank')} | `{_md_cell(item.get('chunk_id'))}` | "
                f"{_number(item.get('similarity_score'))} |"
            )

    if rerank is not None:
        lines += [
            "",
            "##### 重排结果",
            "",
            "| 名次 | chunk_id | 候选名次 | 重排分 |",
            "| --- | --- | --- | --- |",
        ]
        for item in rerank.payload.get("results", []):
            lines.append(
                f"| {item.get('rank')} | `{_md_cell(item.get('chunk_id'))}` | "
                f"{item.get('candidate_rank')} | {_number(item.get('reranker_score'))} |"
            )

    return lines


def _branch_details(planned: TraceEvent | None, completed: TraceEvent | None) -> list[str]:
    """展示一轮子句分支的召回配置和合并后的候选。"""

    lines: list[str] = []
    if planned is not None or completed is not None:
        lines += ["", "##### 子句分支检索", ""]
        if planned is not None:
            lines.append(
                f"- 子句 {planned.payload.get('clause_count')} 个，BM25 子句 "
                f"{planned.payload.get('bm25_clause_count')} 个，原查询深度 "
                f"{planned.payload.get('original_depth')}"
            )
        if completed is not None:
            lines += [
                f"- 重排 pair {completed.payload.get('pair_count')}，输出 "
                f"{completed.payload.get('output_count')}",
                "",
                "| 分支 | 类型 | Dense 候选 | BM25 候选 |",
                "| --- | --- | --- | --- |",
            ]
            for branch in completed.payload.get("branches", []):
                lines.append(
                    f"| {branch.get('branch')} | {branch.get('kind')} | "
                    f"{branch.get('dense_count')} | {branch.get('bm25_count')} |"
                )
            lines += [
                "",
                "| 名次 | chunk_id | 分支 | 候选名次 | 重排分 |",
                "| --- | --- | --- | --- | --- |",
            ]
            for item in completed.payload.get("results", []):
                lines.append(
                    f"| {item.get('rank')} | `{_md_cell(item.get('chunk_id'))}` | "
                    f"{item.get('branch')} | {item.get('candidate_rank')} | "
                    f"{_number(item.get('reranker_score'))} |"
                )
    return lines


def _budget_parts(result: WebUIResult) -> list[str]:
    agent = result.agent
    if agent is None:
        return []
    used = [
        ("llm_calls", agent.llm_calls),
        ("rerank_pairs", agent.rerank_pairs),
        ("searches", agent.searches),
        ("opens", agent.opens),
    ]
    parts = []
    for key, value in used:
        label = _BUDGET_LABELS.get(key, key)
        cap = result.limits.get(key)
        parts.append(f"{label} {value}/{cap}" if cap is not None else f"{label} {value}")
    return parts


def _agent_section(result: WebUIResult) -> list[str]:
    """按执行顺序显示工具动作，并标出预算用量与终止原因。"""

    agent = result.agent
    if agent is None:
        return []
    lines = ["", "#### Agent 动作、预算与终止", ""]
    if agent.actions:
        lines.append("- 动作序列：" + " → ".join(f"`{action}`" for action in agent.actions))
    budget = _budget_parts(result)
    if budget:
        lines.append("- 预算使用：" + "，".join(budget))
    termination = agent.termination_reason
    lines.append(f"- 终止原因：{_TERMINATION_LABELS.get(termination, termination)}（`{termination}`）")
    return lines
