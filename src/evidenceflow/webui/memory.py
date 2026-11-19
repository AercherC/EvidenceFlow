"""按工作台会话持久化有界的问答状态，证据原文按快照重新读取。"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from evidenceflow.domain.answer import SourceReference
from evidenceflow.domain.generation import AnswerStatus

if TYPE_CHECKING:
    from evidenceflow.webui.service import WebUIResult


class RememberedTurn(TypedDict):
    """保留可重新核对的引文身份和最近一轮所属的文档范围。"""

    turn_id: str
    question: str
    resolved_query: str
    answer: str
    snapshot: str
    document_id: str | None
    sources: list[dict]
    settings: dict


class _ThreadState(TypedDict, total=False):
    """仅保存有界的会话状态，原文、模型和运行句柄均留在请求内。"""

    turn_id: str
    recent: list[RememberedTurn]


_FOLLOW_UP = re.compile(
    r"^(?:那(?:么|它|篇|个|这个|该)|那[，, ]|它|该(?:文|模型|方法|模块)|这(?:篇|个|种)|"
    r"其(?:损失|方法|训练|结构|性能|结果)|(?:还能|再|继续)(?:解释|总结|说明|补充)|"
    r"what about\b|and (?:its|their)\b|how does it\b|what (?:does|is) it\b|its\b)",
    re.IGNORECASE,
)


def normalized_question(text: str) -> str:
    """去掉句尾标点和空白差异，保留词语、数字和技术标识本身。"""

    return " ".join(text.strip().rstrip("。？！?!.").strip().casefold().split())


def follow_up_query(query: str, sources: list[SourceReference], previous_query: str) -> str:
    """用已引用文档或上一题消解指代，只附加一条简短线索。"""

    if not _FOLLOW_UP.match(query) or not sources:
        return query
    names = dict.fromkeys(source.file_name for source in sources)
    topic = next(iter(names)) if len(names) == 1 else previous_query[:240]
    return f"关于 {topic}，{query}"


class ConversationMemory:
    """LangGraph 在同步问答线程中读取和提交 SQLite 会话检查点。"""

    def __init__(self, path: Path) -> None:
        self.path = path

    def run(
        self, thread_id: str, turn_id: str, execute: Callable[[list[RememberedTurn]], WebUIResult],
    ) -> WebUIResult:
        """仅在回答有已校验引文时写入记忆，失败和取消保留原会话状态。"""

        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.graph import END, START, StateGraph
        from langsmith import tracing_context

        result: WebUIResult | None = None

        def answer_node(state: _ThreadState) -> _ThreadState:
            """把历史交给本轮问答。成功后更新最近四轮，不保存检索正文。"""
            nonlocal result
            recent = state.get("recent", [])
            result = execute(recent)
            answer = result.answer
            if answer is None or answer.status != AnswerStatus.ANSWERED or not answer.sources or not result.snapshot:
                return {}
            item: RememberedTurn = {
                "turn_id": turn_id,
                "question": result.question,
                "resolved_query": result.resolved_query or result.question,
                "answer": answer.answer,
                "snapshot": result.snapshot,
                "document_id": result.document_id,
                "sources": [source.model_dump(mode="json") for source in answer.sources],
                "settings": dict(result.limits),
            }
            return {"recent": [*(turn for turn in recent if turn["turn_id"] != turn_id), item][-4:]}

        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 本地会话仅写入本机检查点，不继承机器上其他项目的 LangSmith 采集设置。
        with tracing_context(enabled=False), SqliteSaver.from_conn_string(str(self.path)) as saver:
            builder = StateGraph(_ThreadState)
            builder.add_node("answer", answer_node)
            builder.add_edge(START, "answer")
            builder.add_edge("answer", END)
            builder.compile(checkpointer=saver).invoke(
                {"turn_id": turn_id}, {"configurable": {"thread_id": thread_id}},
            )
        assert result is not None
        return result

    def delete(self, thread_id: str) -> None:
        """删除工作台对话时，清除它的全部追问检查点。"""

        if not self.path.is_file():
            return
        from langgraph.checkpoint.sqlite import SqliteSaver

        with SqliteSaver.from_conn_string(str(self.path)) as saver:
            saver.delete_thread(thread_id)
