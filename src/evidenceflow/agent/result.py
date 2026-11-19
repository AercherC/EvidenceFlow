"""生产证据 Agent 与固定评测路径共享的结果合同，不依赖任何编排服务。"""

from typing import Literal
from pydantic import BaseModel, ConfigDict
from evidenceflow.domain.answer import RAGAnswer
from evidenceflow.agent.verify import Verification

TerminationReason = Literal[
    "answered", "insufficient_context", "insufficient_evidence", "budget_exhausted", "fallback",
]
Support = Literal["full", "partial", "none", "unassessed"]


class AgentAnswer(BaseModel):
    """共享的答案与运行元数据；support 独立于回答状态，固定评测路径保留旧字段。"""

    model_config = ConfigDict(frozen=True)

    answer: RAGAnswer
    mode: Literal["single", "multi"] | None
    clauses: tuple[str, ...]
    support: Support
    termination_reason: TerminationReason
    llm_calls: int
    rerank_pairs: int
    rounds: int
    coverage_gaps: tuple[str, ...] = ()
    verification: Verification | None = None
    fallback_error: str | None = None
    searches: int = 0
    opens: int = 0
    actions: tuple[str, ...] = ()
