"""工作台只开放可按会话安全调整的 Agent 预算。"""

from pydantic import BaseModel, ConfigDict, Field

from evidenceflow.config import Settings


class AgentPreferences(BaseModel):
    """浏览器传入的会话覆盖项，空值沿用服务端默认。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    searches: int | None = Field(default=None, ge=1, le=3)
    opens: int | None = Field(default=None, ge=0, le=20)
    deadline_seconds: float | None = Field(default=None, ge=0.1, le=600)
    reuse_evidence: bool = True

    def effective(self, settings: Settings) -> dict:
        """公开本轮实际预算。增开检索时为决策和重排留出配套额度。"""

        searches = self.searches if self.searches is not None else settings.agent_adaptive_searches_max
        extra = max(0, searches - settings.agent_adaptive_searches_max)
        pair_limit = settings.reranker_input_k
        if settings.multi_doc_strategy == "clause_branch":
            pair_limit = max(pair_limit, settings.clause_branch_pair_budget)
        return {
            "searches": searches,
            "opens": self.opens if self.opens is not None else settings.agent_adaptive_opens_max,
            "deadline_seconds": self.deadline_seconds if self.deadline_seconds is not None
            else settings.agent_adaptive_deadline_seconds,
            "decisions": settings.agent_adaptive_decision_calls_max + 2 * extra,
            "llm_calls": settings.agent_adaptive_llm_calls_max + 2 * extra,
            "rerank_pairs": max(settings.agent_adaptive_rerank_pairs_max, searches * pair_limit)
            if extra else settings.agent_adaptive_rerank_pairs_max,
            "reranker_input": settings.reranker_input_k,
            "evidence_tokens": settings.max_context_tokens,
            "reuse_evidence": self.reuse_evidence,
        }

    def run_limits(self, effective: dict) -> dict[str, int | float]:
        """转换为本轮预算，不修改运行时缓存中的 Agent 或索引。"""

        keys = {
            "searches": "searches_max", "opens": "opens_max", "deadline_seconds": "deadline_seconds",
            "decisions": "decision_calls_max", "llm_calls": "llm_calls_max", "rerank_pairs": "rerank_pairs_max",
        }
        return {target: effective[source] for source, target in keys.items()}
