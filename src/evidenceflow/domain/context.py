"""LLM Context domain models constrained by the Token budget.

A Context entry records where its Chunk came from: a production ``search`` hit (with its real
retrieval rank) or an opened neighbourhood (with no retrieval rank at all). Opened neighbours never
carry a similarity or reranker score, because no retrieval ever scored them (PROJECT_SPEC §10.3).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceflow.domain.answer import SourceReference
from evidenceflow.domain.chunk import DocumentChunk

ContextOrigin = Literal["retrieval", "opened_neighborhood", "conversation_memory"]


class ContextEvidence(BaseModel):
    """One candidate handed to the Context builder, with an explicit evidence origin.

    ``retrieval`` requires a real ``retrieval_rank`` from the production retriever.
    ``opened_neighborhood`` must not carry a rank or a search index: those would be invented scores
    or positions for a Chunk that was never retrieved.
    """

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk = Field(description="要被完整纳入 Context 的真实 Chunk。")
    origin: ContextOrigin = Field(
        default="retrieval",
        description="证据来自检索、邻域阅读或已经核对的会话原文。",
    )
    search_index: int | None = Field(
        default=None,
        ge=1,
        description="adaptive 路径中第几次真实 search 命中该 Chunk；普通构建为 None。",
    )
    retrieval_rank: int | None = Field(
        default=None,
        ge=1,
        description="该 Chunk 的真实检索名次；opened 邻域没有名次，为 None。",
    )

    @model_validator(mode="after")
    def validate_origin(self) -> "ContextEvidence":
        """Reject a retrieval entry without a rank, and an opened entry with invented ranks."""

        if self.origin == "retrieval":
            if self.retrieval_rank is None:
                raise ValueError("retrieval evidence requires retrieval_rank")
        elif self.retrieval_rank is not None or self.search_index is not None:
            raise ValueError(
                f"{self.origin} evidence must not carry retrieval_rank or search_index"
            )
        return self


class ContextEntry(BaseModel):
    """Records one Chunk that was fully included in the Context together with its source mapping."""

    source_label: str = Field(
        pattern=r"^\[S[1-9]\d*\]$",
        description="按实际纳入顺序分配的连续临时来源标签。",
    )
    retrieval_rank: int | None = Field(
        default=None,
        ge=1,
        description="该 Chunk 的原始融合检索名次；opened 邻域没有名次。",
    )
    chunk_id: str = Field(min_length=1, description="被完整纳入的 Chunk 稳定身份。")
    content: str = Field(description="未经截断或改写的完整 Chunk 正文。")
    source_reference: SourceReference = Field(
        description="与当前标签及 Chunk 一一对应的精简来源映射。"
    )
    token_count: int = Field(
        ge=1,
        description="标签、来源头和正文组成的单个完整渲染条目 Token 数。",
    )
    origin: ContextOrigin = Field(
        default="retrieval",
        description="该条目的证据来源；opened_neighborhood 不携带任何伪造检索分数。",
    )
    search_index: int | None = Field(
        default=None,
        ge=1,
        description="adaptive 路径中第几次真实 search 命中该 Chunk。",
    )

    @model_validator(mode="after")
    def validate_source_mapping(self) -> "ContextEntry":
        """Guarantees entry identity, label, and public source mapping do not drift."""

        if self.source_reference.source_label != self.source_label:
            raise ValueError("source_reference label must match context entry label")
        if self.source_reference.chunk_id != self.chunk_id:
            raise ValueError("source_reference chunk_id must match context entry chunk_id")
        if self.origin == "retrieval":
            if self.retrieval_rank is None:
                raise ValueError("retrieval context entry requires retrieval_rank")
        elif self.retrieval_rank is not None or self.search_index is not None:
            raise ValueError(
                f"{self.origin} context entry must not carry retrieval_rank or search_index"
            )
        return self


class ContextSupport(BaseModel):
    """已选入 Context 的一条支持原文及生成阶段可用的引用标签。"""

    model_config = ConfigDict(frozen=True)
    source_label: str
    quote: str


class ContextRequirement(BaseModel):
    """实际生成上下文中的要点覆盖；是模型支持判断经预算选择后的结果。"""

    model_config = ConfigDict(frozen=True)
    key: str
    text: str
    status: Literal["covered", "partial", "missing", "budget_excluded"]
    missing_detail: str | None = Field(
        default=None, description="部分支持仍缺的细节，供生成阶段明确回答边界。"
    )
    evidence: tuple[ContextSupport, ...] = Field(
        default=(), description="一个完整支持组合；标签对应本次生成使用的 Context。",
    )


class BuiltContext(BaseModel):
    """Context that can be handed straight to the Generator, plus deterministic statistics."""

    context_text: str = Field(description="由已纳入完整条目按顺序拼接的 LLM Context。")
    entries: list[ContextEntry] = Field(description="按实际纳入顺序排列的来源条目。")
    total_tokens: int = Field(
        ge=0,
        description="完整 context_text 使用注入 Tokenizer 计算的实际 Token 数。",
    )
    candidate_count: int = Field(ge=0, description="本次检查的检索候选总数。")
    included_count: int = Field(ge=0, description="完整纳入 Context 的候选数。")
    skipped_count: int = Field(
        ge=0,
        description="未纳入的候选数；顺序构建时是超预算跳过，需求选择时也包含未选中的候选。",
    )
    requirements: tuple[ContextRequirement, ...] = Field(
        default=(), description="Agent 的要点覆盖说明；普通 RAG 无此评估，保持为空。"
    )

    @model_validator(mode="after")
    def validate_counts(self) -> "BuiltContext":
        """Guarantees the aggregate counters agree with the actual number of entries."""

        if self.included_count != len(self.entries):
            raise ValueError("included_count must match entries length")
        if self.candidate_count != self.included_count + self.skipped_count:
            raise ValueError("candidate_count must equal included_count plus skipped_count")
        if not self.entries and (self.context_text or self.total_tokens != 0):
            raise ValueError("empty entries require empty context_text and zero total_tokens")
        return self
