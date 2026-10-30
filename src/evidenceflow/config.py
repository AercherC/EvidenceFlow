"""EvidenceFlow application configuration definitions."""

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"

# Legal Chunk strategy values (§9.3) and the list of fixed-identity Collections (§9.4
# identity-isolation guard).
CHUNK_STRATEGIES = ("hybrid", "fixed")
FIXED_IDENTITY_COLLECTIONS = frozenset(
    {
        "ef_research_bge_m3_c512_o64_v1",
        "ef_bge_m3_c512_o64",
    }
)


class Settings(BaseSettings):
    """Load the unified runtime configuration from environment variables and an optional .env file."""

    # Environment variables are matched case-insensitively so that upper-case names such as
    # LLM_API_KEY can map onto the lower-case Python-style attributes below.
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    llm_api_key: str | None = Field(
        default=None,
        description="OpenAI-compatible LLM 的 API Key，仅从运行环境读取。",
    )
    llm_base_url: str = Field(
        default="https://api.deepseek.com",
        description="OpenAI-compatible LLM API 的基础地址；默认 DeepSeek（A3 修订）。",
    )
    llm_model: str = Field(
        default="deepseek-flash",
        description="当前 LLM 服务商提供的模型名称；默认 DeepSeek 的 deepseek-flash（A3 修订）。",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
        allow_inf_nan=False,
        description="Grounded Answer 的采样温度；默认零以提高可复现性。",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        gt=0,
        allow_inf_nan=False,
        description="单次 OpenAI-compatible LLM 请求的超时秒数。",
    )
    llm_max_attempts: int = Field(
        default=2,
        ge=1,
        le=2,
        description="单次 LLM 业务调用的最大尝试次数；仅瞬态错误允许第二次尝试。",
    )
    llm_trust_env: bool = Field(
        default=False,
        description=(
            "LLM HTTP Client 是否读取 HTTP(S)_PROXY 等系统代理环境变量。默认 False，避免本地代理"
            "未运行时截断 TLS 握手；在必须经代理出网的网络中设为 True。"
        ),
    )
    llm_thinking: Literal["auto", "enabled", "disabled"] = Field(
        default="disabled",
        description=(
            "是否向 OpenAI-compatible 服务发送 thinking 开关。auto = 不发送，沿用服务端默认"
            "（DeepSeek 下思考默认开启）；enabled/disabled 显式开或关。默认 disabled 匹配 DeepSeek；"
            "非 DeepSeek 服务必须覆盖为 auto，否则可能返回 400。"
        ),
    )

    embedding_model: str = Field(
        default="BAAI/bge-m3",
        description="文档和查询共同使用的本地 Embedding 模型。",
    )
    source_docs_path: Path = Field(
        default=Path("data/source_docs"),
        description="正式摄取根目录；所有 document_id 均相对此路径稳定生成。",
    )

    chunk_size: int = Field(
        default=512,
        gt=0,
        description="单个 Chunk 允许包含的最大 Token 数，按 Embedding Tokenizer 计算。",
    )
    chunk_overlap: int = Field(
        default=64,
        ge=0,
        description="相邻 Chunk 之间重复保留的 Token 数（仅 fixed 策略使用；hybrid 冻结为 0）。",
    )
    chunk_strategy: Literal["hybrid", "fixed"] = Field(
        default="hybrid",
        description=(
            "Chunk 切分策略（Phase 13F 晋升）：hybrid = 结构解析 + standalone 独立出块 "
            "+ section 聚合（PH1 语义，窗口零重叠）；fixed = 原固定滑动窗口（回滚通道）。"
        ),
    )

    max_context_tokens: int = Field(
        default=4096,
        gt=0,
        description="LLM Context 可容纳完整 Chunk 的最大 Token 预算。",
    )

    reranker_model: str = Field(
        default=DEFAULT_RERANKER_MODEL,
        min_length=1,
        description="正式 Cross-Encoder 模型名称或本地路径。",
    )
    reranker_input_k: int = Field(
        default=50,
        gt=0,
        description="Original Dense ANN 送入 Reranker 的固定候选数量。默认 50（原 20，rag_e2e_v1 实测）。",
    )
    reranker_final_k: int = Field(
        default=6,
        gt=0,
        description="Learned Reranker 最终交给 Context 的最大候选数量。默认 6（原 5，rag_e2e_v1 实测）。",
    )
    reranker_batch_size: int = Field(
        default=4,
        gt=0,
        description="Cross-Encoder 单次前向处理的 Query–Passage 对数量。",
    )
    reranker_max_length: int = Field(
        default=512,
        gt=0,
        description="Query–Passage 联合编码的最大 Token 长度，超出部分由 tokenizer 截断。",
    )
    multi_doc_strategy: Literal["off", "clause_branch"] = Field(
        default="off",
        description=(
            "多子问题查询的检索策略：off = 全部走 Dense Top50 -> Reranker -> Top6；clause_branch = "
            "R2 规则拆出 2–3 个子句时按子句分支检索（W = R2_c25_5en_rr_k8），其余查询不变。"
        ),
    )
    clause_branch_pool_k: int = Field(
        default=25,
        gt=0,
        description="clause_branch：每个子句的 Dense 候选数。",
    )
    clause_branch_bm25_k: int = Field(
        default=5,
        ge=0,
        description="clause_branch：不含 CJK 字符的子句额外并入的 BM25 候选数；0 = 不建 BM25 索引。",
    )
    clause_branch_pair_budget: int = Field(
        default=100,
        gt=0,
        description="clause_branch：每个查询的 Reranker pair 上限；原查询分支深度 = min(reranker_input_k, 上限 − 子句候选总数)。",
    )
    clause_branch_final_k: int = Field(
        default=8,
        gt=0,
        description="clause_branch：子句分支轮转选出的最终候选数（未触发的查询仍为 reranker_final_k）。",
    )
    agent_llm_calls_max: int = Field(
        default=2,
        ge=1,
        description="Agent（ask --agent）：每个回答的 LLM 调用上限（生成 + 一次重生成）。",
    )
    agent_rerank_pairs_max: int = Field(
        default=150,
        gt=0,
        description="Agent：每个回答的 Reranker pair 上限（clause_branch 100 + widen 50）。",
    )
    agent_rounds_max: int = Field(
        default=2,
        ge=1,
        description="Agent：检索轮数上限（首轮 + 一次 widen）。",
    )
    agent_deadline_seconds: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False,
        description="Agent：单个回答的墙钟时限；超时回退到普通 RAGService 结果。",
    )
    model_device: Literal["auto", "cpu", "cuda"] = Field(
        default="auto",
        description="Embedding 与 Reranker 共用的推理设备；auto 优先 CUDA，不可用时回退 CPU。",
    )
    reranker_dtype: str = Field(
        default="float32",
        pattern="^float32$",
        description="Phase 9 已验证的 Reranker 数值类型；低精度留待性能阶段。",
    )

    vector_db_path: Path = Field(
        default=Path("data/chroma"),
        description="本地 Vector Store 持久化数据目录。",
    )
    collection_name: str = Field(
        default="ef_research_bge_m3_hybrid_c512_v2",
        min_length=1,
        description=(
            "按 Embedding 与 Chunk 配置隔离的向量 Collection 名称；仅在不存在 lifecycle 状态文件"
            "（测试、全新环境）时作为回落值。线上以 index_state_path 的 active 指针为准，"
            "显式设置且与 active 不一致会拒绝启动（见 runtime.resolve_active_collection）。"
        ),
    )
    index_state_path: Path = Field(
        default=Path("artifacts/phase15a/index_state.json"),
        description=(
            "Phase 15A 索引生命周期状态文件（active/previous/staging 指针）；"
            "相对路径按当前工作目录解析。文件存在时线上读取的 Collection 由 active 指针决定。"
        ),
    )

    @field_validator("llm_base_url", "llm_model", "reranker_model")
    @classmethod
    def validate_non_blank_llm_setting(cls, value: str) -> str:
        """Reject whitespace-only endpoints or model names while preserving the caller's original value."""

        if not value.strip():
            raise ValueError("LLM base URL and model must not be blank")
        return value

    @model_validator(mode="after")
    def validate_chunk_configuration(self) -> "Settings":
        """Validate Chunk identity isolation and the Reranker layering-depth contract."""

        # When Overlap reaches or exceeds Chunk Size, the next window makes no effective progress.
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        # §9.4 identity isolation: hybrid must not reuse a fixed-identity Collection,
        # because mixing vectors from different chunking semantics into one Collection lets
        # retrieval semantics drift.
        if (
            self.chunk_strategy == "hybrid"
            and self.collection_name in FIXED_IDENTITY_COLLECTIONS
        ):
            raise ValueError(
                "chunk_strategy='hybrid' must not reuse the fixed-identity collection "
                f"{self.collection_name!r}; rebuild into a hybrid-identity collection"
            )
        # The Reranker may only shrink the candidate set; implicit expansion or silently clamping
        # the final depth is forbidden.
        if self.reranker_final_k > self.reranker_input_k:
            raise ValueError("reranker_final_k must not exceed reranker_input_k")
        # A clause pool is reranked in one call, which accepts at most reranker_input_k candidates.
        if (
            self.multi_doc_strategy == "clause_branch"
            and self.clause_branch_pool_k + self.clause_branch_bm25_k > self.reranker_input_k
        ):
            raise ValueError(
                "clause_branch_pool_k + clause_branch_bm25_k must not exceed reranker_input_k"
            )
        return self
