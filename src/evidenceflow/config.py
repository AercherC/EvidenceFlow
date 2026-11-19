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
        description="Exact Dense 送入 Reranker 的固定候选数量。默认 50（原 20，rag_e2e_v1 实测）。",
    )
    reranker_use_document_context: bool = Field(
        default=False,
        description="重排时把文档名和章节与正文一起评分。候选池、引用原文和重排对数不变。",
    )
    reranker_dense_rrf_weight: float = Field(
        default=0.0, ge=0, le=1, allow_inf_nan=False,
        description="基础 Dense 路径中 Dense 排名的 RRF 权重；0 沿用模型重排，候选池与对数不变。",
    )
    reranker_final_k: int = Field(
        default=8,
        gt=0,
        description=(
            "Learned Reranker 最终交给 Context 的最大候选数量。默认 8（原 6）。"
            "提为 8 的实测依据（207 题，配对 bootstrap）：k=6 时 4096 token 预算从不触底"
            "（207/207 skipped=0），说明 6 是未经检验的保守默认值而非预算所迫；"
            "P0 口径 6→8 为 152→159（+7，CI [+2,+12]，p=0.0012），"
            "W 口径 161→167（+6，CI [+2,+11]，p=0.0028，仅 1/207 触底）；"
            "逐级 6→30 验证提高 k 从不丢失已入包 chunk。"
            "收益在 k≈15 饱和（预算截断），并非无条件越多越好。"
        ),
    )
    reranker_batch_size: int = Field(
        default=4,
        gt=0,
        description="Cross-Encoder 单次前向处理的 Query–Passage 对数量。",
    )
    reranker_max_length: int = Field(
        default=512,
        gt=0,
        description=(
            "Query–Passage 联合编码的最大 Token 长度，超出部分由 tokenizer 截断。"
            "上限由模型架构决定，不是可调性能旋钮：bge-reranker-base 是 XLM-RoBERTa，"
            "max_position_embeddings = 514（512 + 2 special tokens），超过会触发 CUDA "
            "scatter-gather 越界。实测 640 会在第一次前向就崩溃，因此默认值保持 512 —— 即便"
            " chunk_size 也是 512、而本参数计的是 query + passage + specials，导致约 11% 的 "
            "chunk 尾部被截断，也不能靠调大本参数解决；要消除截断只能调小 chunk_size 或换"
            "长上下文重排器。"
        ),
    )
    multi_doc_strategy: Literal["off", "clause_branch"] = Field(
        default="clause_branch",
        description=(
            "多子问题查询的检索策略：off = 全部走 Dense Top50 -> Reranker -> Top8；clause_branch = "
            "R2 规则拆出 2–3 个子句时按子句分支检索（W = R2_c25_5en_rr_k8），其余查询不变。"
            "默认 clause_branch：W 已通过两道 gate（161/233 组 vs P0 152，0 context drop，"
            "pairs ≤100，p95 1.91 s），代价是 38 道题答案变化。off 保留为可回滚通道。"
        ),
    )
    clause_branch_pool_k: int = Field(
        default=25,
        gt=0,
        description="clause_branch：每个子句的 Dense 候选数。",
    )
    clause_branch_balance_two_part_queries: bool = Field(
        default=True,
        description="双题项共享重排预算时优先扩大题项候选，可唯一识别的论文题项单独定向。",
    )
    clause_branch_bm25_k: int = Field(
        default=5,
        ge=0,
        description="clause_branch：不含 CJK 字符的子句额外并入的 BM25 候选数；0 = 不建 BM25 索引。",
    )
    clause_branch_single_query_bm25: bool = Field(
        default=False,
        description="常规非 CJK 单句保留 Dense 候选，在整题重排上限内追加正分 BM25 命中，Top-K 不变。",
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
    clause_branch_fusion_order: Literal["clause_first", "original_first", "rank_aware"] = Field(
        default="rank_aware",
        description="每轮优先取整题证据，子问题按本路与整题逆名次轮转；另保留两种历史顺序。",
    )
    clause_branch_paper_manifest: Path | None = Field(
        default=None,
        description="可选语料清单。启用论文候选补充、章节信息和点名英文子句的 MiniLM 重排。",
    )
    clause_branch_file_aliases: bool = Field(
        default=False,
        description="从索引中唯一的 PDF 文件名识别点名查询，复用定向召回与章节重排，适用于未维护手工别名的新资料。",
    )
    clause_branch_parallel_papers: bool = Field(
        default=False,
        description="使用已配置的论文别名，为含明确比较词且未拆句的双论文问答分配同文档候选。",
    )
    agent_llm_calls_max: int = Field(
        default=2,
        ge=1,
        description="固定 Agent 评测对照：每个回答的 LLM 调用上限（生成 + 一次重生成）。",
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
    agent_paper_manifest: Path | None = Field(
        default=None,
        description=(
            "Agent：语料 manifest，用标题冒号前的短名（如 PoinTr）识别问题点名的论文，"
            "做覆盖检查与一次 widen；默认 None = 关闭（A4 离线门槛 +2 组，未达 +3）。"
            "可设为 evals/research_ingestion/academic_corpus_manifest_v1.json 开启。"
        ),
    )
    agent_deadline_seconds: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False,
        description="Agent：单个回答的墙钟时限；超时回退到普通 RAGService 结果。",
    )
    agent_adaptive_llm_calls_max: int = Field(
        default=5,
        ge=3,
        description=(
            "自适应路径：每个回答的 LLM 调用总上限，必须同时覆盖决策调用、生成和一次格式修复；"
            "生成预留固定为 2 次，其余额度供决策（含一次无效动作修复）。"
        ),
    )
    agent_adaptive_decision_calls_max: int = Field(
        default=3,
        ge=1,
        description="自适应路径：决策模型调用上限，含无效动作后的那一次修复。",
    )
    agent_adaptive_searches_max: int = Field(
        default=2,
        ge=1,
        le=3,
        description="自适应路径：含首轮的检索上限；默认允许一次改写，设为 3 可对照两次改写。",
    )
    agent_adaptive_opens_max: int = Field(
        default=2,
        ge=0,
        description="自适应路径：open 工具调用上限；半径固定为 1，模型不能指定。",
    )
    agent_adaptive_targeted_evidence: bool = Field(
        default=False,
        description="候选策略：多子问题摘要保留各分支候选，首轮在现有 open 预算内补读前页续接原文。",
    )
    agent_adaptive_rerank_pairs_max: int = Field(
        default=200,
        gt=0,
        description=(
            "自适应路径：每个回答的 Reranker pair 上界。每次真实 search 按生产 retriever 的"
            "单次 pair 上限记账，因此总成本上界 = search 次数 × 单次上限。"
        ),
    )
    agent_adaptive_deadline_seconds: float = Field(
        default=20.0,
        gt=0,
        allow_inf_nan=False,
        description=(
            "自适应路径：单个回答的墙钟截止时间。LLM 传输和重试使用剩余时间；"
            "已开始的本地计算和同步网络请求无法强制截断。"
        ),
    )
    api_max_concurrent_requests: int = Field(
        default=1, ge=1, description="每个 API 进程同时执行的问答数，从单模型实例串行起步。"
    )
    api_max_queued_requests: int = Field(
        default=4, ge=0, description="每个 API 进程可等待执行的问答数，超出立即返回 503。"
    )
    api_queue_timeout_seconds: float = Field(
        default=5.0, gt=0, allow_inf_nan=False, description="问答的最长排队时间，和 Agent 执行时限分别计时。"
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
