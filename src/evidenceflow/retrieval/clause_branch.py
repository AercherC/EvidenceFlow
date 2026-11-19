"""将查询计划转换为有预算的候选分支，再独立重排和融合。"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from evidenceflow.domain.chunk import DocumentChunk, RerankedRetrievedChunk, RetrievedChunk
from evidenceflow.domain.question import QuestionPlan, plan_question, scoped_training_question
from evidenceflow.domain.question import split_clauses as split_clauses
from evidenceflow.retrieval.clause_bm25 import BM25Index
from evidenceflow.retrieval.reranker import DenseRerankerRetriever, LearnedReranker, RerankerError
from evidenceflow.retrieval.retriever import DenseRetriever, InvalidQueryError
from evidenceflow.tracing import TraceRecorder, resolve_recorder, safe_record

logger = logging.getLogger(__name__)

# 比较两篇论文的显式分配词。英语 each 另限于紧跟在论文名之后，避免 each point 误路由。
_PARALLEL_PAPER_MARKER = re.compile(r"分别|各自|\brespectively\b", re.IGNORECASE)
_AFTER_PAPER_EACH = re.compile(r"\s+each\b", re.IGNORECASE)
# 只把直接限定题项的论文名用于硬范围，例如 “PF-Net's decoder” 或 “PF-Net 的解码器”。
_PAPER_OWNER = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z][A-Za-z0-9_.-]+)(?:['’]s\b|\s*的)")
# 题项开头的「论文名 + 实验」明确限定来源，引用编号则可用于跨语言词项召回。
_PAPER_EXPERIMENT = re.compile(r"^([A-Za-z][A-Za-z0-9_.-]*)\s*实验")
_REFERENCE_NUMBER = re.compile(r"[\[［]([0-9]{1,4})[\]］]")
# 默认的 25 个 Dense 位置中，最多用 5 个尾部位置补入同论文候选。
PAPER_TAIL_SLOTS = 5
FusionOrder = Literal["clause_first", "original_first", "rank_aware"]


def has_cjk(text: str) -> bool:
    """CJK symbols/ideographs (U+3000–U+9FFF) or full-width forms (U+FF00–U+FFEF)."""

    return any(0x3000 <= ord(ch) <= 0x9FFF or 0xFF00 <= ord(ch) <= 0xFFEF for ch in text)


def paper_passage(chunk: DocumentChunk) -> str:
    """仅向重排模型补充论文名和章节路径，交付给生成器的块保留原文。"""
    title = chunk.metadata.get("title") or Path(chunk.file_name).stem.replace("_", " ")
    heading = chunk.metadata.get("heading_path")
    return "\n".join(part for part in (title, heading, chunk.content) if isinstance(part, str) and part)


def _round_robin_sources(
    branches: Sequence[tuple[int, Sequence[str]]], k: int,
) -> Iterator[tuple[int, str]]:
    """每路取一块并保留实际选中它的分支，重复块不消耗席位。"""
    seen: set[str] = set()
    pending = deque((index, iter(branch)) for index, branch in branches)
    while pending and len(seen) < k:
        index, branch = pending.popleft()
        for chunk_id in branch:
            if chunk_id not in seen:
                seen.add(chunk_id)
                pending.append((index, branch))
                yield index, chunk_id
                break


def round_robin(branches: Sequence[Sequence[str]], k: int) -> list[str]:
    """按分支顺序轮转、去重并截断，供历史评测重放同一取块规则。"""
    return [chunk_id for _, chunk_id in _round_robin_sources(list(enumerate(branches)), k)]


def _rank_aware_sources(
    branches: Sequence[tuple[int, Sequence[str]]], k: int,
) -> Iterator[tuple[int, str]]:
    """每轮保留整题证据，子问题按本路与整题的名次共同决定本轮顺序。"""
    if not branches:
        return
    seen: set[str] = set()
    positions = {index: 0 for index, _ in branches}
    original, original_ranking = branches[0]
    original_weight = {chunk_id: 1 / rank for rank, chunk_id in enumerate(original_ranking, 1)}

    def next_position(index: int, ranking: Sequence[str]) -> int:
        """跳过已取证据，保留候选在重排分支中的原始位置。"""
        position = positions[index]
        while position < len(ranking) and ranking[position] in seen:
            position += 1
        positions[index] = position
        return position

    while len(seen) < k:
        previous_count = len(seen)
        position = next_position(original, original_ranking)
        if position < len(original_ranking):
            chunk_id = original_ranking[position]
            seen.add(chunk_id)
            positions[original] = position + 1
            yield original, chunk_id
        pending = dict(branches[1:])
        while pending and len(seen) < k:
            # 候选只有本轮各路的首个新块。按逆名次合并局部和整题支持，平局保持题序。
            available = [
                (index, position, ranking[position])
                for index, ranking in pending.items()
                if (position := next_position(index, ranking)) < len(ranking)
            ]
            if not available:
                break
            index, position, chunk_id = max(
                available, key=lambda head: 1 / (head[1] + 1) + original_weight.get(head[2], 0),
            )
            seen.add(chunk_id)
            positions[index] = position + 1
            del pending[index]
            yield index, chunk_id
        if len(seen) == previous_count:
            break


@dataclass(frozen=True)
class ClauseSearchResult:
    """保留融合命中、子句命中和按题序确认的论文范围。"""

    hits: list[RerankedRetrievedChunk]
    clause_hits: tuple[tuple[str, ...], ...] = ()
    document_scopes: tuple[str, ...] = ()


@dataclass(frozen=True)
class _BranchInput:
    """单路重排的查询、候选和词项成本；召回规划完成后才能调用重排器。"""

    query: str
    kind: str
    pool: list[RetrievedChunk]
    bm25_count: int = 0
    reranker: LearnedReranker | None = None
    with_paper_context: bool = False
    document_id: str | None = None


@dataclass(frozen=True)
class _Branch:
    """保留分支类型、候选来源数量和未截断的重排结果。"""

    kind: str
    dense_count: int
    bm25_count: int
    ranked: list[RerankedRetrievedChunk]


def _fuse_branches(
    branches: list[_Branch], top_k: int, *, fusion_order: FusionOrder,
) -> tuple[list[RerankedRetrievedChunk], dict[str, tuple[int, RerankedRetrievedChunk]]]:
    """保留实际出块分支的 Dense 名次和重排分数，生成全局连续名次。"""
    by_branch = [{item.chunk.chunk_id: item for item in branch.ranked} for branch in branches]
    order = list(enumerate([list(items) for items in by_branch]))
    if fusion_order != "clause_first":
        order.insert(0, order.pop())
    select = _rank_aware_sources if fusion_order == "rank_aware" else _round_robin_sources
    source = {
        chunk_id: (index, by_branch[index][chunk_id])
        for index, chunk_id in select(order, top_k)
    }
    results = [
        item.model_copy(update={"rank": rank})
        for rank, (_, item) in enumerate(source.values(), start=1)
    ]
    return results, source


class ClauseBranchRetriever:
    """Route multi-clause queries through clause branches; delegate the rest to production retrieval."""

    retrieval_strategy = "clause_branch"

    def __init__(
        self,
        base: DenseRerankerRetriever,
        bm25: BM25Index | None,
        *,
        dense_depth: int,
        clause_pool_k: int = 25,
        bm25_k: int = 5,
        pair_budget: int = 100,
        final_top_k: int = 8,
        fusion_order: FusionOrder = "clause_first",
        trace_recorder: TraceRecorder | None = None,
        paper_aliases: Mapping[str, str] | None = None,
        exact_paper_names: Mapping[str, str] | None = None,
        paper_reranker: LearnedReranker | None = None,
        single_query_bm25: bool = False,
        parallel_papers: bool = False,
        balance_two_part_queries: bool = False,
    ) -> None:
        """dense_depth 覆盖整个索引，便于补入同论文候选并保留其真实 Dense 名次。

        paper_aliases 绑定到 document_id，paper_reranker 仅用于唯一点名的英文子句。
        exact_paper_names 供题面有明确主语的训练任务检索使用，歧义文件名不进入此表。
        fusion_order 只控制最终取块，不改变候选与重排预算。
        single_query_bm25 为常规单句追加词项候选，保留基础 Dense 候选与整题重排上界。
        parallel_papers 为未拆句的明确双论文比较分配同文档候选与整题候选。
        balance_two_part_queries 给双题项扩大各自候选，仍共用整题重排总预算。
        """

        for value, name in (
            (dense_depth, "dense_depth"),
            (clause_pool_k, "clause_pool_k"),
            (pair_budget, "pair_budget"),
            (final_top_k, "final_top_k"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(bm25_k, bool) or not isinstance(bm25_k, int) or bm25_k < 0:
            raise ValueError("bm25_k must be a non-negative integer")
        if bm25_k and bm25 is None:
            raise ValueError("bm25_k > 0 needs a BM25 index")
        if clause_pool_k + bm25_k > base.reranker.max_candidates:
            raise ValueError("a clause pool must fit the reranker's max_candidates")
        if fusion_order not in {"clause_first", "original_first", "rank_aware"}:
            raise ValueError("fusion_order must be clause_first, original_first or rank_aware")
        self.base = base
        self.bm25 = bm25
        self.dense_depth = dense_depth
        self.clause_pool_k = clause_pool_k
        self.bm25_k = bm25_k
        self.pair_budget = pair_budget
        self.final_top_k = final_top_k
        self.fusion_order = fusion_order
        self.trace_recorder = trace_recorder
        self._paper_reranker = paper_reranker
        self.single_query_bm25 = single_query_bm25
        self.parallel_papers = parallel_papers
        self.balance_two_part_queries = balance_two_part_queries
        # 运行时将别名绑定到当前索引的 document_id。优先匹配长名称并排除数据集名。
        self._paper_aliases = {
            alias.casefold(): document_id for alias, document_id in (paper_aliases or {}).items()
        }
        self._exact_paper_names = {**(exact_paper_names or {}), **self._paper_aliases}
        # 用受索引约束的文件全名、首个名称词或显式别名识别题项。碰撞时不猜文档。
        owners: dict[str, set[str]] = defaultdict(set)
        for name, document_id in (exact_paper_names or {}).items():
            for alias in (name, name.partition(" ")[0]):
                owners[self._scope_key(alias)].add(document_id)
        for alias, document_id in self._paper_aliases.items():
            owners[self._scope_key(alias)].add(document_id)
        self._scope_owners = {name: next(iter(ids)) for name, ids in owners.items() if len(ids) == 1}
        aliases = "|".join(re.escape(alias) for alias in sorted(self._paper_aliases, key=len, reverse=True))
        self._paper_names = (
            re.compile(
                r"(?<![A-Za-z0-9_-])(?:" + aliases + r")(?![A-Za-z0-9_-]|\s*(?:数据集|dataset\b))",
                re.I | re.ASCII,
            )
            if aliases
            else None
        )

    # 诊断脚本通过这些只读属性访问基础检索组件和深度。

    @property
    def dense_retriever(self) -> DenseRetriever:
        """The Dense port the wrapped retriever queries (evaluation probes read rankings here)."""

        return self.base.dense_retriever

    @property
    def reranker(self) -> LearnedReranker:
        """The Learned Reranker the wrapped retriever scores with."""

        return self.base.reranker

    @property
    def candidate_pool_k(self) -> int:
        """Candidate pool depth of the untriggered path (the clause path has its own ``clause_pool_k``)."""

        return self.base.candidate_pool_k

    @property
    def reranker_input_k(self) -> int:
        """Reranker input depth of the untriggered path."""

        return self.base.reranker_input_k

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
    ) -> list[RerankedRetrievedChunk]:
        """按查询意图检索，返回融合后的 Top-K。"""
        return self.retrieve_with_branches(query, top_k).hits

    def top_k_for_query(self, query: str) -> int:
        """公开线上默认输出深度，供离线观测复用而不重写路由规则。"""
        return self._default_top_k(plan_question(query))

    def _default_top_k(self, plan: QuestionPlan) -> int:
        return self.final_top_k if len(plan.parts) > 1 else self.base.final_top_k

    def retrieve_with_branches(
        self, query: str, top_k: int | None = None, *, document_scope: str | None = None,
    ) -> ClauseSearchResult:
        """规划候选并融合。补搜可传入首轮已确认的文档身份，改写词不改变范围。"""
        if not isinstance(query, str) or not query.strip():
            raise InvalidQueryError("query must not be empty or whitespace")
        plan = plan_question(query)
        single_query = document_scope is not None or len(plan.parts) == 1
        default_top_k = self.base.final_top_k if single_query else self.final_top_k
        effective_top_k = default_top_k if top_k is None else top_k
        if (
            isinstance(effective_top_k, bool)
            or not isinstance(effective_top_k, int)
            or effective_top_k <= 0
        ):
            raise ValueError("top_k must be a positive integer")

        recorder = resolve_recorder(self.trace_recorder)
        started = time.perf_counter()
        scopes = () if single_query else self._document_scopes(plan)
        if document_scope is not None:
            pool, lexical = self._scoped_pool(query, document_scope, local_rank=True)
            inputs = [_BranchInput(query, "scoped_clause", pool, lexical, with_paper_context=True,
                                   document_id=document_scope)]
        elif single_query:
            inputs = self._single_candidates(query, plan, effective_top_k)
        else:
            inputs = self._multi_candidates(query, plan, scopes)
        if not inputs:
            return ClauseSearchResult(self.base.retrieve(query, top_k))
        has_original = inputs[-1].kind == "original"
        clause_count = len(inputs) - has_original
        safe_record(
            recorder,
            "retrieval.branch.planned",
            clause_count=clause_count,
            bm25_clause_count=sum(bool(branch.bm25_count) for branch in inputs),
            original_depth=len(inputs[-1].pool) if has_original else 0,
            **({"queries": [
                {"query": branch.query, "kind": branch.kind, "document_id": branch.document_id,
                 "file_name": branch.pool[0].chunk.file_name if branch.document_id and branch.pool else None}
                for branch in inputs
            ]} if getattr(recorder, "tool_details", False) else {}),
        )
        branches = [
            self._rerank(
                branch.query, branch.kind, branch.pool, branch.bm25_count,
                reranker=branch.reranker, with_paper_context=branch.with_paper_context,
            )
            for branch in inputs
        ]
        fusion_order = self.fusion_order if has_original and not single_query else "clause_first"
        if (
            self.balance_two_part_queries and clause_count == 2
            and {branch.kind for branch in branches[:clause_count]} == {"clause", "scoped_clause"}
        ):
            # 一项已定向且另一项仍需全库搜索时，先轮到两个题项，再消费整题名额。
            fusion_order = "clause_first"
        results, source = _fuse_branches(branches, effective_top_k, fusion_order=fusion_order)
        # 相同子句也会分别调用模型，成本不能按 query/chunk 去重。
        pair_count = sum(len(branch.ranked) for branch in branches)
        logger.info(
            "retrieval completed",
            extra={
                "retrieval_strategy": self.retrieval_strategy,
                "clause_count": clause_count,
                "reranker_pair_count": pair_count,
                "reranker_output_count": len(results),
                "latency": time.perf_counter() - started,
            },
        )
        if recorder is not None:
            safe_record(
                recorder,
                "retrieval.branch.completed",
                pair_count=pair_count,
                output_count=len(results),
                branches=[
                    {
                        "branch": index,
                        "kind": branch.kind,
                        "dense_count": branch.dense_count,
                        "bm25_count": branch.bm25_count,
                        # 完整分支排名用于评测归因。全局 Dense 排名和融合列表无法反推这些候选。
                        "ranking": [
                            {"chunk_id": item.chunk.chunk_id, "document_id": item.chunk.document_id}
                            for item in branch.ranked
                        ],
                    }
                    for index, branch in enumerate(branches)
                ],
                results=[
                    {
                        "rank": item.rank,
                        "chunk_id": item.chunk.chunk_id,
                        "branch": source[item.chunk.chunk_id][0],
                        "candidate_rank": item.candidate_rank,
                        "reranker_score": item.reranker_score,
                    }
                    for item in results
                ],
            )
        if single_query:
            return ClauseSearchResult(results)
        selected_ids = {item.chunk.chunk_id for item in results}
        clause_hits = tuple(
            tuple(item.chunk.chunk_id for item in branch.ranked if item.chunk.chunk_id in selected_ids)
            for branch in branches[:clause_count]
        )
        return ClauseSearchResult(results, clause_hits, scopes)

    def _multi_candidates(
        self, query: str, plan: QuestionPlan, scopes: tuple[str, ...],
    ) -> list[_BranchInput]:
        """为每个独立目标分配候选，并将剩余配额交给整题分支。"""
        papers = [self._named_paper(part) for part in plan.parts]
        balanced = self.balance_two_part_queries and len(plan.parts) == 2
        part_documents = (
            scopes or tuple(self._clause_document(part) for part in plan.parts)
            if balanced else (None,) * len(plan.parts)
        )
        depth = self.clause_pool_k
        if balanced and (any(part_documents) or not any(map(has_cjk, plan.parts))):
            # 把可借用的整题候选均分给两个题项，给整题分支留至少半个普通候选池。
            # 跨语言且两项都无法定向时保留整题候选深度，利用整题词义消歧。
            extra_depth = (self.pair_budget - self.reranker_input_k // 2) // 2 - self.bm25_k
            depth = max(depth, min(extra_depth, self.reranker_input_k - self.bm25_k))
        inputs = []
        for index, (part, paper) in enumerate(zip(plan.parts, papers, strict=True)):
            if scopes:
                pool, lexical = self._scoped_pool(part, scopes[index], dense_k=depth)
                inputs.append(_BranchInput(part, "scoped_clause", pool, lexical, with_paper_context=True,
                                           document_id=scopes[index]))
                continue
            training = scoped_training_question(part) if plan.coordinated else None
            scoped_paper = self._exact_paper_names.get(training[0].casefold()) if training else None
            if training and scoped_paper:
                # 用题面的实体保持向量召回的指向，重排时只问该论文内待找的细节。
                ranking = self.dense_retriever.retrieve(part, top_k=self.dense_depth)
                pool = [item for item in ranking if item.chunk.document_id == scoped_paper][:depth]
                inputs.append(_BranchInput(training[1], "scoped_clause", pool, document_id=scoped_paper))
            elif document_id := part_documents[index]:
                # 另一题项的来源未知时，已点名的题项仍可在该论文内检索。
                pool, lexical = self._scoped_pool(part, document_id, dense_k=depth)
                inputs.append(_BranchInput(part, "scoped_clause", pool, lexical, with_paper_context=True,
                                           document_id=document_id))
            else:
                pool, lexical = self._clause_pool(part, paper, dense_k=depth)
                inputs.append(_BranchInput(
                    part, "clause", pool, lexical,
                    reranker=self._paper_reranker if paper and not has_cjk(part) else None,
                    with_paper_context=any(papers),
                ))
        remaining = self.pair_budget - sum(len(branch.pool) for branch in inputs)
        if remaining < 0:
            raise ValueError("clause candidates exceed pair_budget")
        if remaining:
            pool = self.dense_retriever.retrieve(query, top_k=min(self.reranker_input_k, remaining))
            inputs.append(_BranchInput(query, "original", pool))
        return inputs

    @staticmethod
    def _scope_key(name: str) -> str:
        """统一文件名中的分隔符，例如 Hyper-PCN 和 hyper_pcn。"""
        return re.sub(r"[\s_-]+", "", name.casefold())

    def _clause_document(self, part: str) -> str | None:
        """只对能唯一识别来源的题项使用硬文档范围。"""
        names = _PAPER_OWNER.findall(part)
        if experiment := _PAPER_EXPERIMENT.match(part):
            names.append(experiment[1])
        documents = {
            document_id for name in names
            if (document_id := self._scope_owners.get(self._scope_key(name))) is not None
        }
        return next(iter(documents)) if len(documents) == 1 else None

    def _document_scopes(self, plan: QuestionPlan) -> tuple[str, ...]:
        """只有多个题项分别指向不同的唯一论文时，才给它们固定范围。"""
        scopes = []
        for part in plan.parts:
            papers = {
                document_id for name in _PAPER_OWNER.findall(part)
                if (document_id := self._scope_owners.get(self._scope_key(name))) is not None
            }
            if len(papers) != 1:
                return ()
            scopes.append(papers.pop())
        return tuple(scopes) if len(set(scopes)) == len(scopes) else ()

    def _scoped_pool(
        self, query: str, document_id: str, *, dense_k: int | None = None, local_rank: bool = False,
    ) -> tuple[list[RetrievedChunk], int]:
        """显式文档查询下推向量过滤，自动定向的题项保留真实全库名次。"""
        dense_k = self.clause_pool_k if dense_k is None else dense_k
        lexical_enabled = self.bm25 is not None and self.bm25_k and not has_cjk(query)
        if local_rank:
            ranking = self.dense_retriever.retrieve(
                query, top_k=self.dense_depth if lexical_enabled else dense_k, document_id=document_id,
            )
        else:
            ranking = [
                hit for hit in self.dense_retriever.retrieve(query, top_k=self.dense_depth)
                if hit.chunk.document_id == document_id
            ]
        dense = ranking[:dense_k]
        if not lexical_enabled:
            return dense, 0
        remainder = {hit.chunk.chunk_id: hit for hit in ranking[dense_k:]}
        if not remainder:
            return dense, 0
        if local_rank:
            # 显式范围也限制词项召回，避免按全库 BM25 排序再丢掉其他文档。
            ids = self.bm25.search(query, self.bm25_k, positive_only=True, allowed_ids=set(remainder))
            return dense + [remainder[chunk_id] for chunk_id in ids], len(ids)
        lexical = [
            remainder[chunk_id] for chunk_id in self.bm25.search(query, self.bm25.size, positive_only=True)
            if chunk_id in remainder
        ][:self.bm25_k]
        return dense + lexical, len(lexical)

    def _single_candidates(self, query: str, plan: QuestionPlan, top_k: int) -> list[_BranchInput]:
        """单目标仅在有明确的论文范围或词项信号时使用专用候选。"""
        base_depth = self.reranker_input_k
        depth = min(base_depth, self.pair_budget)
        parallel = self._parallel_papers(query) if self.parallel_papers else ()
        if parallel:
            # 两篇合用一次全库 Dense 排名及基础重排预算，余额交给整题。
            per_paper = min(self.clause_pool_k, depth // 4)
            if not per_paper:
                return []
            ranking = self.dense_retriever.retrieve(query, top_k=self.dense_depth)
            inputs = [
                _BranchInput(query, "paper",
                             [item for item in ranking if item.chunk.document_id == paper][:per_paper],
                             with_paper_context=True)
                for paper in parallel
            ]
            original_depth = depth - sum(len(branch.pool) for branch in inputs)
            return [*inputs, _BranchInput(query, "original", ranking[:original_depth])]

        paper = self._named_paper(query)
        if paper is not None:
            pool, _ = self._clause_pool(query, paper, dense_k=depth, bm25_k=0)
            return [_BranchInput(
                query, "clause", pool, reranker=self._paper_reranker if not has_cjk(query) else None,
                with_paper_context=True,
            )]

        # 命名题用题面描述召回正分词项，替换 Dense 尾部且不增加单次重排数。
        # 两个以上明确编号可匹配英文原文。沿用词项尾部配额，保留 Dense 前排。
        reference_lookup = len(set(_REFERENCE_NUMBER.findall(query))) > 1
        lexical_query = plan.name_description or (query if reference_lookup else None)
        lexical_slots = (
            self._single_lexical_slots(query, depth, allow_cjk=reference_lookup) if lexical_query else 0
        )
        if lexical_slots:
            if top_k > base_depth:
                raise ValueError("top_k must not exceed reranker_input_k")
            pool, added = self._single_lexical_pool(
                query, depth, lexical_slots, lexical_query=lexical_query,
            )
            return [_BranchInput(query, "name_lookup", pool, added)]
        if self.single_query_bm25 and not has_cjk(query):
            slots = min(self.bm25_k, self.pair_budget - base_depth, self.reranker.max_candidates - base_depth)
            if slots > 0:
                if top_k > base_depth:
                    raise ValueError("top_k must not exceed reranker_input_k")
                pool, added = self._single_lexical_pool(query, base_depth, slots, preserve_dense=True)
                return [_BranchInput(query, "clause", pool, added)]
        return []

    def _named_paper(self, clause: str) -> str | None:
        """仅当子句能唯一定位一篇论文时，返回其索引身份。"""
        if self._paper_names is None:
            return None
        papers = {self._paper_aliases[m.group().casefold()] for m in self._paper_names.finditer(clause)}
        return next(iter(papers)) if len(papers) == 1 else None

    def _parallel_papers(self, query: str) -> tuple[str, ...]:
        """仅对同一句中明确要求分别回答两篇已识别论文的问题保留两个范围。"""
        if self._paper_names is None:
            return ()
        matches = list(self._paper_names.finditer(query))
        papers = tuple(dict.fromkeys(self._paper_aliases[m.group().casefold()] for m in matches))
        if len(papers) != 2:
            return ()
        if _PARALLEL_PAPER_MARKER.search(query) or _AFTER_PAPER_EACH.match(query, matches[-1].end()):
            return papers
        return ()

    def retrieve_for_recovery(self, query: str) -> list[RerankedRetrievedChunk]:
        """单要点补证时，用术语命中替换少量 Dense 尾部位置，重排预算与分数口径不变。"""
        depth = min(self.base.reranker_input_k, self.pair_budget)
        lexical_slots = self._single_lexical_slots(query, depth)
        if not lexical_slots or split_clauses(query) is not None:
            return self.retrieve(query)
        pool, added = self._single_lexical_pool(query, depth, lexical_slots)
        ranked = self._rerank(query, "recovery", pool, added).ranked
        return ranked[:self.base.final_top_k]

    def _single_lexical_slots(self, query: str, depth: int, *, allow_cjk: bool = False) -> int:
        """保持 Dense 前部，且只对当前词项索引适用的查询预留候选位置。"""
        if self.bm25 is None or (not allow_cjk and has_cjk(query)):
            return 0
        return max(0, min(self.bm25_k, depth - self.base.final_top_k))

    def _single_lexical_pool(
        self, query: str, depth: int, lexical_slots: int, *,
        preserve_dense: bool = False, lexical_query: str | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
        """追加正分 BM25 命中时保护完整 Dense 前缀；实验补证仍可在原配额内替换尾部。"""
        assert self.bm25 is not None
        ranking = self.base.dense_retriever.retrieve(query, top_k=self.dense_depth)
        by_id = {item.chunk.chunk_id: item for item in ranking}
        protected_depth = depth if preserve_dense else depth - lexical_slots
        protected = {item.chunk.chunk_id for item in ranking[:protected_depth]}
        lexical = [
            by_id[key] for key in self.bm25.search(lexical_query or query, self.bm25_k, positive_only=True)
            if key not in protected
        ][:lexical_slots]
        lexical_ids = {item.chunk.chunk_id for item in lexical}
        dense = [item for item in ranking if item.chunk.chunk_id not in lexical_ids]
        dense_depth = depth if preserve_dense else depth - len(lexical)
        return dense[:dense_depth] + lexical, len(lexical)

    def _clause_pool(
        self, clause: str, paper: str | None = None, *, dense_k: int | None = None,
        bm25_k: int | None = None,
    ) -> tuple[list[RetrievedChunk], int]:
        """按原配额召回。唯一识别到论文时，仅用该论文的候选替换 Dense 尾部。"""
        dense_k = self.clause_pool_k if dense_k is None else dense_k
        bm25_k = self.bm25_k if bm25_k is None else bm25_k
        use_lexical = bm25_k > 0 and not has_cjk(clause)
        if paper is None and not use_lexical:
            # 纯 Dense 子句只取配额内的块。Exact Dense 仍计算全库相似度，省去其余块的排序和对象构造。
            return self.base.dense_retriever.retrieve(clause, top_k=min(dense_k, self.dense_depth)), 0

        ranking = self.base.dense_retriever.retrieve(clause, top_k=self.dense_depth)
        dense = list(ranking[:dense_k])
        slots = min(PAPER_TAIL_SLOTS, len(dense) // 5)
        lexical = []
        bm25_ids = []
        if use_lexical:
            by_id = {item.chunk.chunk_id: item for item in ranking}
            dense_ids = {item.chunk.chunk_id for item in dense}
            bm25_ids = self.bm25.search(clause, self.bm25.size if paper and slots else bm25_k)
            lexical = [by_id[chunk_id] for chunk_id in bm25_ids[:bm25_k] if chunk_id not in dense_ids]

        if paper and slots:
            scoped = {item.chunk.chunk_id: item for item in ranking if item.chunk.document_id == paper}
            preferred = list(scoped.values())[:dense_k]
            preferred_ids = {item.chunk.chunk_id for item in preferred}
            scoped_bm25 = [scoped[i] for i in bm25_ids if i in scoped and i not in preferred_ids]
            preferred.extend(scoped_bm25[:bm25_k])
            existing_ids = {item.chunk.chunk_id for item in dense + lexical}
            additions = [item for item in preferred if item.chunk.chunk_id not in existing_ids]
            dense[-slots:] = (additions + dense[-slots:])[:slots]
        return dense + lexical, len(lexical)

    def _rerank(
        self,
        branch_query: str,
        kind: str,
        pool: list[RetrievedChunk],
        bm25_count: int,
        *,
        reranker: LearnedReranker | None = None,
        with_paper_context: bool = False,
    ) -> _Branch:
        """用本分支查询重排，保留原块及 Dense 名次。"""

        try:
            chunks = [item.chunk for item in pool]
            # 显式论文分支及独立的重排输入实验共用同一个原文渲染函数。
            passage_text = paper_passage if with_paper_context else self.base.passage_text
            reranked = (reranker or self.base.reranker).rerank(branch_query, chunks, passage_text=passage_text)
        except Exception as exc:
            # As in DenseRerankerRetriever: never fall back to an unreranked order.
            raise RerankerError("learned reranker failed") from exc
        by_id = {item.chunk.chunk_id: item for item in pool}
        ranked = [
            RerankedRetrievedChunk(
                chunk=item.chunk,
                rank=item.rank,
                candidate_rank=by_id[item.chunk.chunk_id].rank,
                similarity_score=by_id[item.chunk.chunk_id].similarity_score,
                reranker_score=item.relevance_score,
            )
            for item in reranked
        ]
        return _Branch(kind, len(pool) - bm25_count, bm25_count, ranked)
