"""决策模型的需求与证据合同：校验原文身份，将支持关系交给统一选择器。

原文匹配只能确认模型确实引用了当前候选，不能证明其语义判断正确；语义收益
由同题评测验证。此模块不读取评测映射，也不根据检索分数推断支持关系。
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from evidenceflow.agent.selection import EvidenceRequirement
from evidenceflow.domain.context import ContextEvidence
from evidenceflow.domain.grounding import quote_in_source


class EvidenceQuote(BaseModel):
    """一块候选中支持需求的连续原文，允许 PDF 空白、连字和行尾断词差异。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


class SearchClue(BaseModel):
    """可定位的检索线索，来源为原问题中的描述性短语或某一候选的原文。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    quote: str = Field(min_length=1)
    chunk_id: str | None = Field(description="候选块身份；原问题中的线索用 null。")


class RequirementAssessment(BaseModel):
    """一个回答要点及其已知事实，可附上待补的具体问题。"""

    model_config = ConfigDict(frozen=True, extra="forbid")
    key: str = Field(min_length=1)
    text: str = Field(min_length=1)
    answer: str | None = Field(description="证据直接支持的全部或部分事实；无支持时为 null。")
    evidence_sets: tuple[tuple[EvidenceQuote, ...], ...]
    cross_document: bool | None = Field(
        default=None, description="只有原问题确实要求跨文档比较或综合时，该要点才设为 true。"
    )
    missing_detail: str | None = Field(
        default=None, description="待补的具体信息；有支持组合时，该字段表示仅部分支持。"
    )
    search_query: str | None = Field(
        default=None, description="只查当前缺口的独立查询，可包含原文已有的论文、章节或术语。",
    )
    search_clue: SearchClue | None = None

    @field_validator("missing_detail", "search_query")
    @classmethod
    def normalize_gap(cls, value: str | None) -> str | None:
        """将模型输出的空白缺口或补搜查询统一为未提供。"""
        return (value.strip() or None) if value is not None else None


class AssessmentError(ValueError):
    """只携带稳定错误码，修复请求和 Trace 不回显原文或模型输出。"""


def grounded_search_phrase(clue: SearchClue, question: str, sources: dict[str, str]) -> str | None:
    """核对线索原文，执行该短语本身，避免把模型推测的答案混进补搜。"""
    source = question if clue.chunk_id is None else sources.get(clue.chunk_id)
    phrase = " ".join(clue.quote.split())
    return phrase if source is not None and quote_in_source(phrase, source) else None


def checked_requirements(
    assessments: Sequence[RequirementAssessment],
    evidence: Sequence[ContextEvidence],
    previous: Sequence[RequirementAssessment] = (),
    *,
    single_document_default: bool = False,
) -> tuple[EvidenceRequirement, ...]:
    """验证原文及需求连续性；实时评估可按单文档处理省略范围的响应。"""
    keys = [item.key for item in assessments]
    if not keys or len(keys) != len(set(keys)):
        raise AssessmentError("invalid_requirement_keys")
    # 不能通过删掉未回答的要点，使下一轮表面覆盖率变成 full；允许发现新的要点。
    current = {item.key: item.text for item in assessments}
    if any(current.get(item.key) != item.text for item in previous):
        raise AssessmentError("requirements_changed_or_dropped")
    by_id = {item.chunk.chunk_id: item.chunk for item in evidence}
    result = []
    for assessment in assessments:
        if not assessment.key.strip() or not assessment.text.strip():
            raise AssessmentError("blank_requirement")
        if bool(assessment.answer and assessment.answer.strip()) != bool(assessment.evidence_sets):
            raise AssessmentError("answer_support_mismatch")
        bundles = []
        for bundle in assessment.evidence_sets:
            ids = [item.chunk_id for item in bundle]
            if not ids:
                raise AssessmentError("invalid_support_bundle")
            for item in bundle:
                chunk = by_id.get(item.chunk_id)
                if chunk is None:
                    raise AssessmentError("unknown_support_chunk")
                if not quote_in_source(item.quote, chunk.content):
                    raise AssessmentError("quote_not_in_chunk")
            # 单篇论文的要点允许多个同源段落共同支持，避免用另一篇的组件补齐关系。
            single_document = (
                assessment.cross_document is False
                or (assessment.cross_document is None and single_document_default)
            )
            if single_document and len({by_id[chunk_id].document_id for chunk_id in ids}) > 1:
                continue
            # 同一块的不同原文片段可以共同支持事实；选择和预算中仍只占一块。
            bundles.append(tuple(dict.fromkeys(ids)))
        result.append(EvidenceRequirement(
            key=assessment.key, text=assessment.text, evidence_sets=tuple(bundles),
            missing_detail=assessment.missing_detail if bundles else None,
        ))
    return tuple(result)
