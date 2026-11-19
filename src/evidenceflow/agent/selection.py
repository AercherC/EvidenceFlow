"""按回答需求选择证据组合，优先覆盖完整要点，并保留预算内的部分支持。

支持关系由 Agent 的原文校验层或离线回放调用方提供，不读取评测金标准，
也不把块的存在当作语义支持证明。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, model_validator

from evidenceflow.context.builder import ContextBuilder
from evidenceflow.domain.context import BuiltContext, ContextEvidence, ContextRequirement


class EvidenceRequirement(BaseModel):
    """一个待回答要点；每个替代集合的块共同支持已知事实。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1, description="本次问题内稳定且唯一的需求标识。")
    text: str = Field(min_length=1, description="需要证据支持的事实或比较维度。")
    evidence_sets: tuple[tuple[str, ...], ...] = Field(
        default=(), description="按具体、直接程度排列的支持组合；空列表表示尚未找到支持。"
    )
    missing_detail: str | None = Field(
        default=None, description="支持组合只覆盖部分事实时，需要补查的具体信息。"
    )

    @model_validator(mode="after")
    def validate_sets(self) -> "EvidenceRequirement":
        """空组合不能被解释为无条件支持，重复块不能虚增证据。"""
        if not self.key.strip() or not self.text.strip():
            raise ValueError("requirement key and text must not be blank")
        if self.missing_detail is not None and (
            not self.missing_detail.strip() or not self.evidence_sets
        ):
            raise ValueError("partial support requires evidence and a non-blank gap")
        for chunk_set in self.evidence_sets:
            if not chunk_set or any(not value.strip() for value in chunk_set):
                raise ValueError("evidence sets must contain non-blank chunk IDs")
            if len(chunk_set) != len(set(chunk_set)):
                raise ValueError("an evidence set must not repeat chunk IDs")
        return self


@dataclass(frozen=True)
class EvidenceSelection:
    """实际生成输入及需求去向；covered 仅表示提供的支持组合已完整纳入。

    budget_excluded 表示本次选择未能在预算内纳入，不是全局无可行组合的证明。
    """

    context: BuiltContext
    covered: tuple[str, ...]
    missing: tuple[str, ...]
    budget_excluded: tuple[str, ...]
    partial: tuple[str, ...]


class EvidenceSelector:
    """以新增覆盖／token 代价选择完整组合，复用正式渲染器的精确计量。

    这是确定性的贪心选择，不承诺组合优化的全局最优；相同收益优先选择更紧凑的
    组合，再按输入需求顺序打破平局。覆盖后按模型偏好改选，并用余量保留完整替代
    组合以补充细节。未关联需求的干扰块不会用来填满上下文。
    """

    def __init__(self, builder: ContextBuilder) -> None:
        """注入生成阶段使用的同一个 builder，避免两套 token 计量产生丢块。"""
        self._builder = builder

    def select(
        self,
        evidence: Sequence[ContextEvidence],
        requirements: Sequence[EvidenceRequirement],
    ) -> EvidenceSelection:
        """选择证据并直接返回 BuiltContext；调用方不得再按另一规则重新裁剪。"""
        candidates = list(evidence)
        needs = list(requirements)
        by_id = {item.chunk.chunk_id: item for item in candidates}
        if len(by_id) != len(candidates):
            raise ValueError("evidence must not contain duplicate chunk IDs")
        if len({need.key for need in needs}) != len(needs):
            raise ValueError("requirements must not contain duplicate keys")
        for need in needs:
            for option in need.evidence_sets:
                if any(chunk_id not in by_id for chunk_id in option):
                    raise ValueError("requirement references an unknown chunk ID")

        full_needs = [need for need in needs if need.missing_detail is None]
        partial_needs = [need for need in needs if need.missing_detail is not None]

        def covered_by(ids: set[str]) -> set[str]:
            return {
                need.key for need in full_needs
                if any(set(option) <= ids for option in need.evidence_sets)
            }

        selected: list[str] = []
        context = self._builder.build_evidence([])
        covered: set[str] = set()
        while True:
            best = None
            best_priority = None
            for need in full_needs:
                if need.key in covered:
                    continue
                for option in need.evidence_sets:
                    trial_ids = list(dict.fromkeys([*selected, *option]))
                    trial = self._builder.build_evidence([by_id[key] for key in trial_ids])
                    # 正式 builder 可能因预算跳过条目；整组未放下就不能接受这个方案。
                    if trial.included_count != len(trial_ids):
                        continue
                    trial_covered = covered_by(set(trial_ids))
                    gain = len(trial_covered - covered)
                    cost = max(1, trial.total_tokens - context.total_tokens)
                    priority = (gain / cost, gain, -trial.total_tokens)
                    if gain and (best_priority is None or priority > best_priority):
                        best_priority = priority
                        best = trial_ids, trial, trial_covered
            if best is None:
                break
            selected, context, covered = best

        # 先确保覆盖，再在不丢失已覆盖要点的前提下采用更具体的首选证据。
        # 最短摘要虽能提到模块名称，却可能省略机制细节，不能仅因省 token 永久优先。
        for need in full_needs:
            if need.key not in covered:
                continue
            selected_ids = set(selected)
            for option in need.evidence_sets:
                if set(option) <= selected_ids:
                    break
                trial_ids = list(dict.fromkeys([*selected, *option]))
                for chunk_id in reversed(selected):
                    if chunk_id not in option and covered <= covered_by(set(trial_ids) - {chunk_id}):
                        trial_ids.remove(chunk_id)
                trial = self._builder.build_evidence([by_id[key] for key in trial_ids])
                if trial.included_count == len(trial_ids):
                    selected, context = trial_ids, trial
                    covered = covered_by(set(selected))
                    break

        # 在补充同一要点的替代材料之前，先给每个部分支持的要点尝试保留一个组合。
        for need in partial_needs:
            if any(set(option) <= set(selected) for option in need.evidence_sets):
                continue
            for option in need.evidence_sets:
                trial_ids = list(dict.fromkeys([*selected, *option]))
                trial = self._builder.build_evidence([by_id[key] for key in trial_ids])
                if trial.included_count == len(trial_ids):
                    selected, context = trial_ids, trial
                    break

        # 有余量时保留完整支持的替代组合，以补充同一要点的细节。
        for need in full_needs:
            for option in need.evidence_sets:
                if set(option) <= set(selected):
                    continue
                trial_ids = list(dict.fromkeys([*selected, *option]))
                trial = self._builder.build_evidence([by_id[key] for key in trial_ids])
                if trial.included_count == len(trial_ids):
                    selected, context = trial_ids, trial
                    covered = covered_by(set(selected))

        selected_ids = set(selected)
        partial = {
            need.key for need in partial_needs
            if any(set(option) <= selected_ids for option in need.evidence_sets)
        }
        # 选择器的未入选包含无关候选，不能沿用 builder 的“只因超预算跳过”解释。
        context = context.model_copy(update={
            "candidate_count": len(candidates),
            "skipped_count": len(candidates) - len(selected),
            "requirements": tuple(
                ContextRequirement(
                    key=need.key, text=need.text,
                    status=(
                        "covered" if need.key in covered else
                        "partial" if need.key in partial else
                        "missing" if not need.evidence_sets else "budget_excluded"
                    ),
                    missing_detail=need.missing_detail if need.key in partial else None,
                )
                for need in needs
            ),
        })
        return EvidenceSelection(
            context=context,
            covered=tuple(need.key for need in needs if need.key in covered),
            missing=tuple(need.key for need in needs if not need.evidence_sets),
            budget_excluded=tuple(
                need.key for need in needs
                if need.evidence_sets and need.key not in covered | partial
            ),
            partial=tuple(need.key for need in needs if need.key in partial),
        )
