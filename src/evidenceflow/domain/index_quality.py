"""索引质量报告的共享数据合同和发布复核，供评测输出与生产发布使用。"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from evidenceflow.domain.evaluation_v2 import ContractModel


REPORT_SCHEMA_VERSION = "rag-e2e-index-quality-v1"
QualityStatus = Literal["passed", "failed", "unverifiable"]


class CaseCoverage(ContractModel):
    """单题在一个索引上的证据覆盖（组级，复用 diagnosis 的 stage）。"""

    case_id: str
    slice: str
    evidence_groups: int = Field(ge=0)
    usable_groups: int = Field(ge=0)
    candidate_covered_groups: int = Field(ge=0, description="完整进入候选池的可用证据组数。")
    context_covered_groups: int = Field(ge=0, description="完整进入最终 Context 的可用证据组数。")
    unmappable_groups: int = Field(ge=0, description="gold 在该索引上无法定位的证据组数。")
    all_context_covered: bool = Field(description="该题所有可用证据组是否都进入最终 Context。")


class IndexSide(ContractModel):
    """一侧索引的身份与覆盖汇总；``index_signature`` 必须是 lifecycle 指针给出的真实值。"""

    role: Literal["active", "staging"]
    collection_name: str
    index_signature: str = Field(min_length=1)
    chunk_count: int = Field(ge=0)
    governance: Literal["lifecycle_active", "lifecycle_staging"]
    cases: int = Field(ge=0)
    evidence_groups: int = Field(ge=0)
    usable_groups: int = Field(ge=0)
    candidate_covered_groups: int = Field(ge=0)
    context_covered_groups: int = Field(ge=0)
    cases_all_context_covered: int = Field(ge=0)
    answerable_cases: int = Field(ge=0)
    per_case: tuple[CaseCoverage, ...] = Field(default=())


class CaseChange(ContractModel):
    """一题从 active 到 staging 的证据组变化；只列真正发生变化的题。"""

    case_id: str
    slice: str
    candidate_groups_gained: tuple[str, ...] = ()
    candidate_groups_lost: tuple[str, ...] = ()
    context_groups_gained: tuple[str, ...] = ()
    context_groups_lost: tuple[str, ...] = ()

    @property
    def direction(self) -> Literal["gain", "loss", "mixed", "unchanged"]:
        """先看 Context（门禁口径），再看候选池；用于报告里的逐题变化分类。"""

        if self.context_groups_lost:
            return "loss"
        if self.context_groups_gained:
            return "gain" if not self.candidate_groups_lost else "mixed"
        if self.candidate_groups_gained:
            return "gain"
        if self.candidate_groups_lost:
            return "loss"
        return "unchanged"


class SliceChange(ContractModel):
    """受影响切片：该切片内发生变化的题数与证据组增减。"""

    name: str
    cases: int = Field(default=0, ge=0)
    changed_cases: int = Field(default=0, ge=0)
    candidate_groups_gained: int = Field(default=0, ge=0)
    candidate_groups_lost: int = Field(default=0, ge=0)
    context_groups_gained: int = Field(default=0, ge=0)
    context_groups_lost: int = Field(default=0, ge=0)


class IndexQualityReport(ContractModel):
    """一份可被激活门禁复核的 active/staging 质量报告。"""

    schema_version: Literal["rag-e2e-index-quality-v1"] = REPORT_SCHEMA_VERSION
    status: QualityStatus
    dataset_id: str = Field(min_length=1)
    dataset_fingerprint: str = Field(min_length=1)
    active: IndexSide
    staging: IndexSide
    changes: tuple[CaseChange, ...] = ()
    slices: tuple[SliceChange, ...] = ()
    dataset_cases: int = Field(default=0, ge=0, description="题集总题数（未截断）。")
    evaluated_cases: int = Field(default=0, ge=0, description="两侧都实际观测到的题数。")
    new_document_ids: tuple[str, ...] = ()
    new_documents_without_human_gold: tuple[str, ...] = ()
    new_document_context_misses: tuple[str, ...] = ()
    blockers: tuple[str, ...] = Field(default=(), description="无法据此声称质量通过的原因。")
    reasons: tuple[str, ...] = Field(default=(), description="判定理由，按固定顺序，便于复核。")

    def to_payload(self) -> dict:
        """写出用的 JSON 兼容字典（与 schema_version 一起构成文件合同）。"""

        return self.model_dump(mode="json")


class QualityGateResult(ContractModel):
    """激活门禁对一份报告文件的复核结果；``errors`` 非空即拒绝激活。"""

    passed: bool
    errors: tuple[str, ...] = ()


def verify_quality_report(
    payload: dict,
    *,
    active_pointer: tuple[str, str, int],
    staging_pointer: tuple[str, str, int],
) -> QualityGateResult:
    """复核报告文件能否支撑这次激活。

    ``payload`` 是磁盘上的 JSON；``*_pointer`` 是 ``(collection_name, index_signature, chunk_count)``，
    由 lifecycle 在构建完成后重新读取。任何结构问题、身份不一致、blocker、Context 丢失，
    都会让 ``passed`` 为 False 并把原因写进 ``errors``。
    """

    errors: list[str] = []
    try:
        report = IndexQualityReport.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - 报告无法解析即视为"没有有效质量报告"
        return QualityGateResult(
            passed=False,
            errors=(f"quality report is not valid {REPORT_SCHEMA_VERSION}: {type(exc).__name__}",),
        )
    if report.status != "passed":
        errors.append(f"quality report status is {report.status!r}, expected 'passed'")
    if report.blockers:
        errors.append(f"quality report has blockers: {', '.join(report.blockers)}")
    if report.new_documents_without_human_gold:
        errors.append(
            "quality report cannot claim new-document quality without human gold: "
            f"{list(report.new_documents_without_human_gold)[:10]}"
        )
    if report.new_document_context_misses:
        errors.append(
            "quality report records new-document Context misses: "
            f"{list(report.new_document_context_misses)[:10]}"
        )
    for side, pointer, role in (
        (report.active, active_pointer, "active"),
        (report.staging, staging_pointer, "staging"),
    ):
        if side.collection_name != pointer[0]:
            errors.append(f"{role} collection {side.collection_name!r} != pointer {pointer[0]!r}")
        if side.index_signature != pointer[1]:
            errors.append(
                f"{role} index_signature {side.index_signature[:16]}... != pointer {pointer[1][:16]}..."
            )
        if side.chunk_count != pointer[2]:
            errors.append(f"{role} chunk_count {side.chunk_count} != pointer {pointer[2]}")
        if side.governance != ("lifecycle_active" if role == "active" else "lifecycle_staging"):
            errors.append(f"{role} governance {side.governance!r} does not match its pointer role")
    if any(change.context_groups_lost for change in report.changes):
        lost = [change.case_id for change in report.changes if change.context_groups_lost]
        errors.append(f"quality report records final-Context evidence loss on cases: {lost[:10]}")
    return QualityGateResult(passed=not errors, errors=tuple(errors))
