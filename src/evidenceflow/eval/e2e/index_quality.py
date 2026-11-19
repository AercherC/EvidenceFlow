"""Active/staging 索引质量报告与激活门禁（PROJECT_SPEC §13.1 索引身份、§13.2 staging 门禁）。

这份报告只做一件事：把**同一份题集与 gold** 在两个真实索引上的证据覆盖放在一起比较，
并把结论绑定到两个 ``index_signature``。它不重新定义指标：

- 证据组是否进入候选池、是否进入最终 Context，直接复用
  ``eval.e2e.diagnosis.diagnose_case`` 已经算出的 group stage
  （``CANDIDATE_MISS`` / ``RERANK_MISS`` / ``CONTEXT_DROP`` / ``IN_CONTEXT`` / ...）；
- 逐题与切片口径复用 ``eval.e2e.slices`` 与 ``diagnosis`` 的问答层；
- gold 映射复用 ``eval.e2e.mapping`` 产出的 ``E2EMapping``，并按
  ``collection_name`` / ``index_signature`` / ``chunk_count`` 逐项核对，两个索引不能混用一份 mapping。

结论只有三种，代码里不发明"普适阈值"：

- ``passed``：共享题目在 staging 的最终 Context 覆盖不劣于 active，且新增 PDF 的人工批准
  证据组题全部进入最终 Context；
- ``failed``：旧题发生 Context 覆盖丢失，或新增 PDF 的人工批准证据组题漏检；
- ``unverifiable``：报告本身不能支撑"质量通过"的结论，例如 staging 新增了没有人工 gold 的
  文档、新文档 gold 在 staging 上无法定位、两侧观测的题目集合不一致，或只观测了题集的一部分
  （``--limit`` 冒烟）。新 PDF 题集缺少人工 gold 时一律落在这里，而不是被当成通过或失败。

``verify_quality_report`` 是 ``scripts/build_staging_index.py --activate`` 用的门禁：它再次核对
报告里的两个身份与当前 lifecycle 指针完全一致，并要求 ``status == "passed"``、无 blocker、
无报告中记录的 Context 丢失；单改 ``status`` 字段不能掩盖这些记录。
"""

from __future__ import annotations

from typing import Literal

from evidenceflow.domain.index_quality import (
    REPORT_SCHEMA_VERSION,
    CaseChange,
    CaseCoverage,
    IndexQualityReport,
    IndexSide,
    QualityGateResult,
    QualityStatus,
    SliceChange,
    verify_quality_report,
)
from evidenceflow.eval.e2e.dataset import E2EDataset
from evidenceflow.eval.e2e.diagnosis import CaseDiagnosis, E2EObservation, diagnose_dataset
from evidenceflow.eval.e2e.mapping import E2EMapping
from evidenceflow.eval.e2e.slices import case_slice

# 复用 diagnosis 的 group stage：这些 stage 表示"该组已完整进入候选池"。
CANDIDATE_COVERED_STAGES = frozenset(
    {"RERANK_MISS", "CONTEXT_DROP", "IN_CONTEXT", "IN_CONTEXT_NOT_CITED", "CITED"}
)
# 这些 stage 表示"该组已完整进入最终 Context"。
CONTEXT_COVERED_STAGES = frozenset({"IN_CONTEXT", "IN_CONTEXT_NOT_CITED", "CITED"})


class QualityError(ValueError):
    """报告结构或身份不一致：调用方应把它当成"没有有效质量报告"，而不是质量不达标。"""


def _check_side_identity(
    mapping: E2EMapping,
    identity_fields: tuple[str, str, int],
    role: str,
) -> None:
    """mapping 必须描述同一侧索引，否则报告会把 active 的 gold 套到 staging 上。"""

    collection_name, index_signature, chunk_count = identity_fields
    if mapping.collection_name != collection_name:
        raise QualityError(
            f"{role} mapping is for collection {mapping.collection_name!r}, "
            f"the {role} pointer names {collection_name!r}"
        )
    if not mapping.index_signature:
        raise QualityError(f"{role} mapping has no index_signature")
    if mapping.index_signature != index_signature:
        raise QualityError(
            f"{role} mapping index_signature {mapping.index_signature[:16]}... does not match the "
            f"{role} pointer {index_signature[:16]}..."
        )
    if mapping.chunk_count is not None and mapping.chunk_count != chunk_count:
        raise QualityError(
            f"{role} mapping was built on {mapping.chunk_count} chunks, "
            f"the {role} pointer declares {chunk_count}"
        )


def _side_coverage(
    dataset: E2EDataset,
    mapping: E2EMapping,
    observations: list[E2EObservation],
    *,
    fingerprint: str,
    role: Literal["active", "staging"],
    collection_name: str,
    index_signature: str,
    chunk_count: int,
    governance: Literal["lifecycle_active", "lifecycle_staging"],
) -> tuple[IndexSide, dict[str, CaseDiagnosis]]:
    """用现有 diagnosis 计算一侧的组级覆盖；不做任何新的指标定义。"""

    _check_side_identity(mapping, (collection_name, index_signature, chunk_count), role)
    diagnoses = diagnose_dataset(dataset, mapping, observations, fingerprint)
    cases = {case.case_id: case for case in dataset.cases}
    rows: list[CaseCoverage] = []
    for diagnosis in diagnoses:
        case = cases[diagnosis.case_id]
        usable = [group for group in diagnosis.groups if group.stage != "UNMAPPABLE"]
        candidate = [group for group in usable if group.stage in CANDIDATE_COVERED_STAGES]
        context = [group for group in usable if group.stage in CONTEXT_COVERED_STAGES]
        rows.append(
            CaseCoverage(
                case_id=diagnosis.case_id,
                slice=case_slice(case, separate_unanswerable=True),
                evidence_groups=len(diagnosis.groups),
                usable_groups=len(usable),
                candidate_covered_groups=len(candidate),
                context_covered_groups=len(context),
                unmappable_groups=len(diagnosis.groups) - len(usable),
                all_context_covered=bool(usable) and len(context) == len(usable),
            )
        )
    rows.sort(key=lambda row: row.case_id)
    by_case = {diagnosis.case_id: diagnosis for diagnosis in diagnoses}
    side = IndexSide(
        role=role,
        collection_name=collection_name,
        index_signature=index_signature,
        chunk_count=chunk_count,
        governance=governance,
        cases=len(rows),
        evidence_groups=sum(row.evidence_groups for row in rows),
        usable_groups=sum(row.usable_groups for row in rows),
        candidate_covered_groups=sum(row.candidate_covered_groups for row in rows),
        context_covered_groups=sum(row.context_covered_groups for row in rows),
        cases_all_context_covered=sum(row.all_context_covered for row in rows),
        answerable_cases=sum(
            row.slice in {"multi_doc", "single_doc"} and row.usable_groups > 0 for row in rows
        ),
        per_case=tuple(rows),
    )
    return side, by_case


def _group_ids(diagnosis: CaseDiagnosis, stages: frozenset[str]) -> set[str]:
    """取某题落在给定 stage 集合里的证据组 ID。"""

    return {group.group_id for group in diagnosis.groups if group.stage in stages}


def unstested_new_documents(
    dataset: E2EDataset,
    active_document_ids: set[str],
    staging_document_ids: set[str],
) -> list[str]:
    """找出 staging 新增、却没有人工批准证据组 gold 的文档。

    机器草稿或仅有 claim 锚点不足以验收新 PDF 的检索覆盖；已有 active 文档沿用旧题。
    """

    referenced: set[str] = set()
    for case in dataset.cases:
        if case.provenance.approved_by != "user":
            continue
        for group in case.required_evidence_groups:
            referenced.update(alt.document_id for alt in group.alternatives)
    new_documents = staging_document_ids - active_document_ids
    return sorted(new_documents - referenced)


def _slice_names(dataset: E2EDataset) -> list[str]:
    """报告固定使用这些切片名，便于跨报告比对。"""

    names = {"multi_doc", "single_doc", "unanswerable", "all"}
    names.update(case_slice(case, separate_unanswerable=True) for case in dataset.cases)
    return sorted(names)


def build_index_quality_report(
    dataset: E2EDataset,
    *,
    fingerprint: str,
    active_identity: tuple[str, str, int],
    active_mapping: E2EMapping,
    active_observations: list[E2EObservation],
    staging_identity: tuple[str, str, int],
    staging_mapping: E2EMapping,
    staging_observations: list[E2EObservation],
    new_document_ids: list[str] | tuple[str, ...] = (),
    new_documents_without_human_gold: list[str] | tuple[str, ...] = (),
) -> IndexQualityReport:
    """比较两个真实索引的组级证据覆盖，并给出 passed / failed / unverifiable。

    ``active_identity`` / ``staging_identity`` 是 ``(collection_name, index_signature, chunk_count)``，
    必须来自 lifecycle 指针（调用方读指针得到）。两个 mapping 分别绑定各自的签名；只接受
    同一批 case 的观测。判定规则见模块说明。
    """

    active_side, active_by_case = _side_coverage(
        dataset,
        active_mapping,
        active_observations,
        fingerprint=fingerprint,
        role="active",
        collection_name=active_identity[0],
        index_signature=active_identity[1],
        chunk_count=active_identity[2],
        governance="lifecycle_active",
    )
    staging_side, staging_by_case = _side_coverage(
        dataset,
        staging_mapping,
        staging_observations,
        fingerprint=fingerprint,
        role="staging",
        collection_name=staging_identity[0],
        index_signature=staging_identity[1],
        chunk_count=staging_identity[2],
        governance="lifecycle_staging",
    )
    cases = {case.case_id: case for case in dataset.cases}
    blockers: list[str] = []
    reasons: list[str] = []
    if set(active_by_case) != set(staging_by_case):
        missing = sorted(set(active_by_case) ^ set(staging_by_case))
        blockers.append(f"observations cover different cases on active and staging: {missing[:10]}")
    # 只观测了部分题（例如 --limit 冒烟）时，报告不能代表整份题集：明确标为无法判定。
    unobserved = sorted(set(cases) - (set(active_by_case) | set(staging_by_case)))
    if unobserved:
        blockers.append(
            f"only {len(cases) - len(unobserved)}/{len(cases)} dataset cases were observed; "
            f"a partial run cannot claim dataset-wide quality (missing e.g. {unobserved[:5]})"
        )
    changes: list[CaseChange] = []
    lost_context: list[str] = []
    for case_id in sorted(set(active_by_case) & set(staging_by_case)):
        before, after = active_by_case[case_id], staging_by_case[case_id]
        candidate_before = _group_ids(before, CANDIDATE_COVERED_STAGES)
        candidate_after = _group_ids(after, CANDIDATE_COVERED_STAGES)
        context_before = _group_ids(before, CONTEXT_COVERED_STAGES)
        context_after = _group_ids(after, CONTEXT_COVERED_STAGES)
        change = CaseChange(
            case_id=case_id,
            slice=case_slice(cases[case_id], separate_unanswerable=True),
            candidate_groups_gained=tuple(sorted(candidate_after - candidate_before)),
            candidate_groups_lost=tuple(sorted(candidate_before - candidate_after)),
            context_groups_gained=tuple(sorted(context_after - context_before)),
            context_groups_lost=tuple(sorted(context_before - context_after)),
        )
        if change.context_groups_lost:
            lost_context.append(case_id)
        if change.candidate_groups_gained or change.candidate_groups_lost or change.context_groups_gained or change.context_groups_lost:
            changes.append(change)
    by_slice = {name: SliceChange(name=name, cases=0, changed_cases=0) for name in _slice_names(dataset)}
    for case in dataset.cases:
        name = case_slice(case, separate_unanswerable=True)
        for key in (name, "all"):
            row = by_slice[key]
            by_slice[key] = row.model_copy(update={"cases": row.cases + 1})
    for change in changes:
        for key in (change.slice, "all"):
            row = by_slice[key]
            by_slice[key] = row.model_copy(
                update={
                    "changed_cases": row.changed_cases + 1,
                    "candidate_groups_gained": row.candidate_groups_gained + len(change.candidate_groups_gained),
                    "candidate_groups_lost": row.candidate_groups_lost + len(change.candidate_groups_lost),
                    "context_groups_gained": row.context_groups_gained + len(change.context_groups_gained),
                    "context_groups_lost": row.context_groups_lost + len(change.context_groups_lost),
                }
            )
    # gold 无法在 staging 上定位：既不能算"通过"，也不能算"覆盖下降"，只能标为无法判定。
    unmappable_regressions: list[str] = []
    for case_id in sorted(set(active_by_case) & set(staging_by_case)):
        before_groups = {group.group_id: group for group in active_by_case[case_id].groups}
        for group in staging_by_case[case_id].groups:
            prior = before_groups.get(group.group_id)
            if prior is None:
                continue
            if prior.stage != "UNMAPPABLE" and group.stage == "UNMAPPABLE":
                unmappable_regressions.append(f"{case_id}:{group.group_id}")
    if unmappable_regressions:
        blockers.append(
            "gold evidence could not be located on staging for: "
            f"{unmappable_regressions[:10]}"
        )
    # 新文档必须有人工批准的证据组题；有 gold 后还须在 staging 的最终 Context 命中。
    new_ids = set(new_document_ids)
    missing_human_gold = sorted(
        set(new_documents_without_human_gold)
        | set(unstested_new_documents(dataset, set(), new_ids))
    )
    new_doc_unmappable: list[str] = []
    new_doc_context_misses: list[str] = []
    for case in dataset.cases:
        if case.provenance.approved_by != "user" or case.case_id not in staging_by_case:
            continue
        stages = {group.group_id: group.stage for group in staging_by_case[case.case_id].groups}
        for group in case.required_evidence_groups:
            if not any(alt.document_id in new_ids for alt in group.alternatives):
                continue
            key = f"{case.case_id}:{group.group_id}"
            stage = stages.get(group.group_id)
            if stage is None or stage == "UNMAPPABLE":
                new_doc_unmappable.append(key)
            elif stage not in CONTEXT_COVERED_STAGES:
                new_doc_context_misses.append(key)
    if new_doc_unmappable:
        blockers.append(
            "new-document human gold could not be mapped on staging: "
            f"{new_doc_unmappable[:10]}"
        )
    if missing_human_gold:
        blockers.append(
            "staging contains documents with no human-approved gold: "
            f"{missing_human_gold[:10]}"
        )
    if blockers:
        status: QualityStatus = "unverifiable"
        reasons.extend(blockers)
    elif lost_context or new_doc_context_misses:
        status = "failed"
        if lost_context:
            reasons.append(
                f"{len(lost_context)} case(s) lost final-Context evidence coverage: {lost_context[:10]}"
            )
        if new_doc_context_misses:
            reasons.append(
                "human-approved new-document evidence missed final Context: "
                f"{new_doc_context_misses[:10]}"
            )
    else:
        status = "passed"
        reasons.append(
            "no case lost final-Context evidence coverage relative to active "
            f"(context groups {staging_side.context_covered_groups}/{staging_side.usable_groups} "
            f"vs active {active_side.context_covered_groups}/{active_side.usable_groups})"
        )
    return IndexQualityReport(
        status=status,
        dataset_id=dataset.dataset_id,
        dataset_fingerprint=fingerprint,
        active=active_side,
        staging=staging_side,
        changes=tuple(changes),
        slices=tuple(by_slice[name] for name in sorted(by_slice)),
        dataset_cases=len(cases),
        evaluated_cases=len(set(active_by_case) & set(staging_by_case)),
        new_document_ids=tuple(sorted(new_ids)),
        new_documents_without_human_gold=tuple(missing_human_gold),
        new_document_context_misses=tuple(new_doc_context_misses),
        blockers=tuple(dict.fromkeys(blockers)),
        reasons=tuple(reasons),
    )


def render_markdown(report: IndexQualityReport) -> str:
    """把报告渲染成便于人工复核的表格；结论与 blocker 原样写出。"""

    def side_row(side: IndexSide) -> str:
        return (
            f"| {side.role} | {side.collection_name} | `{side.index_signature[:16]}...` | "
            f"{side.chunk_count} | {side.candidate_covered_groups}/{side.usable_groups} | "
            f"{side.context_covered_groups}/{side.usable_groups} | {side.cases_all_context_covered}/{side.cases} |"
        )

    lines = [
        "# active/staging index quality report",
        "",
        f"- status: **{report.status}**",
        f"- dataset: `{report.dataset_id}` (`{report.dataset_fingerprint[:16]}...`)",
        f"- cases: {report.evaluated_cases}/{report.dataset_cases} observed on both indexes",
        f"- new documents: {', '.join(report.new_document_ids) or '(none)'}",
        "",
        "| side | collection | index_signature | chunks | candidate evidence | final Context | cases fully covered |",
        "| --- | --- | --- | --- | --- | --- | --- |",
        side_row(report.active),
        side_row(report.staging),
        "",
        "## Verdict",
        "",
        *[f"- {reason}" for reason in report.reasons],
    ]
    if report.blockers:
        lines += ["", "## Blockers", "", *[f"- {blocker}" for blocker in report.blockers]]
    lines += ["", "## Affected slices", "",
              "| slice | cases | changed | candidate + | candidate - | context + | context - |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in report.slices:
        lines.append(
            f"| {row.name} | {row.cases} | {row.changed_cases} | {row.candidate_groups_gained} "
            f"| {row.candidate_groups_lost} | {row.context_groups_gained} | {row.context_groups_lost} |"
        )
    lines += ["", "## Per-question changes", "",
              "| case | slice | candidate + | candidate - | context + | context - |",
              "| --- | --- | --- | --- | --- | --- |"]
    for change in report.changes:
        lines.append(
            f"| {change.case_id} | {change.slice} | {', '.join(change.candidate_groups_gained) or '-'} "
            f"| {', '.join(change.candidate_groups_lost) or '-'} "
            f"| {', '.join(change.context_groups_gained) or '-'} "
            f"| {', '.join(change.context_groups_lost) or '-'} |"
        )
    if not report.changes:
        lines.append("| (none) | | | | | |")
    return "\n".join(lines) + "\n"
