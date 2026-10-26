"""End-to-end comparison of generation runs: answer/citation quality, noise floor and LLM cost.

Pure computation over artifacts already on disk: for every run it reads the observations written by
``scripts/e2e_diagnose.py observe`` (with generation) and the matching ``attribute`` report, and
compares runs case by case. No model is loaded and no request is sent.

Slices are taken from the dataset's ``question_type`` (``evidenceflow.eval.e2e.slices``), with the
single-document cases that expect insufficient_context split off as ``unanswerable``.

What it reports:
- question-level §13.1(e) layers per run and slice;
- group-level stopping stages per run and slice;
- status on cases that expect ``insufficient_context`` (unanswerable / false premise) and
  abstentions on answerable cases;
- case-level layer flips between two runs (a candidate change vs. a repeat of the baseline, which
  is the noise floor);
- groups that reach the final top-k in one run but not the other, and what the answer then did
  with them;
- LLM cost per run from the observations' ``generation_cost``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.dataset import E2EDataset, EvalCase
from evidenceflow.eval.e2e.diagnosis import CaseDiagnosis, E2EObservation
from evidenceflow.eval.e2e.slices import SLICES, case_slice, percentile

GEN_COMPARE_SCHEMA_VERSION = "rag-e2e-gen-compare-v1"
FALSE_PREMISE_TAG = "false_premise"
# 关注的问题层，报告中优先列出；其余出现的层排在后面
HIGHLIGHT_LAYERS = ("PASS", "GENERATION", "CITATION", "STATUS", "ERROR", "UNVERIFIED",
                    "CANDIDATE_MISS", "RERANK_MISS", "CONTEXT", "UNATTRIBUTABLE")
GROUP_STAGES = ("UNMAPPABLE", "CANDIDATE_MISS", "RERANK_MISS", "CONTEXT_DROP", "IN_CONTEXT",
                "IN_CONTEXT_NOT_CITED", "CITED")
# A group "reaches the final set" when one of its chunk sets is fully in the final top-k; that is
# every stage from CONTEXT_DROP onward (the same count as the offline step-0/step-1 sweeps).
REACHED_FINAL = frozenset({"CONTEXT_DROP", "IN_CONTEXT", "IN_CONTEXT_NOT_CITED", "CITED"})


@dataclass(frozen=True)
class RunData:
    """One generation run: its attribute diagnoses and raw observations, keyed by case_id."""

    name: str
    candidate_pool_k: int
    final_top_k: int
    diagnoses: Mapping[str, CaseDiagnosis]
    observations: Mapping[str, E2EObservation]


def load_run(name: str, report: dict, observations: Sequence[E2EObservation]) -> RunData:
    """Build a run from an attribute report (``build_report`` output) and its observations."""

    if not report.get("generation_ran"):
        raise ValueError(f"run {name}: attribute report is retrieval-only; generation must have run")
    diagnoses = {case["case_id"]: CaseDiagnosis.model_validate(case) for case in report["cases"]}
    by_id = {o.case_id: o for o in observations}
    if set(diagnoses) != set(by_id):
        raise ValueError(f"run {name}: attribute report and observations cover different cases")
    return RunData(
        name=name,
        candidate_pool_k=report["candidate_pool_k"],
        final_top_k=report["final_top_k"],
        diagnoses=diagnoses,
        observations=by_id,
    )


def slice_of(case: EvalCase) -> str:
    """multi_doc, single_doc (other answerable cases) or unanswerable."""

    return case_slice(case, separate_unanswerable=True)


def _in_slice(case: EvalCase, name: str) -> bool:
    return name == "all" or slice_of(case) == name


def _pct(count: int, total: int) -> float | None:
    return round(100 * count / total, 1) if total else None


def layer_summary(run: RunData, cases: Mapping[str, EvalCase]) -> dict:
    """Question-level layer counts and shares per slice (shares in percent of the slice's cases)."""

    out = {}
    for name in SLICES:
        selected = [d for case_id, d in run.diagnoses.items() if _in_slice(cases[case_id], name)]
        counts = Counter(d.layer for d in selected)
        out[name] = {
            "cases": len(selected),
            "layers": {layer: {"count": n, "pct": _pct(n, len(selected))} for layer, n in sorted(counts.items())},
        }
    return out


def stage_summary(run: RunData, cases: Mapping[str, EvalCase]) -> dict:
    """Group-level stopping-stage counts per slice, plus how many groups reached the final top-k."""

    out = {}
    for name in SLICES:
        groups = [g for case_id, d in run.diagnoses.items() if _in_slice(cases[case_id], name) for g in d.groups]
        usable = [g for g in groups if g.stage != "UNMAPPABLE"]
        counts = Counter(g.stage for g in groups)
        out[name] = {
            "groups": len(groups),
            "usable_groups": len(usable),
            "stages": dict(sorted(counts.items())),
            "reached_final": sum(g.stage in REACHED_FINAL for g in usable),
        }
    return out


def status_summary(run: RunData, cases: Mapping[str, EvalCase]) -> dict:
    """Status on cases expecting insufficient_context and on false-premise cases, plus abstentions
    on answerable cases.

    False-premise cases are answerable with abstention allowed (dataset rule D8): answering is
    correct only when the corrected fact is stated, which the GENERATION layer checks; forbidden
    claims are not checked by §13.1(e).
    """

    def row(case_id: str, case: EvalCase, kind: str | None) -> dict:
        observation = run.observations[case_id]
        return {
            "case_id": case_id,
            "kind": kind,
            "status": None if observation.status is None else observation.status.value,
            "error_type": observation.error_type,
            "layer": run.diagnoses[case_id].layer,
        }

    refusals, false_premise = [], []
    for case_id, case in sorted(cases.items()):
        if case_id not in run.diagnoses:
            continue
        if case.expected_status == AnswerStatus.INSUFFICIENT_CONTEXT:
            refusals.append(row(case_id, case, case.unanswerable_kind))
        elif FALSE_PREMISE_TAG in case.tags:
            false_premise.append(row(case_id, case, FALSE_PREMISE_TAG))
    abstained = sorted(
        case_id for case_id, case in cases.items()
        if case_id in run.observations and case.expected_status == AnswerStatus.ANSWERED
        and run.observations[case_id].status == AnswerStatus.INSUFFICIENT_CONTEXT
    )
    return {
        "expected_insufficient": refusals,
        "expected_insufficient_correct": sum(item["layer"] == "PASS" for item in refusals),
        "false_premise": false_premise,
        "answerable_abstained": abstained,
        "answerable_abstained_by_slice": dict(sorted(Counter(slice_of(cases[c]) for c in abstained).items())),
    }


def compare_generation_runs(base: RunData, other: RunData, cases: Mapping[str, EvalCase]) -> dict:
    """Case-level layer flips and group-level reach changes from ``base`` to ``other``."""

    shared = sorted(set(base.diagnoses) & set(other.diagnoses))
    flips = []
    for case_id in shared:
        before, after = base.diagnoses[case_id].layer, other.diagnoses[case_id].layer
        if before != after:
            direction = "gain" if after == "PASS" else "loss" if before == "PASS" else "other"
            flips.append({"case_id": case_id, "slice": slice_of(cases[case_id]), "from": before, "to": after,
                          "direction": direction})
    pass_before = sum(base.diagnoses[c].layer == "PASS" for c in shared)
    pass_after = sum(other.diagnoses[c].layer == "PASS" for c in shared)

    newly, lost, stage_changes = [], [], 0
    for case_id in shared:
        before_groups = {g.group_id: g for g in base.diagnoses[case_id].groups}
        for group in other.diagnoses[case_id].groups:
            prior = before_groups[group.group_id]
            stage_changes += prior.stage != group.stage
            reached_before, reached_after = prior.stage in REACHED_FINAL, group.stage in REACHED_FINAL
            if reached_before == reached_after:
                continue
            item = {
                "case_id": case_id, "group_id": group.group_id, "slice": slice_of(cases[case_id]),
                "stage_from": prior.stage, "stage_to": group.stage,
                "layer_from": base.diagnoses[case_id].layer, "layer_to": other.diagnoses[case_id].layer,
            }
            (newly if reached_after else lost).append(item)
    return {
        "base": base.name,
        "other": other.name,
        "cases": len(shared),
        "flipped_cases": len(flips),
        "pass_gained": sum(f["direction"] == "gain" for f in flips),
        "pass_lost": sum(f["direction"] == "loss" for f in flips),
        "net_pass": pass_after - pass_before,
        "flips": flips,
        "group_stage_changes": stage_changes,
        "groups_newly_reached": newly,
        "groups_no_longer_reached": lost,
        "newly_reached_cited": sum(item["stage_to"] == "CITED" for item in newly),
        "newly_reached_case_pass": len({item["case_id"] for item in newly if item["layer_to"] == "PASS"}),
        "newly_reached_cases": len({item["case_id"] for item in newly}),
    }


def cost_summary(run: RunData) -> dict:
    """LLM token cost, calls, repairs, contract errors and generation wall-clock for one run."""

    observations = list(run.observations.values())
    costs = [o.generation_cost for o in observations if o.generation_cost is not None]
    usages = [usage for cost in costs for usage in cost.usages]
    called = [cost for cost in costs if cost.llm_called]
    seconds = [cost.seconds for cost in called]
    prompt = sum((u or {}).get("prompt_tokens") or 0 for u in usages)
    completion = sum((u or {}).get("completion_tokens") or 0 for u in usages)
    repairs = sum(cost.repair_attempted for cost in costs)
    return {
        "cases": len(observations),
        "cases_without_cost_fields": len(observations) - len(costs),
        "cases_llm_called": len(called),
        "llm_calls": len(usages),
        "calls_without_usage": sum(u is None for u in usages),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "mean_prompt_tokens_per_called_case": round(prompt / len(called), 1) if called else None,
        "mean_completion_tokens_per_called_case": round(completion / len(called), 1) if called else None,
        "repair_attempted": repairs,
        "repair_succeeded": sum(cost.repair_succeeded for cost in costs),
        "repair_rate": round(repairs / len(called), 4) if called else None,
        "errors": dict(sorted(Counter(o.error_type for o in observations if o.error_type).items())),
        "generation_seconds_mean": round(sum(seconds) / len(seconds), 3) if seconds else None,
        "generation_seconds_p95": percentile(seconds, 95),
        "generation_seconds_total": round(sum(seconds), 1),
    }


def _ratio(a: float | None, b: float | None) -> float | None:
    return round(a / b, 3) if a is not None and b else None


def build_comparison(dataset: E2EDataset, runs: Sequence[RunData], *, baseline: str, repeat: str,
                     candidate: str) -> dict:
    """Assemble the full comparison: per-run summaries, baseline→candidate and baseline→repeat."""

    cases = {case.case_id: case for case in dataset.cases}
    by_name = {run.name: run for run in runs}
    for name in (baseline, repeat, candidate):
        if name not in by_name:
            raise ValueError(f"unknown run {name!r}")
    for run in runs:
        unknown = sorted(set(run.diagnoses) - set(cases))
        if unknown:
            raise ValueError(f"run {run.name} has cases outside the dataset: {unknown[:5]}")
    costs = {run.name: cost_summary(run) for run in runs}
    base_cost, cand_cost = costs[baseline], costs[candidate]
    return {
        "schema_version": GEN_COMPARE_SCHEMA_VERSION,
        "slice_rule": "multi_doc = question_type multi_document; single_doc = other answerable cases",
        "slice_cases": dict(sorted(Counter(slice_of(c) for c in dataset.cases).items())),
        "roles": {"baseline": baseline, "repeat": repeat, "candidate": candidate},
        "runs": {
            run.name: {
                "candidate_pool_k": run.candidate_pool_k,
                "final_top_k": run.final_top_k,
                "question_layers": layer_summary(run, cases),
                "group_stages": stage_summary(run, cases),
                "status": status_summary(run, cases),
                "cost": costs[run.name],
            }
            for run in runs
        },
        "comparisons": {
            "candidate_vs_baseline": compare_generation_runs(by_name[baseline], by_name[candidate], cases),
            "repeat_vs_baseline": compare_generation_runs(by_name[baseline], by_name[repeat], cases),
            "candidate_vs_repeat": compare_generation_runs(by_name[repeat], by_name[candidate], cases),
        },
        "cost_ratio_candidate_vs_baseline": {
            key: _ratio(cand_cost[key], base_cost[key])
            for key in ("prompt_tokens", "completion_tokens", "total_tokens", "llm_calls", "generation_seconds_mean",
                        "generation_seconds_p95")
        },
    }


# ---- Markdown ----------------------------------------------------------------------------


def _cell(entry: dict | None) -> str:
    if not entry:
        return "0"
    return f"{entry['count']} ({entry['pct']}%)" if entry["pct"] is not None else str(entry["count"])


def _flip_rows(comparison: dict) -> list[str]:
    rows = ["| case | slice | from | to | direction |", "| --- | --- | --- | --- | --- |"]
    for flip in comparison["flips"]:
        rows.append(f"| {flip['case_id']} | {flip['slice']} | {flip['from']} | {flip['to']} | {flip['direction']} |")
    if not comparison["flips"]:
        rows.append("| (none) | | | | |")
    return rows


def render_markdown(report: dict, *, notes: str | None = None) -> str:
    """Render the comparison as tables; ``notes`` (hand-written conclusions) is appended verbatim."""

    names = list(report["runs"])
    runs = report["runs"]
    roles = report["roles"]
    lines = [
        "# rag_e2e_v1 generation run comparison",
        "",
        f"Runs: " + ", ".join(
            f"**{name}** (pool {runs[name]['candidate_pool_k']}, final top-{runs[name]['final_top_k']})" for name in names),
        f"Baseline {roles['baseline']}, repeat {roles['repeat']} (noise floor), candidate {roles['candidate']}.",
        f"Slices by question_type: {report['slice_cases']}.",
        "",
        "## Question-level layers (§13.1(e))",
        "",
    ]
    for slice_name in SLICES:
        total = {name: runs[name]["question_layers"][slice_name]["cases"] for name in names}
        seen = set().union(*(runs[name]["question_layers"][slice_name]["layers"] for name in names))
        order = [layer for layer in HIGHLIGHT_LAYERS if layer in seen] + sorted(seen - set(HIGHLIGHT_LAYERS))
        lines += [f"### {slice_name} ({', '.join(f'{n}: {total[n]}' for n in names)} cases)", "",
                  "| layer | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
        for layer in order:
            lines.append(f"| {layer} | " + " | ".join(
                _cell(runs[name]["question_layers"][slice_name]["layers"].get(layer)) for name in names) + " |")
        lines.append("")
    lines += ["## Evidence-group stages", "",
              "| slice | stage | " + " | ".join(names) + " |", "| --- | --- |" + " --- |" * len(names)]
    for slice_name in SLICES:
        for stage in GROUP_STAGES:
            values = [runs[name]["group_stages"][slice_name]["stages"].get(stage, 0) for name in names]
            if any(values):
                lines.append(f"| {slice_name} | {stage} | " + " | ".join(map(str, values)) + " |")
        lines.append(f"| {slice_name} | **reached final top-k** (of usable "
                     f"{runs[names[0]]['group_stages'][slice_name]['usable_groups']}) | "
                     + " | ".join(str(runs[name]["group_stages"][slice_name]["reached_final"]) for name in names) + " |")
    lines += ["", "## Status on unanswerable and false-premise cases", "",
              "Unanswerable cases expect insufficient_context; false-premise cases are answerable with "
              "abstention allowed (PASS needs the corrected fact; forbidden claims are not checked).", "",
              "| case | kind | " + " | ".join(names) + " |", "| --- | --- |" + " --- |" * len(names)]
    for key in ("expected_insufficient", "false_premise"):
        # 各 run 的列表都按 case_id 排序，可按下标对齐
        for index, item in enumerate(runs[names[0]]["status"][key]):
            cells = []
            for name in names:
                entry = runs[name]["status"][key][index]
                cells.append(f"{entry['status'] or entry['error_type']} ({entry['layer']})")
            lines.append(f"| {item['case_id']} | {item['kind']} | " + " | ".join(cells) + " |")
    lines += ["", "Answerable cases where the model abstained (insufficient_context):", ""]
    for name in names:
        status = runs[name]["status"]
        lines.append(f"- {name}: {len(status['answerable_abstained'])} {status['answerable_abstained_by_slice']} "
                     f"{', '.join(status['answerable_abstained'])}")
    comparisons = report["comparisons"]
    lines += ["", "## Case-level flips", "", "| comparison | flipped cases | PASS gained | PASS lost | net PASS "
              "| group stage changes | groups newly reaching final | groups no longer reaching final |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, comparison in comparisons.items():
        lines.append(f"| {comparison['base']} → {comparison['other']} | {comparison['flipped_cases']} "
                     f"| {comparison['pass_gained']} | {comparison['pass_lost']} | {comparison['net_pass']:+d} "
                     f"| {comparison['group_stage_changes']} | {len(comparison['groups_newly_reached'])} "
                     f"| {len(comparison['groups_no_longer_reached'])} |")
    for key in ("candidate_vs_baseline", "repeat_vs_baseline"):
        comparison = comparisons[key]
        lines += ["", f"### {comparison['base']} → {comparison['other']}", "", *_flip_rows(comparison)]
    candidate = comparisons["candidate_vs_baseline"]
    lines += ["", f"## Groups newly reaching the final top-k ({candidate['base']} → {candidate['other']})", "",
              f"{len(candidate['groups_newly_reached'])} groups in {candidate['newly_reached_cases']} cases; "
              f"{candidate['newly_reached_cited']} cited by the {candidate['other']} answer; "
              f"{candidate['newly_reached_case_pass']} of those cases PASS in {candidate['other']}.", "",
              "| case | group | slice | stage A→B | layer A→B |", "| --- | --- | --- | --- | --- |"]
    for item in candidate["groups_newly_reached"]:
        lines.append(f"| {item['case_id']} | {item['group_id']} | {item['slice']} | {item['stage_from']} → "
                     f"{item['stage_to']} | {item['layer_from']} → {item['layer_to']} |")
    if candidate["groups_no_longer_reached"]:
        lines += ["", "Groups that reached the final top-k in the baseline but not in the candidate:", "",
                  "| case | group | slice | stage A→B | layer A→B |", "| --- | --- | --- | --- | --- |"]
        for item in candidate["groups_no_longer_reached"]:
            lines.append(f"| {item['case_id']} | {item['group_id']} | {item['slice']} | {item['stage_from']} → "
                         f"{item['stage_to']} | {item['layer_from']} → {item['layer_to']} |")
    lines += ["", "## LLM cost", "", "| metric | " + " | ".join(names) + " |", "| --- |" + " --- |" * len(names)]
    for metric in ("cases_llm_called", "llm_calls", "prompt_tokens", "completion_tokens", "total_tokens",
                   "mean_prompt_tokens_per_called_case", "mean_completion_tokens_per_called_case",
                   "repair_attempted", "repair_succeeded", "repair_rate", "errors", "generation_seconds_mean",
                   "generation_seconds_p95", "generation_seconds_total"):
        lines.append(f"| {metric} | " + " | ".join(str(runs[name]["cost"][metric]) for name in names) + " |")
    lines += ["", f"Cost ratio {roles['candidate']} / {roles['baseline']}: "
              + ", ".join(f"{k} {v}" for k, v in report["cost_ratio_candidate_vs_baseline"].items()), ""]
    if notes:
        lines += [notes.rstrip(), ""]
    return "\n".join(lines) + "\n"
