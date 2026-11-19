"""A3: fully rebuild the index into a staging collection, then optionally activate or discard it after
validation; the ACTIVE index itself is never modified directly.

It reuses ``StagingBuildCoordinator`` (stage progression + validation), ``IngestionService`` (as the
builder callback) and ``validate_collection`` instead of duplicating ingestion logic. By default it
only builds and validates, never activates:

    python scripts/build_staging_index.py --expected-chunk-ids-from-active --discard-after-validate

If a previous discard failed halfway (for example the collection was already deleted while the
pointer was not cleared yet), the registered staging can be cleaned up on its own:

    python scripts/build_staging_index.py --discard-staging

Exit codes: 0 = validation passed and the chosen action completed (or --discard-staging completed);
1 = the build or the validation failed (staging and the pointer are kept as-is for diagnosis);
2 = refused to start: nothing was built, deleted or written to the state file (a staging pointer
already exists, the generated staging name collides with active/previous or with an existing
collection, the options conflict, or --discard-staging was requested with no registered staging or
with a target that is active/previous); 3 = validation passed but the wrap-up action after the build
was refused by the guard (staging collection and pointer are kept; this can only happen when the
state file was changed externally during the build); 4 = structural validation passed but
--activate was refused by the quality gate (no quality report, an unreadable report, an identity
mismatch, blockers, or a final-Context coverage loss). Staging and the pointer are kept, the active
pointer is untouched, and the reasons are written to the summary's ``quality_gate``.

--discard-staging exit codes (A3.2): 0 = the collection was deleted and the pointer cleared;
2 = refused, nothing was changed (the state file is missing, unreadable or fails validation, with
stderr ``REFUSED: cannot read lifecycle state <path>: <exception type>: <message>``; no registered
staging; the target is active/previous); 1 = an error occurred while connecting to Chroma, listing
or deleting the collection (stderr ``FAILED: <exception type>: <message>``; when the deletion fails
the staging pointer is left unchanged so the operation can be retried after a fix). Both error kinds
print a single line and never raise a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from evidenceflow.config import Settings  # noqa: E402
from evidenceflow.knowledge import IndexLifecycle, LifecycleState  # noqa: E402
from evidenceflow.knowledge.staging import (  # noqa: E402
    EMBEDDING_DIMENSION,
    Observed,
    ProtectedCollectionError,
    StageReporter,
    StagingBuildRefused,
    StagingPorts,
    activate_validated_staging,
    build_staging_collection_name,
    discard_staging_collection,
    index_update,
    make_real_ports,
    mark_collection_lifecycle,
    run_staging_build,
    verify_quality_report_for_activation,
)

DEFAULT_JOB_DIR = ROOT / "artifacts/stage_a/A3_staging_jobs"


def _one_line(exc: BaseException) -> str:
    """Flatten an exception message into one line: Pydantic-style messages span several lines while the CLI contract is one line per error."""

    return " ".join(str(exc).split())


def run_discard_only(settings: Settings) -> int:
    """--discard-staging: only connect to Chroma to delete the registered staging and clear the pointer, loading no model at all.

    Every error collapses into a single stderr line plus an exit code, never a traceback:
    - the state file is missing, unreadable or fails validation -> ``REFUSED: cannot read lifecycle state ...``, exit code 2 (nothing changed);
    - no registered staging, or the target is protected -> ``REFUSED: ...``, exit code 2 (nothing changed);
    - an error while connecting to Chroma, listing or deleting the collection (or clearing the pointer afterwards) -> ``FAILED: ...``, exit code 1.
      The pointer stays unchanged when the deletion fails: ``discard_staging_collection`` deletes the collection first and clears the pointer second.
    """

    state_path = Path(settings.index_state_path)
    # Step one only reads the state file: Chroma is not connected yet, so any failure here means
    # nothing
    # changed and counts as "refused to start".
    try:
        state = LifecycleState.model_validate_json(state_path.read_text(encoding="utf-8"))
        lifecycle = IndexLifecycle(settings.index_state_path, state.knowledge_base_id)
    except Exception as exc:  # noqa: BLE001 - a missing file / permissions / encoding / JSON / Pydantic validation error all count as a refusal
        print(
            f"REFUSED: cannot read lifecycle state {state_path}: {type(exc).__name__}: {_one_line(exc)}",
            file=sys.stderr,
        )
        return 2
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(settings.vector_db_path))
        discard_staging_collection(
            lifecycle,
            lambda name: client.delete_collection(name),
            lambda: {item.name for item in client.list_collections()},
        )
    except (StagingBuildRefused, ProtectedCollectionError) as exc:
        # Nothing has been deleted and the state file is untouched, so this counts as "refused to
        # start".
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - Chroma connect/list/delete failures are all reported as an execution failure
        # When the deletion fails, discard_staging_collection has not cleared the pointer yet, so
        # the state
        # still matches reality and the operation can be retried after a fix.
        print(f"FAILED: {type(exc).__name__}: {_one_line(exc)}", file=sys.stderr)
        return 1
    print(json.dumps({"outcome": "STAGING_DISCARDED"}, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--expected-chunk-ids-from-active",
        action="store_true",
        help="要求 staging 的 chunk ID 集合与当前 active 完全一致（同一 corpus 全量重建的确定性验证）",
    )
    parser.add_argument("--activate", action="store_true", help="校验通过后激活（默认不激活）")
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=None,
        help=(
            "激活所需的 active/staging 质量报告（scripts/e2e_index_quality.py 生成）。"
            "--activate 时必须提供；结构校验通过但报告缺失、身份不符或结论非 passed 时拒绝激活。"
        ),
    )
    parser.add_argument(
        "--discard-after-validate",
        action="store_true",
        help="校验通过后通过 Chroma API 删除 staging collection 并清空 staging 指针；校验失败时保留",
    )
    parser.add_argument(
        "--discard-staging",
        action="store_true",
        help="不构建，只删除已登记的 staging collection 并清空指针（幂等，用于恢复中途失败的 discard）",
    )
    parser.add_argument(
        "--activate-staging",
        action="store_true",
        help=(
            "不构建，激活已经完成评测的 staging：重跑独立结构校验并通过 --quality-report 门禁后才切换 active 指针"
        ),
    )
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB_DIR, help="BuildJob 与摘要输出目录")
    args = parser.parse_args(argv)

    settings = Settings()
    try:
        with index_update(settings.index_state_path):
            return _run(args, settings)
    except StagingBuildRefused as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace, settings: Settings) -> int:
    """在持有生命周期写锁时执行 CLI 指定的索引操作。"""

    if args.discard_staging:
        if args.activate or args.discard_after_validate or args.expected_chunk_ids_from_active:
            print("REFUSED: --discard-staging cannot be combined with build options", file=sys.stderr)
            return 2
        if args.quality_report is not None or args.activate_staging:
            print("REFUSED: --discard-staging cannot be combined with --quality-report / --activate-staging", file=sys.stderr)
            return 2
        return run_discard_only(settings)
    if args.activate_staging:
        if args.activate or args.discard_after_validate or args.expected_chunk_ids_from_active:
            print("REFUSED: --activate-staging cannot be combined with build options", file=sys.stderr)
            return 2
        try:
            code, summary = activate_validated_staging(
                settings,
                make_real_ports(settings),
                job_dir=args.job_dir,
                quality_report_path=args.quality_report,
                repo_root=ROOT,
            )
        except StagingBuildRefused as exc:
            print(f"REFUSED: {exc}", file=sys.stderr)
            return 2
        gate = summary.get("quality_gate") or {}
        print(
            json.dumps(
                {
                    "outcome": summary["outcome"],
                    "staging_collection": summary["staging_collection"],
                    "validation_passed": summary["validation"].get("passed"),
                    "quality_gate_passed": gate.get("passed"),
                    "quality_gate_errors": gate.get("errors", []),
                    "error": summary.get("error"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return code
    try:
        code, summary = run_staging_build(
            settings,
            make_real_ports(settings),
            job_dir=args.job_dir,
            expected_chunk_ids_from_active=args.expected_chunk_ids_from_active,
            activate=args.activate,
            discard_after_validate=args.discard_after_validate,
            quality_report_path=args.quality_report,
            repo_root=ROOT,
        )
    except StagingBuildRefused as exc:
        # Only pre-build refusals reach this point; post-build guard refusals are returned by
        # run_staging_build through exit code 3.
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    validation = summary["validation"] or {}
    gate = summary.get("quality_gate") or {}
    print(
        json.dumps(
            {
                "outcome": summary["outcome"],
                "staging_collection": summary["staging_collection"],
                "job_status": summary["job"]["status"],
                "validation_passed": validation.get("passed"),
                "actual_chunk_count": validation.get("actual_chunk_count"),
                "missing_chunk_ids": len(validation.get("missing_chunk_ids", [])),
                "unexpected_chunk_ids": len(validation.get("unexpected_chunk_ids", [])),
                "errors": validation.get("errors", []),
                "quality_gate_passed": gate.get("passed"),
                "quality_gate_errors": gate.get("errors", []),
                "job_error": summary["job"].get("error_message"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
