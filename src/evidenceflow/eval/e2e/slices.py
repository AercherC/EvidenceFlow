"""Case slices and small statistics shared by the step-0 / step-1 / generation comparisons.

The multi-document slice is ``question_type == "multi_document"``. The ``attribute`` report
(``diagnosis.build_report``) keeps its own tag-based slices and does not use this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.dataset import EvalCase

SLICES = ("multi_doc", "single_doc", "all")
MULTI_DOCUMENT_TYPE = "multi_document"
SLICE_DEFINITION = f"question_type == {MULTI_DOCUMENT_TYPE!r}"


def is_multi_doc(case: EvalCase) -> bool:
    """Whether the case belongs to the multi-document slice."""

    return case.question_type == MULTI_DOCUMENT_TYPE


def case_slice(case: EvalCase, *, separate_unanswerable: bool = False) -> str:
    """``multi_doc`` or ``single_doc``; with ``separate_unanswerable`` a single-document case that
    expects insufficient_context is ``unanswerable`` instead."""

    if is_multi_doc(case):
        return "multi_doc"
    if separate_unanswerable and case.expected_status != AnswerStatus.ANSWERED:
        return "unanswerable"
    return "single_doc"


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100, no interpolation); None for an empty list."""

    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]
