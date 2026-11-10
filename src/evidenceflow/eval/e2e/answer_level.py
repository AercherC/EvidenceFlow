"""Deterministic answer-level grading and the paired test that compares two configurations.

Why this module exists
---------------------
``eval/quality.py`` already defines the *metrics* (claim precision/recall, groundedness, citation
support), but nothing outside its unit tests ever called it: there was no harness connecting a
generated answer to the frozen dataset's own keys. Every reported number in this project was
retrieval-level, which cannot answer "did the answer get better?".

Two things this module deliberately is NOT
------------------------------------------
1. **Not an LLM judge.** Grading uses only the frozen dataset's ``required_claims`` /
   ``forbidden_claims`` ``answer_keys`` through the existing ``key_match`` rules. That makes it free,
   reproducible, and impossible to flatter by prompt drift. An LLM judge is the natural next layer
   but must be validated against this layer before its numbers mean anything.
2. **Not a per-claim test.** McNemar runs on the *case*, because claims inside one answer share a
   single generation and are not independent -- the same clustering argument that makes the
   retrieval gate resample cases rather than evidence groups.

The key-matching rules, which are easy to get backwards
-------------------------------------------------------
``answer_keys`` is a list of groups:

    answer_keys = [["MRE", "Multi-Resolution Encoder", "多分辨率编码器"], ["PPD", ...]]

* WITHIN a group the alternatives are **OR** -- any one phrasing counts.
* ACROSS groups they are **AND** -- every group must be satisfied.

``claim_hits`` implements exactly this, so it is reused rather than re-derived. A self-test that
concatenates whole groups instead of picking one alternative per group will report false failures;
that mistake was made and caught during development.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Any, Mapping

from evidenceflow.domain.generation import AnswerStatus
from evidenceflow.eval.e2e.key_match import claim_hits


@dataclass(frozen=True)
class GradedAnswer:
    """One graded answer plus what produced it, so a report can be audited."""

    case_id: str
    status: str
    passed: bool
    claims_required: int
    claims_covered: int
    forbidden_hit: int
    answer_chars: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


def gradable_claims(case: Mapping[str, Any]) -> tuple[list[dict], list[dict]]:
    """The claims this module can decide, for one dataset case.

    A claim without ``answer_keys`` (or explicitly ``key_matchable=False``) cannot be graded
    deterministically and is excluded rather than counted as a failure. Including it would silently
    turn "not measurable" into "wrong answer".
    """

    required = [
        claim
        for claim in case.get("required_claims", [])
        if claim.get("required", True)
        and claim.get("key_matchable", True)
        and claim.get("answer_keys")
    ]
    forbidden = [claim for claim in case.get("forbidden_claims", []) if claim.get("answer_keys")]
    return required, forbidden


def grade_answer(case: Mapping[str, Any], answer_text: str, status: str) -> tuple[bool, int, int, int]:
    """Return (passed, claims_required, claims_covered, forbidden_hit).

    Passing requires every gradable required claim covered, no forbidden claim asserted, and a
    non-abstention status when the dataset says the question is answerable. Abstention is accepted
    only where ``acceptable_abstention`` allows it.
    """

    required, forbidden = gradable_claims(case)
    covered = sum(1 for claim in required if claim_hits(claim["answer_keys"], answer_text))
    forbidden_hit = sum(1 for claim in forbidden if claim_hits(claim["answer_keys"], answer_text))

    expected = case.get("expected_status", "answered")
    if expected == "answered":
        # An answerable question must be answered, and every gradable claim must be covered. A case
        # with no gradable claim cannot be judged and therefore cannot pass.
        passed = (
            bool(required)
            and covered == len(required)
            and forbidden_hit == 0
            and status == AnswerStatus.ANSWERED.value
        )
    else:
        # An unanswerable question is scored on abstention, not on claim coverage. The live dataset
        # gives these cases no required claims at all (verified: all five have 0), so requiring
        # `required` here would make a correct abstention impossible to pass.
        #
        # Abstaining is always the safe answer. Answering is only acceptable when the dataset marks
        # the case `acceptable_abstention` -- and even then no forbidden claim may be asserted.
        if forbidden_hit:
            passed = False
        elif status == AnswerStatus.INSUFFICIENT_CONTEXT.value:
            passed = True
        else:
            passed = status == AnswerStatus.ANSWERED.value and bool(
                case.get("acceptable_abstention")
            )
    return passed, len(required), covered, forbidden_hit


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for the discordant counts (b, c).

    The paired test for a binary outcome per case: only the cases where the two arms disagree carry
    information. Verified against the value this project recorded earlier: b=6, c=0 -> 0.03125.
    """

    if b < 0 or c < 0:
        raise ValueError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def stable_flips(
    baseline_runs: list[Mapping[str, GradedAnswer]],
    candidate_runs: list[Mapping[str, GradedAnswer]],
) -> tuple[list[str], list[str], list[str]]:
    """Classify cases as stable gains, stable losses, or within-pair flips.

    A case counts as a gain only when it passes in *every* candidate run and fails in *every*
    baseline run; anything less is sampling noise and goes to ``flips``. This is the rule the
    project's earlier answer experiment used, and it exists because a single generation is a coin
    flip.
    """

    if not baseline_runs or not candidate_runs:
        raise ValueError("both arms need at least one run")
    case_ids = sorted({cid for run in (*baseline_runs, *candidate_runs) for cid in run})
    gains: list[str] = []
    losses: list[str] = []
    flips: list[str] = []
    for case_id in case_ids:
        base = [bool(run.get(case_id) and run[case_id].passed) for run in baseline_runs]
        cand = [bool(run.get(case_id) and run[case_id].passed) for run in candidate_runs]
        if all(cand) and not any(base):
            gains.append(case_id)
        elif all(base) and not any(cand):
            losses.append(case_id)
        elif len(set(base)) > 1 or len(set(cand)) > 1:
            flips.append(case_id)
    return gains, losses, flips


def verdict(gains: int, losses: int, *, alpha: float = 0.05) -> tuple[float, str]:
    """McNemar p-value plus a one-line verdict string."""

    p_value = mcnemar_exact(gains, losses)
    if p_value < alpha and gains > losses:
        return p_value, "improvement"
    if p_value < alpha and losses > gains:
        return p_value, "regression"
    return p_value, "no significant answer-level difference"
