"""从题面提取检索目标，供检索、决策和生成复用同一顺序。"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE = re.compile(r"(?<=[?？；;。])|(?<=\.)(?=\s+[A-Z])")
_ZH_MARKER = re.compile(r"(?<=[，,])\s*(?=另外|还有|另一篇|另一个|又)")
_EN_MARKER = re.compile(
    r"(?<=,)\s*(?=(?i:also,|another\b|a different\b|and what about\b"
    r"|and (?:how|what|which|where|when|why|at how|does)\b))"
)
MIN_CLAUSE_CHARS = 6
MAX_CLAUSES = 3
_NAME_LOOKUP = re.compile(r"what (?:is|was) the name of (.+?)[?？]?$", re.IGNORECASE)
_EACH_COMPOSITION = re.compile(r"what (?:does|do) each\b.+\bconsist of[?？]?$", re.IGNORECASE)
_COORDINATE = re.compile(r",\s+and\s+", re.IGNORECASE)
_TRAINING_TASK = re.compile(
    r"(?P<paper>[\w+.-]+(?:\s+[\w+.-]+){0,3})\s+(?:adds|introduces)\s+"
    r"an?\s+(?:extra|additional)\s+training\s+task\s+for\s+(?:its|the)\s+"
    r"(?P<component>[\w-]+(?:\s+[\w-]+){0,3})",
    re.IGNORECASE,
)
_QUANTITY_REQUEST = re.compile(
    r"多少|几(?:个|层|次|项|维)|具体数值|\bhow (?:many|much)\b|"
    r"\bwhat (?:(?:are|is) (?:the )?)?(?:numeric(?:al)? )?values?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionPlan:
    """按原题顺序保留自足的子问题；命名题额外提供可做词项检索的原文描述。"""

    parts: tuple[str, ...]
    name_description: str | None = None
    coordinated: bool = False


def split_clauses(query: str) -> list[str] | None:
    """保留冻结的 R2 拆句口径，供历史基线和通用查询计划复用。"""
    pieces = _SENTENCE.split(query)
    for marker in (_ZH_MARKER, _EN_MARKER):
        pieces = [part for piece in pieces for part in marker.split(piece)]
    pieces = [piece.strip() for piece in pieces]
    pieces = [piece for piece in pieces if len(piece) >= MIN_CLAUSE_CHARS]
    if len(pieces) < 2:
        return None
    while True:
        index = next(
            (i for i in range(len(pieces) - 1) if pieces[i].endswith((";", "；"))), None
        )
        if index is None:
            break
        pieces[index : index + 2] = [pieces[index] + " " + pieces[index + 1]]
    if len(pieces) > MAX_CLAUSES:
        pieces = pieces[: MAX_CLAUSES - 1] + [" ".join(pieces[MAX_CLAUSES - 1 :])]
    return pieces if len(pieces) >= 2 else None


def plan_question(query: str) -> QuestionPlan:
    """用明确的句法识别并列目标，避免把只含「各自」指代的尾问当成检索目标。"""
    clauses = split_clauses(query)
    if clauses is None:
        match = _NAME_LOOKUP.fullmatch(query.strip())
        return QuestionPlan((query,), name_description=match[1] if match else None)
    if len(clauses) == 2 and _EACH_COMPOSITION.fullmatch(clauses[1]):
        parts = tuple(_COORDINATE.split(clauses[0].rstrip("."), maxsplit=1))
        if len(parts) == 2 and all(len(part) >= MIN_CLAUSE_CHARS for part in parts):
            return QuestionPlan(parts, coordinated=True)
    return QuestionPlan(tuple(clauses))


def scoped_training_question(part: str) -> tuple[str, str] | None:
    """把明确点名的额外训练任务改为范围内待答问题，返回题面论文名和重排查询。"""
    match = _TRAINING_TASK.fullmatch(part)
    if match is None:
        return None
    return match["paper"], f"What is the additional task used when training the {match['component']}?"


def asks_for_quantity(part: str) -> bool:
    """识别明确要求具体数值或数目的题项，供原文覆盖检查使用。"""
    return bool(_QUANTITY_REQUEST.search(part))
