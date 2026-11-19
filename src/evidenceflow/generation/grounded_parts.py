"""逐项回答合同，按真实原文摘录绑定引文并确定性渲染。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from evidenceflow.domain.context import BuiltContext
from evidenceflow.domain.grounding import quote_in_source
from evidenceflow.domain.question import QuestionPlan

PartStyle = Literal["lookup", "composition", "multipart"]
PARTS_SCHEMA_NAME = "grounded_parts"
PARTS_PROMPT_VERSION = "grounded-parts-v2"
_ATTRIBUTE_LOOKUP = re.compile(r"(?:叫什么(?:名字)?|用的是什么[\w-]{0,12})[？?]$")
_SOURCE_LABEL = re.compile(r"\[S\d+\]")


class _Excerpt(BaseModel):
    """生成输入中可定位的一个短摘录。"""

    model_config = ConfigDict(extra="forbid")
    source_label: str
    quote: str


class _Claim(BaseModel):
    """该项可输出的事实及支持它的原文。"""

    model_config = ConfigDict(extra="forbid")
    text: str
    evidence: list[_Excerpt] = Field(min_length=1)


class _Part(BaseModel):
    """保留同一目标的回答和可选目的，缺少证据的字段使用 null。"""

    model_config = ConfigDict(extra="forbid")
    answer: _Claim | None
    purpose: _Claim | None


class _PartsOutput(BaseModel):
    """按题面顺序逐项作答；留空的项也占据原位置。"""

    model_config = ConfigDict(extra="forbid")
    parts: list[_Part]


class GroundedPartsError(ValueError):
    """逐项生成结果与题面或原文不匹配。"""


def answer_style(plan: QuestionPlan) -> PartStyle | None:
    """明确的待答目标用逐项合同；普通陈述拆句不影响回答格式。"""
    if plan.coordinated:
        return "composition"
    if plan.name_description or (len(plan.parts) > 1 and all(_ATTRIBUTE_LOOKUP.search(p) for p in plan.parts)):
        return "lookup"
    # 只有多个独立问句时，检索拆分才同时构成逐项生成的目标。
    if len(plan.parts) > 1 and all(part.endswith(("?", "？")) for part in plan.parts):
        return "multipart"
    return None


def parts_schema() -> dict:
    """复用实际解析模型的 JSON schema，避免模型接口和本地校验漂移。"""
    return _PartsOutput.model_json_schema()


def parts_instruction(style: PartStyle, count: int) -> str:
    """指定答项数量、顺序和引用字段；原文不足时保留空项。"""
    if style == "lookup":
        rules = (
            "For each answer.text, copy only the requested name or value exactly as it appears in "
            "one supporting quote. Do not add the description, paper name, method name or other context "
            "unless that is the requested value. Set every purpose to null."
        )
    elif style == "composition":
        rules = (
            "For each answer.text, name the technique and explain what it consists of, including "
            "its inputs and how they are used when stated. Include architecture details only when "
            "they explain the mechanism. For purpose.text, include the exact name of the relevant "
            "training objective or loss when the source gives one, and explain what is optimized "
            "or distinguished. Cite the loss name and its role, using distinct excerpts if needed. "
            "Check each original target."
        )
    else:
        rules = (
            "For each answer.text, provide the details directly requested by that question and supported by "
            "the excerpts. If only some requested details are supported, also name the missing details. "
            "If none are supported, use null. Set every purpose to null."
        )
    return (
        f"Return ONLY a JSON object with 'parts' containing exactly {count} entries in request order. "
        'Each entry is {"answer": <claim or null>, "purpose": <claim or null>}. '
        'A claim is {"text": "<fact>", "evidence": [{"source_label": "[S1]", "quote": "<verbatim excerpt>"}]}. '
        "Quote only short, continuous excerpts from the cited passage body. Use only the label of a passage "
        "that states the fact. Put labels only in source_label. "
        "Use null for fields that lack evidence and keep the entry in place. "
        + rules
    )


def render_parts(
    json_text: str, context: BuiltContext, *, style: PartStyle, count: int, chinese: bool,
) -> str | None:
    """渲染带引文的答项；全为空时返回 None，交给生成器确定性拒答。"""
    try:
        output = _PartsOutput.model_validate_json(json_text)
    except ValidationError as exc:
        raise GroundedPartsError("invalid parts structure") from exc
    if len(output.parts) != count:
        raise GroundedPartsError("parts count must match the requested count")
    sources = {entry.source_label: entry.content for entry in context.entries}
    lines = []
    answered = False
    missing = "该要点证据不足。" if chinese else "Insufficient evidence for this part."
    for index, part in enumerate(output.parts, start=1):
        if style != "composition" and part.purpose is not None:
            raise GroundedPartsError(f"{style} purpose must be null")
        claims = [
            _render_claim(claim, sources, extractive=style == "lookup")
            for claim in (part.answer, part.purpose) if claim is not None
        ]
        answered |= bool(claims)
        line = " ".join(claims) if claims else missing
        lines.append(f"{index}. {line}" if count > 1 else line)
    return "\n".join(lines) if answered else None


def _render_claim(claim: _Claim, sources: dict[str, str], *, extractive: bool) -> str:
    """只使用程序验证过的来源标签，查值时还要确认输出值直接出现在摘录中。"""
    text = " ".join(claim.text.split())
    if not text or _SOURCE_LABEL.search(text):
        raise GroundedPartsError("claim text must be nonblank and contain no citation labels")
    labels = []
    for excerpt in claim.evidence:
        source = sources.get(excerpt.source_label)
        if source is None or not quote_in_source(excerpt.quote, source):
            raise GroundedPartsError("evidence quote must occur in the cited passage")
        labels.append(excerpt.source_label)
    if extractive and not any(quote_in_source(text, excerpt.quote) for excerpt in claim.evidence):
        raise GroundedPartsError("lookup value must occur in a supporting quote")
    return f"{text} {' '.join(dict.fromkeys(labels))}"
