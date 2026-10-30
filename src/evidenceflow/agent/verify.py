"""Tier-1 claim verification: deterministic checks that an answer introduces no ungrounded number or term.

Two rules, applied after the citation contract has passed:
- **numbers**: every number in a sentence must appear in one of the chunks that sentence cites; a
  sentence without labels is checked against all chunks the answer cites;
- **terms**: every key-like term (an ASCII token of at least 3 characters with an inner capital
  or with both letters and digits, e.g. PoinTr, KITTI, ShapeNet-55, Completion3D) must appear in
  some context chunk, compared without hyphens (or with them read as spaces) and allowing a
  plural "s" in the text.
Numbers and terms that occur in the question are exempt: the answer may repeat what it was asked.
Chunks are matched as the generator saw them: "File: <file_name>" header plus content.

Normalisation follows evidenceflow.eval.e2e.key_match.normalize_text (NFKC, line-end hyphens
folded, thousands separators dropped, casefold, whitespace collapsed); the runtime may not import
the eval layer, so the rules are restated here and pinned to it by tests/test_agent_verify.py.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from evidenceflow.domain.context import BuiltContext

_CITATION = re.compile(r"\[S\d+\]")
_LINE_END_HYPHEN = re.compile(r"-\s*\n\s*")
_WHITESPACE = re.compile(r"\s+")
_THOUSANDS_GROUPED = re.compile(r"(?<![\d.])(?<!\d,)\d{1,3}(?:,\d{3})+(?!,?\d)")
# Sentence ends: . ! ? followed by whitespace or the end, and CJK terminators. A decimal point has
# a digit right after it, so 6.53 is never cut.
_SENTENCE_END = re.compile(r"(?<=[.!?])(?=\s)|(?<=[。！？；])")
_LEADING_LABELS = re.compile(r"^(\s*\[S\d+\])+")
# A number: digits with optional decimal part, after thousands separators are dropped.
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
# Enumeration markers that carry no claim: "(1)", "1)", "1." at the start of a line or sentence.
_ENUMERATION = re.compile(r"(?:^|\s)\(?\d\)|^\s*\d\.(?=\s)")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*[A-Za-z0-9]")


class Verification(BaseModel):
    """``unsupported`` lists each failed check as "number:<value>" or "term:<value>"."""

    model_config = ConfigDict(frozen=True)

    unsupported: tuple[str, ...]
    checked: int

    @property
    def passed(self) -> bool:
        return not self.unsupported


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _LINE_END_HYPHEN.sub("", text)
    text = _CITATION.sub("", text)
    text = _THOUSANDS_GROUPED.sub(lambda m: m.group(0).replace(",", ""), text)
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def _is_ascii_alnum(char: str) -> bool:
    return char.isascii() and char.isalnum()


def contains(text_norm: str, needle: str, *, plural: bool = False) -> bool:
    """Substring match on normalised text with the key matcher's ASCII boundary rule.

    ``plural`` also accepts one trailing "s" after the match (kNN in the text "kNNs")."""

    needle = normalize(needle)
    if not needle:
        return False
    start = text_norm.find(needle)
    while start != -1:
        end = start + len(needle)
        if plural and end < len(text_norm) and text_norm[end] == "s":
            end += 1 if end + 1 == len(text_norm) or not _is_ascii_alnum(text_norm[end + 1]) else 0
        left = start == 0 or not (_is_ascii_alnum(needle[0]) and _is_ascii_alnum(text_norm[start - 1]))
        right = end == len(text_norm) or not (_is_ascii_alnum(needle[-1]) and _is_ascii_alnum(text_norm[end]))
        if left and right:
            return True
        start = text_norm.find(needle, start + 1)
    return False


def _dehyphen(text_norm: str) -> str:
    """Terms are compared without hyphens: PDF line-end folding turns Projected-ShapeNet-55 into
    ProjectedShapeNet-55."""

    return text_norm.replace("-", "")


def _term_variants(term_norm: str) -> tuple[str, ...]:
    """The term with its hyphens dropped, and with them read as spaces (kNN-based vs "kNN based")."""

    return tuple(dict.fromkeys((_dehyphen(term_norm), term_norm.replace("-", " "))))


def split_sentences(answer: str) -> list[tuple[str, tuple[str, ...]]]:
    """Sentences with their own labels. Labels that open a sentence ("… loss. [S2] Next …") are
    moved back to the sentence they close."""

    pieces = [p for p in _SENTENCE_END.split(answer) if p.strip()]
    sentences: list[list] = []
    for piece in pieces:
        lead = _LEADING_LABELS.match(piece)
        if lead and sentences:
            sentences[-1][0] += lead.group(0)
            piece = piece[lead.end():]
            if not piece.strip():
                continue
        sentences.append([piece, None])
    return [(text, tuple(dict.fromkeys(_CITATION.findall(text)))) for text, _ in sentences]


def numbers_in(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", _CITATION.sub("", text))
    text = _ENUMERATION.sub(" ", text)
    text = _THOUSANDS_GROUPED.sub(lambda m: m.group(0).replace(",", ""), text)
    return list(dict.fromkeys(_NUMBER.findall(text)))


def key_terms_in(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", _CITATION.sub("", text))
    terms = []
    for token in _TOKEN.findall(text):
        if len(token) < 3:
            continue
        inner_capital = any(c.isupper() for c in token[1:])
        letters_and_digits = any(c.isalpha() for c in token) and any(c.isdigit() for c in token)
        if inner_capital or letters_and_digits:
            terms.append(token)
    return list(dict.fromkeys(terms))


def render_entry(file_name: str, content: str) -> str:
    """The part of a context entry the generator reads as evidence (ContextBuilder's layout)."""

    return f"File: {file_name}\nContent:\n{content}"


def verify_answer(answer: str, query: str, chunks: Sequence[tuple[str, str]]) -> Verification:
    """``chunks`` are (source_label, rendered text) pairs in context order."""

    by_label = {label: normalize(text) for label, text in chunks}
    context_norm = " ".join(by_label.values())
    query_norm = normalize(query)
    sentences = split_sentences(answer)
    cited_anywhere = tuple(dict.fromkeys(label for _, labels in sentences for label in labels if label in by_label))
    unsupported: list[str] = []
    checked = 0
    for text, labels in sentences:
        scope = [by_label[label] for label in labels if label in by_label] or [by_label[label] for label in cited_anywhere]
        for number in numbers_in(text):
            if contains(query_norm, number):
                continue
            checked += 1
            if not any(contains(chunk, number) for chunk in scope):
                unsupported.append(f"number:{number}")
    context_terms, query_terms = _dehyphen(context_norm), _dehyphen(query_norm)
    for term in key_terms_in(answer):
        variants = _term_variants(normalize(term))
        if any(contains(query_terms, v, plural=True) for v in variants):
            continue
        checked += 1
        if not any(contains(context_terms, v, plural=True) for v in variants):
            unsupported.append(f"term:{term}")
    return Verification(unsupported=tuple(dict.fromkeys(unsupported)), checked=checked)


def verify_against_context(answer: str, query: str, context: BuiltContext) -> Verification:
    return verify_answer(
        answer,
        query,
        [(e.source_label, render_entry(e.source_reference.file_name, e.content)) for e in context.entries],
    )
