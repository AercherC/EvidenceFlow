"""Define the vendor-neutral synchronous LLM text-generation contract and the minimal exception hierarchy."""

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol, runtime_checkable

from evidenceflow.domain.generation import LLMMessage


class LLMError(RuntimeError):
    """Common base class for every configuration and request failure of a formal LLM Client."""


class LLMConfigurationError(LLMError):
    """Signals that the Client cannot be created safely from the current non-sensitive configuration and credentials."""


class LLMRequestError(LLMError):
    """Signals that an LLM request failed or the vendor response does not satisfy the plain-text contract."""


class LLMDeadlineExceeded(LLMRequestError):
    """本次调用的上游截止时间已到，不能再发送请求或使用迟到的回复。"""


_REMAINING_TIME: ContextVar[Callable[[], float] | None] = ContextVar("llm_remaining_time", default=None)


@contextmanager
def llm_deadline(remaining_time: Callable[[], float]) -> Iterator[None]:
    """让本次编排的决策、生成及修复共享同一截止时间。"""
    token = _REMAINING_TIME.set(remaining_time)
    try:
        yield
    finally:
        _REMAINING_TIME.reset(token)


def remaining_llm_time() -> float | None:
    """取发起本次传输前的剩余秒数，普通调用没有上游时限。"""
    remaining = _REMAINING_TIME.get()
    if remaining is None:
        return None
    seconds = remaining()
    if seconds <= 0:
        raise LLMDeadlineExceeded("LLM deadline exceeded")
    return seconds


@runtime_checkable
class LLMClient(Protocol):
    """Accept the project's message model and synchronously return the raw model text."""

    @property
    def last_usage(self) -> dict | None:
        """``prompt_tokens`` / ``completion_tokens`` of the most recent call; None when the service returned no usage."""

        ...

    def generate_json(
        self,
        messages: Sequence[LLMMessage],
        *,
        schema_name: str,
        json_schema: dict,
    ) -> str:
        """Request JSON in the already-selected structured-output mode, return the raw text, and do not parse it at this layer."""

        ...

    def generate(self, messages: Sequence[LLMMessage]) -> str:
        """Produce one non-empty piece of text; carries no RAG or Citation business logic."""

        ...
