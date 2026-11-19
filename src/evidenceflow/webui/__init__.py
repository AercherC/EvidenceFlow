"""本地论文问答与入库工作台，服务装配和 Web 依赖在启动时按需加载。"""

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from evidenceflow.webui.service import (
        RuntimeServices,
        WebUIDependencyError,
        WebUIResult,
        ask_question,
        describe_error,
        render_answer,
        render_sources,
        render_trace,
    )

__all__ = [
    "RuntimeServices",
    "WebUIDependencyError",
    "WebUIResult",
    "ask_question",
    "describe_error",
    "render_answer",
    "render_sources",
    "render_trace",
]


def __getattr__(name: str) -> Any:
    """按需加载公共接口，让命令行入口先设置模型缓存。"""

    if name in __all__:
        return getattr(import_module(".service", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
