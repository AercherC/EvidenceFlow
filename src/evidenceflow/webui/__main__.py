"""``python -m evidenceflow.webui`` 的进程入口。"""

import os
from pathlib import Path


def main() -> int:
    """从项目目录启动，默认复用本地模型缓存并禁用 Hub 下载。"""

    # Hugging Face 在首次导入时读取缓存和离线配置，因此先设置再加载 WebUI。
    os.environ.setdefault("HF_HOME", str(Path.cwd() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from evidenceflow.webui.app import main as run_webui

    return run_webui()


if __name__ == "__main__":
    raise SystemExit(main())
