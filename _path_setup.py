"""
路径初始化 (所有入口脚本顶部 `import _path_setup` 即可)
============================================================
将本工程内置的修改版 `mamba_ssm` (含 VIM 双向扫描改造) 注入 sys.path,
优先于 conda 环境中可能安装的版本。

用法:
    import _path_setup   # noqa: F401, 必须在其他 import 之前

目录结构:
    MMER-Missing/
    ├── _path_setup.py    ← 本文件
    ├── mamba/            ← 内置 mamba_ssm 源码 (已植入 VIM 修改)
    │   └── mamba_ssm/
    ├── models/
    ├── train.py / run.py / tests/...
    └── ...
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_MAMBA_DIR = os.path.join(_HERE, "mamba")

# 把内置 mamba 目录放到最前面, 确保 "import mamba_ssm" 解析到本工程的修改版
if os.path.isdir(os.path.join(_MAMBA_DIR, "mamba_ssm")) and _MAMBA_DIR not in sys.path:
    sys.path.insert(0, _MAMBA_DIR)
