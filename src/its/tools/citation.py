"""引用工具，确保输出可追溯。"""
from __future__ import annotations

from typing import List


class CitationTool:
    """将来源列表转换为统一的引用格式。"""

    def attach(self, sources: List[str]) -> List[str]:
        return [f"来源:{src}" for src in sources]
