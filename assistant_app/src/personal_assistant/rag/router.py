from __future__ import annotations

import re

from personal_assistant.schemas import Route

_MEMORY = re.compile(r"记得|记住|我的偏好|我喜欢|我住|我的信息|memory|remember", re.I)
_GRAPH = re.compile(r"关系|依赖|关联|谁负责|涉及哪些|路径|上下游|graph|relationship", re.I)
_DOCUMENT = re.compile(r"文档|资料|规范|方案|手册|总结|根据.*文件|document|policy|manual", re.I)


def classify_route(message: str) -> Route:
    """Deterministic first-pass router; an LLM router can replace this later."""
    matches = [
        (Route.MEMORY, bool(_MEMORY.search(message))),
        (Route.GRAPH, bool(_GRAPH.search(message))),
        (Route.DOCUMENT, bool(_DOCUMENT.search(message))),
    ]
    selected = [route for route, matched in matches if matched]
    if len(selected) > 1:
        return Route.HYBRID
    return selected[0] if selected else Route.CHAT
