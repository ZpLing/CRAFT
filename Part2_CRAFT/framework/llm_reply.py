"""Reading a chat-completions reply the same way in every module.

Two things differ between models, and neither is decided by the model's name:

  * Where the text is. Most models put their answer in `content`. Some
    reasoning models leave `content` empty and return the text in
    `reasoning_content`. The reply is read from `content`, and from
    `reasoning_content` only when `content` is empty.

  * How many tokens a reply needs. A reasoning model bills its hidden reasoning
    against max_tokens, so a budget sized for the visible answer can run out
    before any of it is written. The API says so: the reply's finish_reason is
    "length". A cut-off reply is requested again with twice the budget, up to
    MAX_TOKENS_CEILING, so a model that needs a larger budget gets one without
    being listed anywhere.

Environment:
    CRAFT_MAX_TOKENS_CEILING   largest budget a request is raised to (default 32000)
"""

from __future__ import annotations

import os
from typing import Any, Optional

MAX_TOKENS_CEILING: int = int(os.getenv("CRAFT_MAX_TOKENS_CEILING", "32000"))


def _get(obj: Any, key: str) -> Any:
    """A field of a reply part, whether the client returned a dict or an object."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def reply_text(message: Any) -> str:
    """The text of a reply message: its content, or its reasoning_content when content is empty."""
    text = _get(message, "content") or _get(message, "reasoning_content") or ""
    return str(text).strip()


def was_cut_off(choice: Any) -> bool:
    """Did the reply stop because it ran out of tokens?"""
    return _get(choice, "finish_reason") == "length"


def larger_budget(budget: int) -> Optional[int]:
    """The budget to ask again with after a cut-off reply, or None at the ceiling."""
    if budget >= MAX_TOKENS_CEILING:
        return None
    return min(max(budget, 1) * 2, MAX_TOKENS_CEILING)
