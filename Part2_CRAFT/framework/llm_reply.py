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
    being listed anywhere. The budget that answered is remembered per model, and
    that model's later requests start from it. Until a model has answered once,
    its first request goes out alone, so a concurrent batch does not repeat the
    climb once per request.

Environment:
    CRAFT_MAX_TOKENS_CEILING   largest budget a request is raised to (default 32000)
"""

from __future__ import annotations

import os
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

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


# What this run has learned about each model's budgets, shared by all its
# requests. _NEEDED is the largest budget a reply of the model has fitted in, so
# later requests start there; it is raised only by a reply that fits, so one
# problem whose reply never fits does not push every other request to the
# ceiling. _CAP is the largest budget the model's API has accepted when a larger
# one was refused (many APIs limit max_tokens below the ceiling); no request of
# that model asks for more, so the refusal is not repeated on every request.
_NEEDED: Dict[str, int] = {}
_CAP: Dict[str, int] = {}


def budgets(model: str, base: int, progress: Optional[Dict[str, int]] = None) -> Iterator[int]:
    """The budgets to try for one request, smallest first.

    It starts at the largest of base, what this model has needed before, and
    what this same request reached before an error (progress), and doubles up
    to MAX_TOKENS_CEILING or the model's learned cap. The caller makes one
    request per budget until a reply is not cut off and calls needed(model,
    budget) with the budget that answered.

    progress is the caller's per-request record: pass the same dict to every
    attempt of one request (its retries after an error), and a retry carries on
    from the budget the climb had reached instead of starting it again. It never
    affects other requests.
    """
    cap = _CAP.get(model, MAX_TOKENS_CEILING)
    budget = min(max(base, _NEEDED.get(model, 0), (progress or {}).get("budget", 0)), cap)
    while True:
        if progress is not None:
            progress["budget"] = budget
        yield budget
        nxt = larger_budget(budget)
        cap = _CAP.get(model, MAX_TOKENS_CEILING)
        if nxt is None or budget >= cap:
            return
        # Another request may have learned a larger need meanwhile.
        budget = min(max(nxt, _NEEDED.get(model, 0)), cap)


def needed(model: str, budget: int) -> None:
    """Record that a reply from this model fitted in budget."""
    if budget > _NEEDED.get(model, 0):
        _NEEDED[model] = budget


def refused(model: str, largest_accepted: int) -> None:
    """Record that the model's API refused a budget above largest_accepted.

    Called when a larger budget is refused (HTTP 400/422) after a smaller one
    was answered in the same climb, so the refusal is about the budget.
    """
    _CAP[model] = min(_CAP.get(model, MAX_TOKENS_CEILING), largest_accepted)
    if _NEEDED.get(model, 0) > _CAP[model]:
        _NEEDED[model] = _CAP[model]


# One request per model goes out first. Until a model has answered once, a batch
# sent concurrently would start every request at the base budget and climb each
# of them separately; the others wait here for the first, then start from the
# budget it needed. Keyed by event loop as well, since each asyncio.run is new.
_FIRST: Dict[Tuple[int, str], asyncio.Event] = {}


@asynccontextmanager
async def first_request_settles(model: str):
    """Hold a request until this model's first request has come back.

    The first caller for a model goes straight through; the others wait until
    it finishes, whether it answered or failed. Once a model has answered,
    nobody waits.
    """
    if model in _NEEDED:
        yield
        return
    key = (id(asyncio.get_running_loop()), model)
    done = _FIRST.get(key)
    if done is None:
        done = _FIRST[key] = asyncio.Event()
        try:
            yield
        finally:
            done.set()
    else:
        await done.wait()
        yield
