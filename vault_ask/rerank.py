"""Rerank: grade the fused candidates against the question and drop the filler.

RRF scores carry no relevance information — rank 1 scores 1/61 whether the
match is perfect or nonsense — so after fusion nothing has actually asked
"is this chunk about the question?". Measured on "what content do we have on
deepfakes?": with 16 slots the tail filled with a photo-gallery agent guide
and a board-governance article. One cheap model call grades every candidate
0-3 in a single request.

Best-effort, like the embedding host: any failure (timeout, 429, unparseable
reply) returns None and the caller keeps the fused order. Reranking can only
make an answer better; it must never be the reason there isn't one.
"""

from __future__ import annotations

import asyncio
import logging
import re

import litellm

from .config import Settings
from .retrieval import SearchHit

log = logging.getLogger("vault_ask.rerank")

#: Enough to tell what a chunk is about; the full body only costs tokens.
_SNIPPET_CHARS = 500
_TIMEOUT_S = 30

_SYSTEM = (
    "You grade search results. For each numbered candidate passage, rate how "
    "directly it is about the user's question: 1 = tangential, 2 = relevant, "
    "3 = directly on topic. Reply with one line per candidate that scores 1, 2 "
    "or 3, in the form `N: grade`, and nothing else. Leave out every candidate "
    "that is unrelated or only a passing mention. If none are relevant, reply "
    "`none`."
)

#: Seconds to wait before the single retry after an upstream 429. Measured:
#: OpenRouter's shared pool for flash-lite returned a 429 mid-run; a transient
#: limit should not cost the answer its reranking.
_RETRY_DELAY_S = 2.0

_LABELLED_RE = re.compile(r"(\d+)\s*[:=]\s*(-?\d+(?:\.\d+)?)")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _prompt(question: str, hits: list[SearchHit]) -> str:
    lines = [f"Question: {question}", "", "Candidates:"]
    for i, hit in enumerate(hits, start=1):
        heading = f" > {hit.heading_path}" if hit.heading_path else ""
        snippet = " ".join(hit.body.split())[:_SNIPPET_CHARS]
        lines.append(f"[{i}] {hit.title}{heading}\n{snippet}")
    return "\n".join(lines)


def _whole_grade(value: float) -> bool:
    return value == int(value) and 0 <= value <= 3


def parse_grades(text: str, n: int) -> list[int] | None:
    """The model's reply as ``n`` grades in 0..3, or None if it is not that.

    Sparse labelled lines (``7: 3``): a candidate the reply does not mention
    is grade 0. That is what the prompt asks for, and it is also what models do
    unprompted — measured live, flash-lite answered ``8: 3 / 10: 3 / 28: 3``
    to a prompt that demanded all 30 and left the zeros out. Labels, rather
    than a positional list, because a positional list misaligns when a model
    loses count (a 9B local model returned 20-26 values for 30 candidates).

    Rejected: a label outside 1..n, one candidate graded two ways, or a grade
    that is not a whole number in 0..3. ``none`` means every grade is 0.

    A bare run of exactly ``n`` numbers is still accepted as a fallback —
    flash-lite answered ``1.0, 0.0, 3.0`` to the original positional prompt —
    but only when the reply carries no labels at all.
    """
    labelled: dict[int, float] = {}
    for label, value in _LABELLED_RE.findall(text):
        index = int(label)
        if not 1 <= index <= n:
            return None
        if index in labelled and labelled[index] != float(value):
            return None  # the same candidate graded two different ways
        labelled[index] = float(value)
    if labelled:
        if not all(_whole_grade(v) for v in labelled.values()):
            return None
        return [int(labelled.get(i, 0)) for i in range(1, n + 1)]
    if re.fullmatch(r"\W*none\W*", text, re.IGNORECASE):
        return [0] * n

    numbers = [float(m) for m in _NUMBER_RE.findall(text)]
    if len(numbers) != n or not all(_whole_grade(g) for g in numbers):
        return None
    return [int(g) for g in numbers]


def apply_grades(hits: list[SearchHit], grades: list[int]) -> list[SearchHit]:
    """Best grade first, fused order within a grade; grade-0 hits dropped.

    If everything grades 0 the grades are ignored rather than obeyed: a
    reranker that finds nothing is more likely wrong than the whole candidate
    set being noise, and turning its doubt into "the vault is silent" would
    hide notes the user can find by searching.
    """
    if not any(grades):
        return hits
    ranked = sorted(
        ((g, i) for i, g in enumerate(grades) if g > 0),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return [hits[i] for _, i in ranked]


async def _complete(cfg: Settings, question: str, hits: list[SearchHit]) -> str:
    """The rerank model's reply. One retry on a rate limit, nothing else."""
    for attempt in (1, 2):
        try:
            response = await litellm.acompletion(
                model=cfg.models.rerank,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": _prompt(question, hits)},
                ],
                api_key=(
                    cfg.openrouter_api_key.get_secret_value() if cfg.openrouter_api_key else None
                ),
                temperature=0,
                timeout=_TIMEOUT_S,
            )
            return response.choices[0].message.content or ""
        except litellm.RateLimitError:
            if attempt == 2:
                raise
            log.info("rerank.rate_limited retrying in %.0fs", _RETRY_DELAY_S)
            await asyncio.sleep(_RETRY_DELAY_S)
    raise AssertionError("unreachable")


async def rerank(
    cfg: Settings, question: str, hits: list[SearchHit]
) -> list[SearchHit] | None:
    """``hits`` reordered and pruned by relevance, or None if reranking failed."""
    if len(hits) < 2:
        return None
    try:
        text = await _complete(cfg, question, hits)
    except Exception:
        log.warning("rerank.call_failed", exc_info=True)
        return None
    grades = parse_grades(text, len(hits))
    if grades is None:
        log.warning("rerank.unparseable reply=%r", text[:200])
        return None
    return apply_grades(hits, grades)
