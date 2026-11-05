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
    "directly it is about the user's question: 0 = unrelated or only a passing "
    "mention, 1 = tangential, 2 = relevant, 3 = directly on topic. Reply with "
    "ONLY a JSON array of integers, one per candidate, in order."
)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _prompt(question: str, hits: list[SearchHit]) -> str:
    lines = [f"Question: {question}", "", "Candidates:"]
    for i, hit in enumerate(hits, start=1):
        heading = f" > {hit.heading_path}" if hit.heading_path else ""
        snippet = " ".join(hit.body.split())[:_SNIPPET_CHARS]
        lines.append(f"[{i}] {hit.title}{heading}\n{snippet}")
    return "\n".join(lines)


def parse_grades(text: str, n: int) -> list[int] | None:
    """The model's reply as ``n`` grades in 0..3, or None if it is not that.

    Lenient about format, strict about content: measured live, flash-lite
    answered ``1.0, 0.0, 3.0`` with no brackets despite being asked for a JSON
    array of integers. Any run of numbers is accepted as long as there are
    exactly ``n`` of them and each is a whole number from 0 to 3.
    """
    numbers = [float(m) for m in _NUMBER_RE.findall(text)]
    if len(numbers) != n or any(g != int(g) or not 0 <= g <= 3 for g in numbers):
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


async def rerank(
    cfg: Settings, question: str, hits: list[SearchHit]
) -> list[SearchHit] | None:
    """``hits`` reordered and pruned by relevance, or None if reranking failed."""
    if len(hits) < 2:
        return None
    try:
        response = await litellm.acompletion(
            model=cfg.models.rerank,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _prompt(question, hits)},
            ],
            api_key=cfg.openrouter_api_key.get_secret_value() if cfg.openrouter_api_key else None,
            temperature=0,
            timeout=_TIMEOUT_S,
        )
        text = response.choices[0].message.content or ""
    except Exception:
        log.warning("rerank.call_failed", exc_info=True)
        return None
    grades = parse_grades(text, len(hits))
    if grades is None:
        log.warning("rerank.unparseable reply=%r", text[:200])
        return None
    return apply_grades(hits, grades)
