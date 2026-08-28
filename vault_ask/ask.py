"""Answering a question: retrieve, assemble, generate.

Hybrid retrieval — FTS5 + vector, fused by RRF, then one-hop graph expansion
on top — when embedding is configured; FTS5 alone (still fused, so scores stay
on the same scale graph expansion assumes) otherwise, or if the vector search
itself fails (e.g. the embedding host is unreachable, or `chunks_vec`'s
dimension doesn't match the query vector's because the model changed without
a --rebuild). Degrading rather than raising here matches the rest of the
pipeline's stance on the embedding host: best-effort, never load-bearing for
the app to answer at all. No rerank yet — a later step; this module's job is
to be the thing it slots into without its callers (the CLI today, the OpenAI
shim next) having to change.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import litellm

from .config import Settings
from .embed import embed_texts
from .prompts import SYSTEM_PROMPT
from .retrieval import (
    SearchHit,
    assemble_context,
    expand_graph,
    fuse_rrf,
    search_fts,
    search_vector,
)
from .web import WebResult, assemble_web_context
from .web import search as web_search

log = logging.getLogger("vault_ask.ask")

# litellm chatters on import and phones home for version checks by default —
# same reasoning as podcast-digest's llm/client.py.
litellm.telemetry = False
litellm.suppress_debug_info = True

#: A vault with nothing relevant is a correct answer (README "Answering
#: contract"), returned without ever calling the model — a guarantee, not a
#: hope that the prompt is followed.
VAULT_SILENT = "The vault is silent on this."


@dataclass(frozen=True)
class Answer:
    text: str
    hits: list[SearchHit] = field(default_factory=list)
    generated: bool = False
    #: Web results that were folded into the context, if any. Empty whenever
    #: `allow_web` is false — see `_should_search_web`.
    web: list[WebResult] = field(default_factory=list)


def _should_search_web(cfg: Settings, hits: list[SearchHit], *, allow_web: bool) -> bool:
    """Whether the vault's own answer looks thin enough to supplement.

    ``allow_web`` is checked first and is not negotiable. It is the same flag
    that restricts retrieval to `open` chunks, and that is the whole design:
    web content and `personal` content can never be in one context, because
    the single switch that admits one excludes the other. Any future caller
    that searches the web without consulting this function breaks the
    invariant README "Corpus and sensitivity" is built on.

    "Thin", not "silent": with ~2,250 notes FTS returns *something* for almost
    any question, so a silence trigger would essentially never fire and the
    feature would appear not to work.
    """
    if not allow_web or not cfg.web.enabled:
        return False
    if len(hits) < cfg.web.thin_hits:
        return True
    distances = [hit.distance for hit in hits if hit.distance is not None]
    if not distances:
        # No vector arm at all (embedding host unconfigured or unreachable, see
        # _search_vector_safe). Without it there is no relevance signal to
        # judge thinness by, so do not guess — FTS returning *something* is not
        # evidence the vault covers the question.
        return False
    return min(distances) > cfg.web.thin_distance


async def retrieve(
    conn: sqlite3.Connection,
    cfg: Settings,
    question: str,
    *,
    allow_web: bool,
    top_k: int | None = None,
) -> list[SearchHit]:
    """``allow_web`` narrows retrieval to `open` chunks — see README "Corpus and
    sensitivity": the decision is made *before* retrieval, not after, so a
    `personal` chunk is never even a candidate for a context that might later
    carry a web tool.

    Shared by every adapter (CLI, OpenAI shim, MCP's `vault_search`) so none of
    them can disagree about what counts as relevant. ``top_k`` overrides
    ``cfg.retrieval.final_top_k`` for the final truncation only — the
    fusion/graph-expansion widths stay config-driven either way; it exists for
    MCP's `vault_search(query, k)`, where the caller picks how many results it
    wants back.
    """
    sensitivity = "open" if allow_web else None
    fts_hits = search_fts(conn, question, top_k=cfg.retrieval.fts_top_k, sensitivity=sensitivity)
    vector_hits = await _search_vector_safe(conn, cfg, question, sensitivity=sensitivity)

    # Always fused, even when one source is empty: fusion is what normalises
    # scores onto the positive, higher-is-better scale expand_graph assumes —
    # raw bm25 (lower/more negative is better) would make its discount
    # multiply the wrong way.
    fused = fuse_rrf(vector_hits, fts_hits, top_k=cfg.retrieval.fusion_top_k)
    expanded = (
        expand_graph(
            conn,
            fused,
            max_siblings=cfg.retrieval.graph_max_siblings,
            discount=cfg.retrieval.graph_discount,
            sensitivity=sensitivity,
        )
        if cfg.retrieval.graph_enabled
        else []
    )
    final_k = top_k if top_k is not None else cfg.retrieval.final_top_k
    return _merge(fused, expanded, final_k=final_k, graph_slots=cfg.retrieval.graph_max_slots)


def _merge(
    fused: list[SearchHit], expanded: list[SearchHit], *, final_k: int, graph_slots: int
) -> list[SearchHit]:
    """Direct hits first, expanded chunks in a bounded quota behind them.

    Not a plain score sort of the union. Expanded chunks carry a score they
    *inherited* from the hit that reached them rather than one earned against
    the question, and the 0.7 discount does not stop that beating a real hit
    (see config.RetrievalConfig.graph_discount for the arithmetic). Measured
    over 20 questions before this: 36 expanded chunks entered final answers
    and evicted 36 direct hits — exactly 1:1, up to 6 of 8 slots — so
    expansion was substitutive, not additive, which is not what README
    "Retrieval" claimed for it.

    The quota is what makes the claim true. Expansion still contributes, and
    still cannot take the answer over. If there are not enough direct hits to
    fill `final_k`, expanded chunks are allowed past the quota rather than
    returning short — a thin answer helps nobody, and that is the case
    expansion was introduced for.
    """
    if not expanded or graph_slots <= 0:
        return sorted(fused, key=lambda hit: -hit.score)[:final_k]

    out: list[SearchHit] = []
    held_back: list[SearchHit] = []
    used = 0
    for hit in sorted((*fused, *expanded), key=lambda h: -h.score):
        if len(out) == final_k:
            break
        if hit.source == "graph":
            # A ceiling, not an allocation: expanded chunks still have to
            # out-score a direct hit to appear at all. Reserving slots for
            # them instead would guarantee them a share of every answer,
            # including the questions where expansion should do nothing —
            # measured, that turned "fired on 45% of questions" into 100%.
            if used >= graph_slots:
                held_back.append(hit)
                continue
            used += 1
        out.append(hit)

    # Only once the direct hits are exhausted: too few of them is precisely the
    # case expansion was introduced for, and a short answer helps nobody.
    if len(out) < final_k:
        out.extend(held_back[: final_k - len(out)])
        out.sort(key=lambda hit: -hit.score)
    return out


def _messages(question: str, context: str, web_context: str = "") -> list[dict[str, str]]:
    """The vault section is always present and always first.

    Web material goes in its own labelled block after it, never interleaved —
    the separation the model is asked to preserve (prompts.SYSTEM_PROMPT rule
    2) is easier to honour when the input already has it.
    """
    user = f"Question: {question}\n\nContext from the vault:\n\n{context}"
    if web_context:
        user += f"\n\n{web_context}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _api_key(cfg: Settings) -> str | None:
    return cfg.openrouter_api_key.get_secret_value() if cfg.openrouter_api_key else None


async def ask(
    conn: sqlite3.Connection,
    cfg: Settings,
    question: str,
    *,
    allow_web: bool = True,
    dry_run: bool = False,
) -> Answer:
    hits = await retrieve(conn, cfg, question, allow_web=allow_web)

    web: list[WebResult] = []
    if _should_search_web(cfg, hits, allow_web=allow_web):
        web = await web_search(cfg, question)

    # Vault-silent AND nothing from the web is still the correct answer, and
    # still costs no model call. With web results, the model runs so it can
    # say plainly that this came from the web and not from the vault.
    if not hits and not web:
        return Answer(text=VAULT_SILENT, hits=[], generated=False)
    if dry_run:
        return Answer(text=_format_hits(hits), hits=hits, generated=False, web=web)

    context = assemble_context(hits, sensitivity="open" if allow_web else None)
    response = await litellm.acompletion(
        model=cfg.models.generation,
        messages=_messages(question, context, assemble_web_context(web)),
        api_key=_api_key(cfg),
    )
    text = response.choices[0].message.content or ""
    return Answer(text=text, hits=hits, generated=True, web=web)


async def ask_stream(
    conn: sqlite3.Connection,
    cfg: Settings,
    question: str,
    *,
    allow_web: bool = True,
) -> AsyncIterator[str]:
    """Same contract as `ask`, as text deltas instead of one complete `Answer`
    — what the OpenAI shim needs for `stream: true` (README "Architecture":
    "Open WebUI is unpleasant without it"). A vault-silent answer is still one
    complete string, yielded once: there is nothing to stream token-by-token
    when no model was ever called.
    """
    hits = await retrieve(conn, cfg, question, allow_web=allow_web)

    web: list[WebResult] = []
    if _should_search_web(cfg, hits, allow_web=allow_web):
        web = await web_search(cfg, question)

    if not hits and not web:
        yield VAULT_SILENT
        return

    context = assemble_context(hits, sensitivity="open" if allow_web else None)
    response = await litellm.acompletion(
        model=cfg.models.generation,
        messages=_messages(question, context, assemble_web_context(web)),
        api_key=_api_key(cfg),
        stream=True,
    )
    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _search_vector_safe(
    conn: sqlite3.Connection, cfg: Settings, question: str, *, sensitivity: str | None
) -> list[SearchHit]:
    """Vector search, or an empty result if embedding isn't configured or the
    call/query fails for any reason — see the module docstring on why this
    degrades instead of raising.
    """
    if not cfg.models.embedding_base_url:
        return []
    try:
        [query_vector] = await embed_texts(cfg, [question])
        return search_vector(
            conn, query_vector, top_k=cfg.retrieval.vector_top_k, sensitivity=sensitivity
        )
    except Exception:
        log.warning("ask.vector_search_failed", exc_info=True)
        return []


def _format_hits(hits: list[SearchHit]) -> str:
    # %.6f, not %7.3f: RRF scores live in ~0.010-0.033, where three decimals
    # collapse adjacent ranks onto the same printed value and make an
    # expanded chunk indistinguishable from the direct hit it was discounted
    # from — which is most of why expansion was never measured.
    lines = [f"{len(hits)} hit(s):"]
    for hit in hits:
        heading = f" ({hit.heading_path})" if hit.heading_path else ""
        source = f"  [{hit.source}]" if hit.source else ""
        lines.append(f"  {hit.score:.6f}{source}  [[{hit.path}|{hit.title}]]{heading}")
    return "\n".join(lines)


def hits_as_json(hits: list[SearchHit]) -> str:
    """Retrieval results as structured JSON — for A/B runs, not for humans.

    The prose form above is diffable by eye but not by machine: no stable
    field boundaries, and no way to aggregate "how many hits came from graph
    expansion" across a question set.
    """
    return json.dumps(
        [
            {
                "rank": i,
                "score": hit.score,
                "source": hit.source,
                "path": hit.path,
                "title": hit.title,
                "heading_path": hit.heading_path,
                "doc_id": hit.doc_id,
                "chunk_id": hit.chunk_id,
                "sensitivity": hit.sensitivity,
            }
            for i, hit in enumerate(hits, start=1)
        ],
        indent=2,
    )
