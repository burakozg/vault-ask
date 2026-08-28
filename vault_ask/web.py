"""Web search — the vault's fallback, never its replacement.

This is the part README "Corpus and sensitivity" calls *the only part that can
leak*, which is why it landed last, after the retrieval pipeline's behaviour
was pinned by tests (`tests/test_sensitivity.py`).

Two invariants, both enforced by the caller in `vault_ask.ask` and asserted in
`tests/test_web.py`:

1. **Searching is only ever possible when ``allow_web`` is true**, which is
   exactly the mode in which retrieval is restricted to `open` chunks. The
   invariant is not "don't send personal text to the web" — it is the stronger
   structural one from the README: a `personal` chunk is never placed in a
   context that might *also* carry web content. Gating both on the same flag
   is what makes that true by construction rather than by care.
2. **The vault leads.** Web results are labelled, kept under their own heading,
   and cited by URL rather than as a wikilink, so a reader can always tell
   which claims came from their own notes (see `vault_ask.prompts`).

What this does *not* protect: the question the user typed is sent to the search
provider verbatim. No gate covers that, and none can — it is what searching
means. Worth stating plainly rather than implying the `allow_web` switch makes
web search private.

Best-effort throughout: a failure returns no results and logs, exactly like
`vault_ask.ask._search_vector_safe`. An unreachable search provider must
degrade an answer to vault-only, never fail the request.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .config import Settings

log = logging.getLogger("vault_ask.web")


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str

    def citation(self) -> str:
        """A URL, never a wikilink — see prompts.SYSTEM_PROMPT rule 2.

        A wikilink means "this is in your vault"; using one for a web source
        would make the answer claim the user has a note they do not have.
        """
        return self.url


def _blocking_search(query: str, *, max_results: int, region: str) -> list[WebResult]:
    # Imported here rather than at module scope so the dependency is only
    # touched when web search is actually enabled — and so a broken/renamed
    # upstream cannot stop the whole application importing.
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=max_results, region=region)
    return [
        WebResult(
            title=str(row.get("title") or "").strip(),
            url=str(row.get("href") or "").strip(),
            snippet=str(row.get("body") or "").strip(),
        )
        for row in rows
        if row.get("href")
    ]


async def search(cfg: Settings, query: str, *, max_results: int | None = None) -> list[WebResult]:
    """Search the web, or return [] if that is not possible right now.

    Never raises. `ddgs` is synchronous and does blocking network I/O, so it
    runs in a worker thread — calling it directly would stall the event loop
    and, with it, every other in-flight request on this single-process server.
    """
    if not cfg.web.enabled:
        return []
    limit = max_results if max_results is not None else cfg.web.max_results
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                _blocking_search, query, max_results=limit, region=cfg.web.region
            ),
            timeout=cfg.web.timeout_s,
        )
    except TimeoutError:
        log.warning("web.search_timed_out after=%.1fs query=%r", cfg.web.timeout_s, query)
        return []
    except Exception:
        # Includes the scraping interface breaking, which it eventually will.
        log.warning("web.search_failed query=%r", query, exc_info=True)
        return []

    log.info("web.searched results=%d query=%r", len(results), query)
    return results


def assemble_web_context(results: list[WebResult]) -> str:
    """Web results rendered for the prompt, under their own heading.

    Kept separate from `retrieval.assemble_context` on purpose: the moment the
    two are concatenated by a single function that treats them alike, the
    labelling that keeps vault and web distinguishable becomes one edit away
    from being lost.
    """
    if not results:
        return ""
    parts = ["## From the web (NOT from the vault — cite these by URL)"]
    for result in results:
        parts.append(f"### {result.title}\n{result.url}\n{result.snippet}")
    return "\n\n".join(parts)
