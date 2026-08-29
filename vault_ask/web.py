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

What this does *not* protect: the question the user typed is sent to whichever
search provider is selected, verbatim. No gate covers that, and none can — it
is what searching means. Worth stating plainly rather than implying the
`allow_web` switch makes web search private.

**Providers.** Selectable at `web.provider`, changeable from the admin console
without a redeploy. Their API keys are *not*: keys are secrets, and secrets in
this project are environment-only, never in `overrides.json` and never in a
browser form (see `vault_ask/overrides.py`). The console shows which providers
have a usable key and refuses to select one that does not, rather than saving
the choice and returning nothing after the next restart.

Best-effort throughout: a failure returns no results and logs, exactly like
`vault_ask.ask._search_vector_safe`. An unreachable provider must degrade an
answer to vault-only, never fail the request.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from .config import Settings

log = logging.getLogger("vault_ask.web")

#: provider -> the Settings field holding its key, or None if it needs none.
#: Drives both the availability check here and what the admin console reports,
#: so the two can never disagree about whether a provider is usable.
PROVIDER_KEYS: dict[str, str | None] = {
    "duckduckgo": None,
    "tavily": "tavily_api_key",
    "brave": "brave_api_key",
}

#: The env var to set for each, named in error messages so a rejected choice
#: tells you what to do about it instead of just refusing.
PROVIDER_ENV: dict[str, str] = {
    "tavily": "VAULTASK_TAVILY_API_KEY",
    "brave": "VAULTASK_BRAVE_API_KEY",
}


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


def provider_key(cfg: Settings, provider: str) -> str | None:
    """The configured API key for ``provider``, or None if it has/needs none."""
    field = PROVIDER_KEYS.get(provider)
    if field is None:
        return None
    secret = getattr(cfg, field, None)
    return secret.get_secret_value() if secret else None


def provider_available(cfg: Settings, provider: str) -> bool:
    """Whether ``provider`` could actually run right now.

    DuckDuckGo needs nothing; the others need a key. Used by the admin API to
    reject a selection that would silently produce no results, and by `search`
    to refuse rather than call an API that will 401.
    """
    if provider not in PROVIDER_KEYS:
        return False
    if PROVIDER_KEYS[provider] is None:
        return True
    return bool(provider_key(cfg, provider))


# --- providers -------------------------------------------------------------
#
# Each returns a plain list[WebResult] and may raise; `search` below owns the
# timeout, the thread offload and the error handling, so a provider is only
# ever responsible for "call the thing, shape the rows".


def _search_duckduckgo(cfg: Settings, query: str, limit: int) -> list[WebResult]:
    """No account, no key — and an unofficial scraping interface, so it is
    rate-limited and will break when their HTML changes. Accepted because web
    results are best-effort by construction; a breakage costs a vault-only
    answer, not a failed request.
    """
    # Imported here rather than at module scope so a broken/renamed upstream
    # cannot stop the whole application importing.
    from ddgs import DDGS

    rows = DDGS().text(query, max_results=limit, region=cfg.web.region)
    return [
        WebResult(
            title=str(row.get("title") or "").strip(),
            url=str(row.get("href") or "").strip(),
            snippet=str(row.get("body") or "").strip(),
        )
        for row in rows
        if row.get("href")
    ]


def _search_tavily(cfg: Settings, query: str, limit: int) -> list[WebResult]:
    """Built for retrieval rather than for humans: returns cleaned, summarised
    extracts instead of raw snippets, which is closer to what a vault chunk
    looks like and so blends better in `assemble_web_context`.
    """
    response = httpx.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {provider_key(cfg, 'tavily')}"},
        json={"query": query, "max_results": limit, "search_depth": "basic"},
        timeout=cfg.web.timeout_s,
    )
    response.raise_for_status()
    return [
        WebResult(
            title=str(row.get("title") or "").strip(),
            url=str(row.get("url") or "").strip(),
            snippet=str(row.get("content") or "").strip(),
        )
        for row in response.json().get("results", [])
        if row.get("url")
    ]


def _search_brave(cfg: Settings, query: str, limit: int) -> list[WebResult]:
    """An independent index rather than a Google/Bing reseller. Returns raw
    results, so the snippet is a description rather than an extract.
    """
    response = httpx.get(
        "https://api.search.brave.com/res/v1/web/search",
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": provider_key(cfg, "brave") or "",
        },
        params={"q": query, "count": limit},
        timeout=cfg.web.timeout_s,
    )
    response.raise_for_status()
    rows = (response.json().get("web") or {}).get("results", [])
    return [
        WebResult(
            title=str(row.get("title") or "").strip(),
            url=str(row.get("url") or "").strip(),
            snippet=str(row.get("description") or "").strip(),
        )
        for row in rows
        if row.get("url")
    ]


def _provider_impl(provider: str) -> Callable[[Settings, str, int], list[WebResult]] | None:
    """Resolve the implementation at call time, not at import time.

    A module-level dict would capture the function objects once, so replacing
    one (a test double, or a future runtime swap) would leave the dispatch
    table pointing at the original — the binding would be stale and the
    substitution silently ineffective. Rebuilding three entries per search
    costs nothing and keeps the module attributes the single source of truth.
    """
    return {
        "duckduckgo": _search_duckduckgo,
        "tavily": _search_tavily,
        "brave": _search_brave,
    }.get(provider)


async def search(cfg: Settings, query: str, *, max_results: int | None = None) -> list[WebResult]:
    """Search with the configured provider, or return [] if that is not
    possible right now.

    Never raises. Every provider here does blocking network I/O — `ddgs` is
    synchronous, and httpx is used in its sync form for symmetry — so the call
    runs in a worker thread. Calling it directly would stall the event loop
    and, with it, every other in-flight request on this single-process server.
    """
    if not cfg.web.enabled:
        return []

    provider = cfg.web.provider
    impl = _provider_impl(provider)
    if impl is None:  # pragma: no cover - the config Literal prevents this
        log.warning("web.unknown_provider provider=%s", provider)
        return []
    if not provider_available(cfg, provider):
        # Refuse rather than call an API that will 401. The admin console
        # blocks this selection, so reaching here means the key was set at the
        # time of the choice and has since been removed.
        log.warning(
            "web.provider_unavailable provider=%s — set %s",
            provider,
            PROVIDER_ENV.get(provider, "its API key"),
        )
        return []

    limit = max_results if max_results is not None else cfg.web.max_results
    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(impl, cfg, query, limit), timeout=cfg.web.timeout_s
        )
    except TimeoutError:
        log.warning(
            "web.search_timed_out provider=%s after=%.1fs query=%r",
            provider, cfg.web.timeout_s, query,
        )
        return []
    except Exception:
        # Includes the DuckDuckGo scrape breaking, which it eventually will,
        # and any provider returning an unexpected shape or an HTTP error.
        log.warning("web.search_failed provider=%s query=%r", provider, query, exc_info=True)
        return []

    log.info("web.searched provider=%s results=%d query=%r", provider, len(results), query)
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
