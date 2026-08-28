"""Web fallback — README calls it "the only part that can leak".

The invariant these tests exist for is not "don't send personal text to a
search engine". It is the stronger structural one from README "Corpus and
sensitivity": **a `personal` chunk is never in a context that also carries web
content.** That holds because one flag governs both — `allow_web=true` admits
web results and simultaneously restricts retrieval to `open` chunks — so the
two can never be true at once. `TestNeverWithPersonalContext` is the test that
would catch a future change decoupling them.

Everything here mocks the search. No test in this suite may touch the network.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import vault_ask.ask as ask_module
import vault_ask.web as web_module
from vault_ask.ask import _should_search_web, ask
from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig, Settings
from vault_ask.db import connect
from vault_ask.retrieval import SearchHit
from vault_ask.web import WebResult, assemble_web_context

CHUNK_CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite")
    yield conn
    conn.close()


def _cfg(**web: Any) -> Settings:
    return Settings(web={"enabled": True, **web}, models={"generation": "openrouter/x/y"})


def _seed(conn: sqlite3.Connection, doc_id: str, sensitivity: str, body: str) -> None:
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, 'r', 'h', ?, 0, '{}')",
        (doc_id, doc_id, doc_id, sensitivity),
    )
    replace_chunks(
        conn, doc_id=doc_id, path=doc_id, title=doc_id,
        sensitivity=sensitivity, markdown=body, cfg=CHUNK_CFG,
    )
    conn.commit()


def _hit(score: float, distance: float | None = 0.5) -> SearchHit:
    return SearchHit(
        chunk_id="c", doc_id="d.md", path="d.md", title="D", heading_path="",
        prelude="", body="b", sensitivity="open", score=score, source="fts",
        distance=distance,
    )


class TestNeverWithPersonalContext:
    """The load-bearing test in this file."""

    def test_allow_web_false_never_searches(self) -> None:
        """allow_web=false is the mode where `personal` chunks are retrievable.
        Searching there would put personal notes and web content in one context
        — the single thing the whole sensitivity design exists to prevent."""
        cfg = _cfg()
        assert _should_search_web(cfg, [], allow_web=False) is False
        assert _should_search_web(cfg, [_hit(0.001)], allow_web=False) is False

    async def test_end_to_end_no_search_when_personal_visible(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db, "Tastings/x.md", "personal", "## H\n\nwhisky notes")
        called: list[str] = []

        async def _spy(cfg: Settings, query: str, **kw: Any) -> list[WebResult]:
            called.append(query)
            return []

        monkeypatch.setattr(ask_module, "web_search", _spy)

        async def _fake(**kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))]
            )

        monkeypatch.setattr(ask_module.litellm, "acompletion", _fake)
        await ask(db, _cfg(), "whisky", allow_web=False)
        assert called == [], "searched the web while personal chunks were retrievable"


class TestThinTrigger:
    def test_disabled_never_searches(self) -> None:
        cfg = Settings(web={"enabled": False})
        assert _should_search_web(cfg, [], allow_web=True) is False

    def test_too_few_hits_triggers(self) -> None:
        cfg = _cfg(thin_hits=3)
        assert _should_search_web(cfg, [_hit(0.9), _hit(0.9)], allow_web=True) is True

    def test_close_matches_do_not_trigger(self) -> None:
        cfg = _cfg(thin_hits=3, thin_distance=1.0)
        hits = [_hit(0.03, 0.81), _hit(0.02, 0.90), _hit(0.02, 0.95)]
        assert _should_search_web(cfg, hits, allow_web=True) is False

    def test_distant_matches_trigger_even_with_enough_hits(self) -> None:
        """Plenty of matches, none of them actually about the question.

        The case a fused-score rule cannot see at all: RRF gives the top hit
        1/61 whether it is a perfect match or nonsense."""
        cfg = _cfg(thin_hits=3, thin_distance=1.0)
        hits = [_hit(0.0164, 1.02), _hit(0.0161, 1.09), _hit(0.0159, 1.13)]
        assert _should_search_web(cfg, hits, allow_web=True) is True

    def test_no_vector_arm_does_not_guess(self) -> None:
        """FTS returning something is not evidence of coverage."""
        cfg = _cfg(thin_hits=1)
        hits = [_hit(0.0164, None), _hit(0.0161, None)]
        assert _should_search_web(cfg, hits, allow_web=True) is False

    def test_silent_vault_triggers(self) -> None:
        assert _should_search_web(_cfg(), [], allow_web=True) is True


class TestWebContext:
    def test_labelled_and_separated(self) -> None:
        out = assemble_web_context([WebResult("T", "https://e.com/a", "snippet")])
        assert "NOT from the vault" in out
        assert "https://e.com/a" in out

    def test_empty_renders_nothing(self) -> None:
        assert assemble_web_context([]) == ""

    def test_citation_is_a_url_never_a_wikilink(self) -> None:
        """A wikilink asserts the user has this note. They do not."""
        cite = WebResult("T", "https://e.com/a", "s").citation()
        assert cite == "https://e.com/a"
        assert "[[" not in cite

    def test_web_block_follows_the_vault_block(self) -> None:
        msgs = ask_module._messages("q", "VAULTCTX", assemble_web_context(
            [WebResult("T", "https://e.com/a", "s")]))
        content = msgs[1]["content"]
        assert content.index("VAULTCTX") < content.index("NOT from the vault")


class TestDegradesGracefully:
    """A search provider that breaks must cost a thinner answer, never a 500.
    This one scrapes an unofficial interface, so it will break eventually."""

    async def test_search_failure_returns_empty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom(*a: Any, **k: Any) -> list[WebResult]:
            raise RuntimeError("ddg changed their HTML")

        monkeypatch.setattr(web_module, "_blocking_search", _boom)
        assert await web_module.search(_cfg(), "q") == []

    async def test_disabled_short_circuits_without_calling_out(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*a: Any, **k: Any) -> list[WebResult]:
            raise AssertionError("must not be called when web.enabled is false")

        monkeypatch.setattr(web_module, "_blocking_search", _boom)
        assert await web_module.search(Settings(web={"enabled": False}), "q") == []

    async def test_vault_silent_and_no_web_costs_no_model_call(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        async def _boom(**kwargs: Any) -> Any:
            raise AssertionError("model called for a vault-silent, web-empty question")

        monkeypatch.setattr(ask_module.litellm, "acompletion", _boom)

        async def _none(cfg: Settings, query: str, **kw: Any) -> list[WebResult]:
            return []

        monkeypatch.setattr(ask_module, "web_search", _none)
        answer = await ask(db, _cfg(), "nothing here", allow_web=True)
        assert answer.text == ask_module.VAULT_SILENT
        assert answer.generated is False

    async def test_web_results_alone_still_answer(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty vault plus web hits is no longer vault-silent — the model
        runs so it can say the material came from the web, not the vault."""
        async def _some(cfg: Settings, query: str, **kw: Any) -> list[WebResult]:
            return [WebResult("T", "https://e.com/a", "s")]

        monkeypatch.setattr(ask_module, "web_search", _some)

        async def _fake(**kwargs: Any) -> Any:
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="from the web"))]
            )

        monkeypatch.setattr(ask_module.litellm, "acompletion", _fake)
        answer = await ask(db, _cfg(), "obscure", allow_web=True)
        assert answer.generated is True
        assert len(answer.web) == 1
