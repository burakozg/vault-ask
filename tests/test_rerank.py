from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from vault_ask.ask import retrieve
from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig, Settings
from vault_ask.db import connect
from vault_ask.rerank import apply_grades, parse_grades, rerank
from vault_ask.retrieval import SearchHit

CHUNKING = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


def _hit(chunk_id: str) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        doc_id=chunk_id,
        path=f"{chunk_id}.md",
        title=chunk_id,
        heading_path="",
        prelude="",
        body=f"body of {chunk_id}",
        sensitivity="open",
        score=1.0,
    )


class _Fake:
    def __init__(self, text: str | None = None, *, raises: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text
        self._raises = raises

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises:
            raise RuntimeError("429")
        message = SimpleNamespace(content=self._text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class TestParseGrades:
    def test_plain_array(self) -> None:
        assert parse_grades("[3, 0, 2]", 3) == [3, 0, 2]

    def test_array_inside_a_code_fence(self) -> None:
        assert parse_grades("```json\n[1,2]\n```", 2) == [1, 2]

    def test_bare_comma_separated_floats(self) -> None:
        # What gemini-2.5-flash-lite actually returned in production.
        assert parse_grades("1.0, 0.0, 3.0", 3) == [1, 0, 3]

    @pytest.mark.parametrize(
        "text", ["[1, 2]", "[1, 2, 9]", "[1, 2, 3.5]", '["a","b","c"]', "[-1, 1, 2]", "nope", ""]
    )
    def test_rejects_anything_that_is_not_n_grades(self, text: str) -> None:
        assert parse_grades(text, 3) is None


class TestApplyGrades:
    def test_orders_by_grade_then_original_order_and_drops_zeros(self) -> None:
        hits = [_hit("a"), _hit("b"), _hit("c"), _hit("d")]
        out = apply_grades(hits, [1, 0, 3, 1])
        assert [h.chunk_id for h in out] == ["c", "a", "d"]

    def test_all_zero_is_ignored_not_obeyed(self) -> None:
        hits = [_hit("a"), _hit("b")]
        assert apply_grades(hits, [0, 0]) == hits


class TestRerank:
    async def test_returns_pruned_reordered_hits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _Fake("[0, 3, 2]")
        monkeypatch.setattr(litellm, "acompletion", fake)
        out = await rerank(Settings(), "q", [_hit("a"), _hit("b"), _hit("c")])
        assert out is not None
        assert [h.chunk_id for h in out] == ["b", "c"]
        assert fake.calls[0]["model"] == Settings().models.rerank

    async def test_model_failure_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "acompletion", _Fake(raises=True))
        assert await rerank(Settings(), "q", [_hit("a"), _hit("b")]) is None

    async def test_unparseable_reply_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(litellm, "acompletion", _Fake("I think the first one."))
        assert await rerank(Settings(), "q", [_hit("a"), _hit("b")]) is None

    async def test_single_candidate_makes_no_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _Fake("[3]")
        monkeypatch.setattr(litellm, "acompletion", fake)
        assert await rerank(Settings(), "q", [_hit("a")]) is None
        assert fake.calls == []


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite")
    for doc_id, sensitivity in (("d1", "open"), ("d2", "open"), ("d3", "personal")):
        conn.execute(
            "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity) "
            "VALUES (?, ?, ?, 'r', 'h', ?)",
            (doc_id, f"{doc_id}.md", doc_id, sensitivity),
        )
        replace_chunks(
            conn,
            doc_id=doc_id,
            path=f"{doc_id}.md",
            title=doc_id,
            sensitivity=sensitivity,
            markdown="## H\n\nwidget notes",
            cfg=CHUNKING,
        )
    conn.commit()
    yield conn
    conn.close()


class TestRetrieveWithRerank:
    async def test_irrelevant_candidates_are_dropped(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VAULTASK_RETRIEVAL__RERANK_ENABLED", "true")
        fake = _Fake("[0, 3, 3]")
        monkeypatch.setattr(litellm, "acompletion", fake)
        hits = await retrieve(db, Settings(), "widget", allow_web=False)
        assert len(fake.calls) == 1
        assert len(hits) == 2

    async def test_rerank_failure_falls_back_to_fused_order(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VAULTASK_RETRIEVAL__RERANK_ENABLED", "true")
        monkeypatch.setattr(litellm, "acompletion", _Fake(raises=True))
        hits = await retrieve(db, Settings(), "widget", allow_web=False)
        assert len(hits) == 3

    async def test_disabled_makes_no_call(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _Fake("[3, 3, 3]")
        monkeypatch.setattr(litellm, "acompletion", fake)
        await retrieve(db, Settings(), "widget", allow_web=False)
        assert fake.calls == []

    async def test_web_mode_never_sends_personal_chunks_to_the_reranker(
        self, db: sqlite3.Connection, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VAULTASK_RETRIEVAL__RERANK_ENABLED", "true")
        fake = _Fake("[3, 3]")
        monkeypatch.setattr(litellm, "acompletion", fake)
        hits = await retrieve(db, Settings(), "widget", allow_web=True)
        assert {h.sensitivity for h in hits} == {"open"}
        assert "d3" not in fake.calls[0]["messages"][1]["content"]
