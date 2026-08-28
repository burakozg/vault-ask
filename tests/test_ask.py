from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from vault_ask.ask import VAULT_SILENT, _merge, ask, retrieve
from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig, Settings
from vault_ask.db import connect
from vault_ask.retrieval import SearchHit

CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite")
    yield conn
    conn.close()


@pytest.fixture
def vec_db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "vec.sqlite", embedding_dim=4)
    yield conn
    conn.close()


@pytest.fixture
def cfg() -> Settings:
    return Settings()


@pytest.fixture
def cfg_with_embedding() -> Settings:
    return Settings(
        models={
            "embedding": "ollama/bge-m3",
            "embedding_base_url": "http://ollama.local:11434",
            "embedding_dim": 4,
        }
    )


def _seed_edge(conn: sqlite3.Connection, src: str, dst: str, kind: str = "topic") -> None:
    conn.execute(
        "INSERT INTO edges (src, dst, kind, resolved) VALUES (?, ?, ?, 1)", (src, dst, kind)
    )
    conn.commit()


def _seed(conn: sqlite3.Connection, doc_id: str, path: str, title: str, sensitivity: str, body: str) -> None:
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, 'rev-1', 'hash', ?, 0, '{}')",
        (doc_id, path, title, sensitivity),
    )
    replace_chunks(
        conn, doc_id=doc_id, path=path, title=title, sensitivity=sensitivity, markdown=body, cfg=CFG
    )
    conn.commit()


class _FakeCompletion:
    def __init__(self, text: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeEmbedding:
    def __init__(self, vector: list[float] | None = None, *, raises: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._vector = vector or [1.0, 0.0, 0.0, 0.0]
        self._raises = raises

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises:
            raise RuntimeError("embedding host unreachable")
        data = [{"embedding": self._vector} for _ in kwargs["input"]]
        return SimpleNamespace(data=data)


class TestVaultSilent:
    async def test_empty_index_returns_silent_without_calling_the_model(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeCompletion("should never be seen")
        monkeypatch.setattr(litellm, "acompletion", fake)
        answer = await ask(db, cfg, "what have I read about Entra ID?")
        assert answer.text == VAULT_SILENT
        assert answer.generated is False
        assert fake.calls == []


class TestDryRun:
    async def test_dry_run_returns_hits_without_calling_the_model(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        fake = _FakeCompletion("should never be seen")
        monkeypatch.setattr(litellm, "acompletion", fake)
        answer = await ask(db, cfg, "founded", dry_run=True)
        assert answer.generated is False
        assert fake.calls == []
        assert "[[10 raw/Anthropic.md|Anthropic]]" in answer.text
        assert len(answer.hits) == 1


class TestSensitivity:
    async def test_allow_web_true_excludes_personal_chunks(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db, "d1", "30 journal/2026-08-28.md", "Journal", "personal", "## Entry\n\nfounded a company today")
        fake = _FakeCompletion("should never be seen")
        monkeypatch.setattr(litellm, "acompletion", fake)
        answer = await ask(db, cfg, "founded", allow_web=True, dry_run=True)
        assert answer.text == VAULT_SILENT  # the only hit is personal, and web is allowed

    async def test_allow_web_false_includes_personal_chunks(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db, "d1", "30 journal/2026-08-28.md", "Journal", "personal", "## Entry\n\nfounded a company today")
        fake = _FakeCompletion("should never be seen")
        monkeypatch.setattr(litellm, "acompletion", fake)
        answer = await ask(db, cfg, "founded", allow_web=False, dry_run=True)
        assert len(answer.hits) == 1


class TestGeneration:
    async def test_calls_the_model_with_context_and_returns_its_text(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        fake = _FakeCompletion("Anthropic was founded in 2021 [[10 raw/Anthropic.md|Anthropic]].")
        monkeypatch.setattr(litellm, "acompletion", fake)
        answer = await ask(db, cfg, "when was Anthropic founded?")
        assert answer.generated is True
        assert answer.text == "Anthropic was founded in 2021 [[10 raw/Anthropic.md|Anthropic]]."
        assert len(fake.calls) == 1
        assert fake.calls[0]["model"] == cfg.models.generation
        messages = fake.calls[0]["messages"]
        assert messages[0]["role"] == "system"
        assert "wikilink" in messages[0]["content"]
        assert "founded in 2021" in messages[1]["content"]


class TestHybridRetrieval:
    async def test_embedding_unconfigured_never_calls_aembedding(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(db, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        fake_embed = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake_embed)
        monkeypatch.setattr(litellm, "acompletion", _FakeCompletion("ok"))
        await ask(db, cfg, "founded", dry_run=True)
        assert fake_embed.calls == []

    async def test_vector_and_fts_hits_fuse(
        self, vec_db: sqlite3.Connection, cfg_with_embedding: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(vec_db, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        chunk_id = vec_db.execute("SELECT chunk_id FROM chunks WHERE doc_id='d1'").fetchone()[0]
        vec_db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, struct.pack("4f", 1.0, 0.0, 0.0, 0.0)),
        )
        vec_db.commit()

        fake_embed = _FakeEmbedding([1.0, 0.0, 0.0, 0.0])
        monkeypatch.setattr(litellm, "aembedding", fake_embed)
        monkeypatch.setattr(litellm, "acompletion", _FakeCompletion("ok"))

        answer = await ask(vec_db, cfg_with_embedding, "founded", dry_run=True)
        assert fake_embed.calls  # the vector path was actually exercised
        assert [h.chunk_id for h in answer.hits] == [chunk_id]

    async def test_vector_search_failure_degrades_to_fts_only(
        self, vec_db: sqlite3.Connection, cfg_with_embedding: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(vec_db, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        monkeypatch.setattr(litellm, "aembedding", _FakeEmbedding(raises=True))
        fake_completion = _FakeCompletion("Anthropic was founded in 2021.")
        monkeypatch.setattr(litellm, "acompletion", fake_completion)

        answer = await ask(vec_db, cfg_with_embedding, "founded")
        assert answer.generated is True
        assert answer.text == "Anthropic was founded in 2021."


class TestGraphEnabledSwitch:
    """`graph_max_siblings=0` looks like an off switch and is not: the
    topic-note pull runs before, and independently of, the sibling query, so
    zeroing siblings still injects every topic note. Without a real switch,
    expansion cannot be A/B'd at all.
    """

    async def test_zero_siblings_still_injects_topic_notes(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        _seed(db, "99 topics/t.md", "99 topics/t.md", "T", "open", "## H\n\nunrelated hub prose")
        _seed(db, "hit.md", "hit.md", "Hit", "open", "## H\n\nwhisky")
        _seed_edge(db, "99 topics/t.md", "hit.md")

        narrowed = cfg.model_copy(
            update={"retrieval": cfg.retrieval.model_copy(update={"graph_max_siblings": 0})}
        )
        hits = await retrieve(db, narrowed, "whisky", allow_web=False)
        reached_by_graph = {h.path for h in hits if h.source == "graph"}
        assert "99 topics/t.md" in reached_by_graph, (
            "graph_max_siblings=0 is not an off switch — this is why graph_enabled exists"
        )

    async def test_graph_enabled_false_is_a_real_off_switch(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        _seed(db, "99 topics/t.md", "99 topics/t.md", "T", "open", "## H\n\nunrelated hub prose")
        _seed(db, "hit.md", "hit.md", "Hit", "open", "## H\n\nwhisky")
        _seed_edge(db, "99 topics/t.md", "hit.md")

        off = cfg.model_copy(
            update={"retrieval": cfg.retrieval.model_copy(update={"graph_enabled": False})}
        )
        hits = await retrieve(db, off, "whisky", allow_web=False)
        assert all(h.source != "graph" for h in hits)
        assert "99 topics/t.md" not in {h.path for h in hits}


class TestGraphQuota:
    """Expansion contributes but cannot take over the answer.

    Before the quota, a plain score sort of `fused + expanded` let expanded
    chunks — carrying a score *inherited* from the hit that reached them, never
    earned against the question — evict direct hits 1:1 (measured: 36 for 36
    over 20 real questions, up to 6 of 8 slots).
    """

    def _hit(self, i: int, score: float, source: str) -> SearchHit:
        return SearchHit(
            chunk_id=f"c{i}", doc_id=f"d{i}.md", path=f"d{i}.md", title=f"D{i}",
            heading_path="", prelude="", body="b", sensitivity="open",
            score=score, source=source,
        )

    def test_expanded_chunks_are_capped(self) -> None:
        direct = [self._hit(i, 0.010 - i * 0.0001, "fts") for i in range(8)]
        # Deliberately scored ABOVE every direct hit, as really happens.
        graph = [self._hit(100 + i, 0.020, "graph") for i in range(6)]
        merged = _merge(direct, graph, final_k=8, graph_slots=2)
        assert len(merged) == 8
        assert sum(1 for h in merged if h.source == "graph") == 2

    def test_direct_hits_keep_the_remaining_slots(self) -> None:
        direct = [self._hit(i, 0.010 - i * 0.0001, "fts") for i in range(8)]
        graph = [self._hit(100 + i, 0.020, "graph") for i in range(6)]
        merged = _merge(direct, graph, final_k=8, graph_slots=2)
        assert sum(1 for h in merged if h.source == "fts") == 6

    def test_zero_slots_excludes_expansion_entirely(self) -> None:
        direct = [self._hit(i, 0.010, "fts") for i in range(3)]
        graph = [self._hit(100, 0.020, "graph")]
        merged = _merge(direct, graph, final_k=8, graph_slots=0)
        assert all(h.source != "graph" for h in merged)

    def test_expansion_may_exceed_the_quota_to_avoid_a_thin_answer(self) -> None:
        """Too few direct hits is the case expansion exists for."""
        direct = [self._hit(0, 0.010, "fts")]
        graph = [self._hit(100 + i, 0.005, "graph") for i in range(6)]
        merged = _merge(direct, graph, final_k=8, graph_slots=2)
        assert len(merged) == 7
        assert sum(1 for h in merged if h.source == "graph") == 6

    def test_no_expansion_is_just_the_top_k(self) -> None:
        direct = [self._hit(i, 0.010 - i * 0.0001, "fts") for i in range(20)]
        merged = _merge(direct, [], final_k=8, graph_slots=2)
        assert [h.chunk_id for h in merged] == [f"c{i}" for i in range(8)]
