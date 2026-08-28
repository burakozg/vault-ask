from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import litellm
import pytest

from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig, Settings
from vault_ask.db import connect
from vault_ask.embed import (
    EmbeddingSpaceChanged,
    embed_missing_chunks,
    embed_texts,
    ensure_embedding_space,
)

CHUNK_CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite", embedding_dim=4)
    yield conn
    conn.close()


@pytest.fixture
def cfg() -> Settings:
    return Settings(
        models={
            "embedding": "ollama/bge-m3",
            "embedding_base_url": "http://ollama.local:11434",
            "embedding_dim": 4,
        }
    )


class _FakeEmbedding:
    def __init__(self, dim: int = 4) -> None:
        self.calls: list[dict[str, Any]] = []
        self._dim = dim

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        data = [{"embedding": [float(i)] * self._dim} for i, _ in enumerate(kwargs["input"], start=1)]
        return type("Resp", (), {"data": data})()


def _seed_doc(conn: sqlite3.Connection, doc_id: str, path: str, markdown: str) -> None:
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, 'rev-1', 'hash', 'open', 0, '{}')",
        (doc_id, path, path),
    )
    replace_chunks(
        conn, doc_id=doc_id, path=path, title=path, sensitivity="open", markdown=markdown, cfg=CHUNK_CFG
    )
    conn.commit()


class TestEmbedTexts:
    async def test_calls_litellm_with_configured_model_and_base_url(
        self, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        vectors = await embed_texts(cfg, ["hello", "world"])
        assert len(vectors) == 2
        assert fake.calls[0]["model"] == "ollama/bge-m3"
        assert fake.calls[0]["api_base"] == "http://ollama.local:11434"
        assert fake.calls[0]["input"] == ["hello", "world"]

    async def test_empty_input_short_circuits(
        self, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        assert await embed_texts(cfg, []) == []
        assert fake.calls == []


class TestEmbedMissingChunks:
    async def test_embeds_chunks_with_no_vector(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_doc(db, "d1", "p.md", "## History\n\nfounded in 2021")
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        n = await embed_missing_chunks(db, cfg, "d1")
        db.commit()
        assert n == 1
        row = db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()
        assert row[0] == 1

    async def test_prelude_and_body_are_both_sent(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_doc(db, "d1", "10 raw/Anthropic.md", "## History\n\nfounded in 2021")
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        await embed_missing_chunks(db, cfg, "d1")
        sent = fake.calls[0]["input"][0]
        assert "10 raw/Anthropic.md" in sent  # prelude
        assert "founded in 2021" in sent  # body

    async def test_already_embedded_chunks_are_skipped(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed_doc(db, "d1", "p.md", "## History\n\nfounded in 2021")
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        await embed_missing_chunks(db, cfg, "d1")
        db.commit()
        n = await embed_missing_chunks(db, cfg, "d1")  # nothing left to do
        assert n == 0
        assert len(fake.calls) == 1  # only the first call happened

    async def test_no_chunks_is_a_noop(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        assert await embed_missing_chunks(db, cfg, "no-such-doc") == 0
        assert fake.calls == []


class TestEnsureEmbeddingSpace:
    def test_fresh_index_records_the_model_and_dim(self, db: sqlite3.Connection, cfg: Settings) -> None:
        ensure_embedding_space(db, cfg, rebuild=False)
        assert db.execute("SELECT value FROM meta WHERE key='embedding_model'").fetchone()[0] == (
            "ollama/bge-m3"
        )
        assert db.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()[0] == "4"

    def test_matching_model_and_dim_is_a_noop(self, db: sqlite3.Connection, cfg: Settings) -> None:
        ensure_embedding_space(db, cfg, rebuild=False)
        ensure_embedding_space(db, cfg, rebuild=False)  # must not raise

    def test_changed_model_without_rebuild_raises(self, db: sqlite3.Connection, cfg: Settings) -> None:
        ensure_embedding_space(db, cfg, rebuild=False)
        changed = cfg.model_copy(update={"models": cfg.models.model_copy(update={"embedding": "ollama/other"})})
        with pytest.raises(EmbeddingSpaceChanged):
            ensure_embedding_space(db, changed, rebuild=False)

    def test_changed_dim_without_rebuild_raises(self, db: sqlite3.Connection, cfg: Settings) -> None:
        ensure_embedding_space(db, cfg, rebuild=False)
        changed = cfg.model_copy(update={"models": cfg.models.model_copy(update={"embedding_dim": 8})})
        with pytest.raises(EmbeddingSpaceChanged):
            ensure_embedding_space(db, changed, rebuild=False)

    def test_changed_dim_with_rebuild_recreates_the_table(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        ensure_embedding_space(db, cfg, rebuild=False)
        db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            ("stale", struct.pack("4f", 1.0, 0.0, 0.0, 0.0)),
        )
        db.commit()

        changed = cfg.model_copy(update={"models": cfg.models.model_copy(update={"embedding_dim": 8})})
        ensure_embedding_space(db, changed, rebuild=True)

        # The old (wrong-width) row is gone — a fresh table, not an altered one.
        assert db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0
        # And it now accepts 8-wide vectors.
        db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            ("fresh", struct.pack("8f", *([0.0] * 8))),
        )
        db.commit()
        assert db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 1
