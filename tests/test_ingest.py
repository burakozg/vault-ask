"""Change-detection correctness — the one thing README calls out as verifiable
here and nowhere later (build order, step 1).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import struct
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest

from vault_ask.config import Settings
from vault_ask.db import connect
from vault_ask.embed import EmbeddingSpaceChanged
from vault_ask.ingest import MassDeletionRefused, run_ingest
from vault_ask.vault import Entry


class FakeVault:
    """A vault double: same `list_prefix`/`read` surface, no network."""

    def __init__(self, files: dict[str, tuple[str, str]], *, torn: set[str] | None = None) -> None:
        # path -> (rev, markdown)
        self._files = files
        self._torn = torn or set()
        self.reads: list[str] = []

    async def list_prefix(self, prefix: str) -> list[Entry]:
        out = [
            Entry(doc_id=path.lower(), path=path, rev=rev, children=(), mtime=1000)
            for path, (rev, _md) in self._files.items()
            if path.startswith(prefix)
        ]
        return sorted(out, key=lambda e: e.doc_id)

    async def read(self, entry: Entry) -> str | None:
        self.reads.append(entry.path)
        if entry.path in self._torn:
            return None
        return self._files[entry.path][1]


def _hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def _seed_doc(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    path: str,
    rev: str,
    markdown: str,
    sensitivity: str = "open",
) -> None:
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (doc_id, path, path, rev, _hash(markdown), sensitivity, 1000, json.dumps({})),
    )
    conn.commit()


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


class _FakeEmbedding:
    def __init__(self, *, raises: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raises = raises

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._raises:
            raise RuntimeError("embedding host unreachable")
        data = [{"embedding": [1.0, 0.0, 0.0, 0.0]} for _ in kwargs["input"]]
        return SimpleNamespace(data=data)


class TestNew:
    async def test_new_doc_is_read_and_planned(self, db: sqlite3.Connection, cfg: Settings) -> None:
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "# Anthropic\n\nbody")})
        report = await run_ingest(vault, db, cfg, dry_run=True)
        assert [e.path for e in report.new] == ["10 raw/Anthropic.md"]
        assert vault.reads == ["10 raw/Anthropic.md"]

    async def test_new_doc_applied_lands_in_docs_table(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "# Anthropic\n\nbody")})
        await run_ingest(vault, db, cfg, dry_run=False)
        row = db.execute("SELECT * FROM docs WHERE doc_id = ?", ("10 raw/anthropic.md",)).fetchone()
        assert row is not None
        assert row["rev"] == "rev-1"
        assert row["content_hash"] == _hash("# Anthropic\n\nbody")


class TestUnchanged:
    async def test_same_rev_is_never_read(self, db: sqlite3.Connection, cfg: Settings) -> None:
        markdown = "# Anthropic\n\nbody"
        _seed_doc(db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown=markdown)
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", markdown)})
        report = await run_ingest(vault, db, cfg, dry_run=True)
        assert [e.path for e in report.unchanged] == ["10 raw/Anthropic.md"]
        assert vault.reads == []  # the whole point: no read, no cost


class TestTouched:
    async def test_rev_differs_hash_same_touches_cache_only(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        markdown = "# Anthropic\n\nbody"
        _seed_doc(db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown=markdown)
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-2", markdown)})  # LiveSync rewrote rev; same text
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert [e.path for e in report.touched] == ["10 raw/Anthropic.md"]
        assert vault.reads == ["10 raw/Anthropic.md"]  # rev differs, so a read was required
        row = db.execute("SELECT rev FROM docs WHERE doc_id = ?", ("10 raw/anthropic.md",)).fetchone()
        assert row["rev"] == "rev-2"


class TestChanged:
    async def test_rev_and_hash_differ_is_a_full_reindex(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        _seed_doc(
            db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown="old body"
        )
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-2", "new body")})
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert [e.path for e in report.changed] == ["10 raw/Anthropic.md"]
        row = db.execute("SELECT rev, content_hash FROM docs WHERE doc_id = ?", ("10 raw/anthropic.md",)).fetchone()
        assert row["rev"] == "rev-2"
        assert row["content_hash"] == _hash("new body")

    async def test_changed_doc_drops_its_stale_chunks(self, db: sqlite3.Connection, cfg: Settings) -> None:
        _seed_doc(
            db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown="old body"
        )
        db.execute(
            "INSERT INTO chunks (chunk_id, doc_id, ordinal, heading_path, prelude, body, sensitivity) "
            "VALUES ('c1', '10 raw/anthropic.md', 0, '', '', 'stale', 'open')"
        )
        db.commit()
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-2", "new body")})
        await run_ingest(vault, db, cfg, dry_run=False)
        # The stale chunk is gone; it was replaced by freshly chunked content
        # from "new body" (the chunker itself is exercised in test_chunk.py).
        rows = db.execute("SELECT chunk_id, body FROM chunks").fetchall()
        assert [dict(r) for r in rows] != [{"chunk_id": "c1", "body": "stale"}]
        assert all(row["chunk_id"] != "c1" for row in rows)


class TestDeleted:
    async def test_gone_from_listing_is_deleted(self, db: sqlite3.Connection, cfg: Settings) -> None:
        _seed_doc(
            db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown="body"
        )
        vault = FakeVault({})  # no longer in the vault (or soft-deleted, already filtered by list_prefix)
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert report.deleted == ["10 raw/anthropic.md"]
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0

    async def test_dry_run_deletes_nothing(self, db: sqlite3.Connection, cfg: Settings) -> None:
        _seed_doc(
            db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown="body"
        )
        vault = FakeVault({})
        report = await run_ingest(vault, db, cfg, dry_run=True)
        assert report.deleted == ["10 raw/anthropic.md"]
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 1


class TestRebuild:
    async def test_rebuild_rereads_even_when_rev_and_hash_match(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        markdown = "# Anthropic\n\nbody"
        _seed_doc(db, doc_id="10 raw/anthropic.md", path="10 raw/Anthropic.md", rev="rev-1", markdown=markdown)
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", markdown)})
        report = await run_ingest(vault, db, cfg, dry_run=True, rebuild=True)
        assert vault.reads == ["10 raw/Anthropic.md"]
        assert [e.path for e in report.changed] == ["10 raw/Anthropic.md"]
        assert report.unchanged == []

    async def test_rebuild_still_detects_deletions(self, db: sqlite3.Connection, cfg: Settings) -> None:
        _seed_doc(db, doc_id="10 raw/gone.md", path="10 raw/Gone.md", rev="rev-1", markdown="body")
        vault = FakeVault({})
        report = await run_ingest(vault, db, cfg, dry_run=True, rebuild=True)
        assert report.deleted == ["10 raw/gone.md"]


class TestTornNote:
    async def test_torn_note_is_skipped_not_crashed(self, db: sqlite3.Connection, cfg: Settings) -> None:
        vault = FakeVault(
            {"10 raw/Broken.md": ("rev-1", "unused")}, torn={"10 raw/Broken.md"}
        )
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert report.skipped_torn == ["10 raw/Broken.md"]
        assert report.new == []
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0


class TestSensitivity:
    async def test_personal_path_classified_on_write(self, db: sqlite3.Connection, cfg: Settings) -> None:
        vault = FakeVault({"Tastings/laphroaig-10.md": ("rev-1", "peaty")})
        await run_ingest(vault, db, cfg, dry_run=False)
        row = db.execute(
            "SELECT sensitivity FROM docs WHERE doc_id = ?", ("tastings/laphroaig-10.md",)
        ).fetchone()
        assert row["sensitivity"] == "personal"

    async def test_case_mismatch_does_not_classify(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        """`personal_paths` matching is case-sensitive, and getting that wrong
        fails open rather than loudly — which is how this project shipped with
        every one of 2,254 notes classified `open`. Pinned so the trap is
        visible in the suite rather than only in production.
        """
        vault = FakeVault({"tastings/laphroaig-10.md": ("rev-1", "peaty")})
        await run_ingest(vault, db, cfg, dry_run=False)
        row = db.execute(
            "SELECT sensitivity FROM docs WHERE doc_id = ?", ("tastings/laphroaig-10.md",)
        ).fetchone()
        assert row["sensitivity"] == "open"

    async def test_unmatched_personal_path_warns(
        self, db: sqlite3.Connection, cfg: Settings, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The guard that would have caught the original drift."""
        vault = FakeVault({"10 raw/Public.md": ("rev-1", "body")})
        with caplog.at_level(logging.WARNING, logger="vault_ask.ingest"):
            await run_ingest(vault, db, cfg, dry_run=False)
        warned = [r for r in caplog.records if "personal_path_matches_nothing" in r.getMessage()]
        # Read the pattern off the record's args rather than parsing the
        # rendered message — a pattern can contain spaces ("30 projects/**").
        assert {"Tastings/**", "30 projects/**"} == {str(r.args[0]) for r in warned if r.args}

    async def test_frontmatter_override_classified_on_write(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        md = "---\nsensitivity: personal\n---\n\nbody"
        vault = FakeVault({"10 raw/Sensitive.md": ("rev-1", md)})
        await run_ingest(vault, db, cfg, dry_run=False)
        row = db.execute(
            "SELECT sensitivity FROM docs WHERE doc_id = ?", ("10 raw/sensitive.md",)
        ).fetchone()
        assert row["sensitivity"] == "personal"


class TestFrontmatterEdgeCases:
    """Real vault, real bug: YAML happily parses an unquoted `date:` or
    `title:` value into a Python date, which json.dumps and sqlite3's
    parameter binding both refuse outright — found indexing a real journal
    note, where this crashed the whole ingest transaction.
    """

    async def test_unquoted_date_frontmatter_value_does_not_crash(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        md = "---\ndate: 2026-08-28\n---\n\ndear diary"
        vault = FakeVault({"30 journal/2026-08-28.md": ("rev-1", md)})
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert report.new
        row = db.execute(
            "SELECT frontmatter FROM docs WHERE doc_id = ?", ("30 journal/2026-08-28.md",)
        ).fetchone()
        assert json.loads(row["frontmatter"]) == {"date": "2026-08-28"}

    async def test_unquoted_date_as_title_is_coerced_to_string(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        md = "---\ntitle: 2026-08-28\n---\n\nbody"
        vault = FakeVault({"30 journal/2026-08-28.md": ("rev-1", md)})
        await run_ingest(vault, db, cfg, dry_run=False)
        row = db.execute(
            "SELECT title FROM docs WHERE doc_id = ?", ("30 journal/2026-08-28.md",)
        ).fetchone()
        assert row["title"] == "2026-08-28"


class TestCorpusFiltering:
    async def test_excluded_path_never_reaches_the_plan(self, db: sqlite3.Connection, cfg: Settings) -> None:
        vault = FakeVault({"00 inbox/draft.md": ("rev-1", "not ready")})
        report = await run_ingest(vault, db, cfg, dry_run=True)
        assert report.new == []
        assert vault.reads == []


class TestEmbedding:
    async def test_unconfigured_embedding_is_skipped(
        self, db: sqlite3.Connection, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "## History\n\nfounded in 2021")})
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert report.embedded == 0
        assert fake.calls == []

    async def test_new_doc_gets_embedded(
        self, vec_db: sqlite3.Connection, cfg_with_embedding: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "## History\n\nfounded in 2021")})
        report = await run_ingest(vault, vec_db, cfg_with_embedding, dry_run=False)
        assert report.embedded == 1
        assert vec_db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 1

    async def test_catches_up_a_doc_chunked_before_embedding_was_configured(
        self, vec_db: sqlite3.Connection, cfg: Settings, cfg_with_embedding: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "## History\n\nfounded in 2021")})
        # First run: no embedding host configured yet — chunks exist, no vectors.
        await run_ingest(vault, vec_db, cfg, dry_run=False)
        assert vec_db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 0

        # Second run: same rev (would be "unchanged" and skip re-chunking
        # entirely), but embedding is now configured — the catch-up pass finds
        # it via the index itself, not via rev/hash change detection.
        fake = _FakeEmbedding()
        monkeypatch.setattr(litellm, "aembedding", fake)
        report = await run_ingest(vault, vec_db, cfg_with_embedding, dry_run=False)
        assert report.unchanged  # confirms this really did skip the normal path
        assert report.embedded == 1
        assert vec_db.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0] == 1

    async def test_embed_failure_does_not_lose_the_docs_and_chunks(
        self, vec_db: sqlite3.Connection, cfg_with_embedding: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "aembedding", _FakeEmbedding(raises=True))
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "## History\n\nfounded in 2021")})
        report = await run_ingest(vault, vec_db, cfg_with_embedding, dry_run=False)
        assert report.embedded == 0
        assert report.embed_failed == 1
        # The doc and its chunks are still there — only the vector is missing.
        assert vec_db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 1
        assert vec_db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

    async def test_changed_embedding_model_without_rebuild_raises(
        self, vec_db: sqlite3.Connection, cfg_with_embedding: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(litellm, "aembedding", _FakeEmbedding())
        vault = FakeVault({"10 raw/Anthropic.md": ("rev-1", "## History\n\nfounded in 2021")})
        await run_ingest(vault, vec_db, cfg_with_embedding, dry_run=False)

        changed = cfg_with_embedding.model_copy(
            update={"models": cfg_with_embedding.models.model_copy(update={"embedding_dim": 8})}
        )
        with pytest.raises(EmbeddingSpaceChanged):
            await run_ingest(vault, vec_db, changed, dry_run=False, rebuild=False)


class TestEdgesIntegration:
    async def test_wikilink_between_two_docs_in_the_same_run_resolves(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        # "A" links forward to "B", which sorts after it — exactly the
        # same-batch ordering case resolve_pending exists for.
        vault = FakeVault(
            {
                "10 raw/A.md": ("rev-1", "see [[10 raw/B.md|B]]"),
                "10 raw/B.md": ("rev-1", "nothing here"),
            }
        )
        report = await run_ingest(vault, db, cfg, dry_run=False)
        row = db.execute(
            "SELECT dst, resolved FROM edges WHERE src = ? AND kind = 'wikilink'",
            ("10 raw/a.md",),
        ).fetchone()
        assert (row["dst"], row["resolved"]) == ("10 raw/b.md", 1)
        assert report.edges_resolved == 1

    async def test_deleted_doc_loses_its_outbound_edges(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        vault = FakeVault(
            {
                "10 raw/A.md": ("rev-1", "see [[10 raw/B.md]]"),
                "10 raw/B.md": ("rev-1", "body"),
            }
        )
        await run_ingest(vault, db, cfg, dry_run=False)
        assert db.execute("SELECT COUNT(*) FROM edges WHERE src = ?", ("10 raw/a.md",)).fetchone()[0] == 1

        vault_without_a = FakeVault({"10 raw/B.md": ("rev-1", "body")})
        await run_ingest(vault_without_a, db, cfg, dry_run=False)
        assert db.execute("SELECT COUNT(*) FROM edges WHERE src = ?", ("10 raw/a.md",)).fetchone()[0] == 0

    async def test_tag_edges_written_from_frontmatter(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        md = "---\ntags: [ai, vendor]\n---\n\nbody"
        vault = FakeVault({"10 raw/A.md": ("rev-1", md)})
        await run_ingest(vault, db, cfg, dry_run=False)
        rows = db.execute(
            "SELECT dst FROM edges WHERE src = ? AND kind = 'tag'", ("10 raw/a.md",)
        ).fetchall()
        assert {r["dst"] for r in rows} == {"tag:ai", "tag:vendor"}


class TestReclassification:
    """A config change must reach docs that did not change.

    Change detection is rev-based, so after editing `personal_paths` every doc
    is `unchanged`, nothing is re-read, and without this pass the new rule
    silently does not apply — the privacy control looks fixed and is not.
    """

    async def test_config_change_reclassifies_unchanged_docs(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        vault = FakeVault({"Archive/secret.md": ("rev-1", "body")})
        await run_ingest(vault, db, cfg, dry_run=False)
        assert db.execute(
            "SELECT sensitivity FROM docs WHERE doc_id = ?", ("archive/secret.md",)
        ).fetchone()["sensitivity"] == "open"

        # Same vault, same revs — only config moved.
        wider = cfg.model_copy(
            update={"sensitivity": cfg.sensitivity.model_copy(update={"personal_paths": ["Archive/**"]})}
        )
        report = await run_ingest(vault, db, wider, dry_run=False)

        assert len(report.unchanged) == 1, "doc must not have been re-read"
        assert report.reclassified == 1
        assert db.execute(
            "SELECT sensitivity FROM docs WHERE doc_id = ?", ("archive/secret.md",)
        ).fetchone()["sensitivity"] == "personal"
        # Denormalised copy on chunks must move too — it is what retrieval filters on.
        assert db.execute(
            "SELECT DISTINCT sensitivity FROM chunks WHERE doc_id = ?", ("archive/secret.md",)
        ).fetchone()["sensitivity"] == "personal"

    async def test_reclassification_does_not_drop_embeddings(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        """The reason this is an UPDATE and not a replace_chunks rewrite."""
        vault = FakeVault({"Archive/secret.md": ("rev-1", "body")})
        await run_ingest(vault, db, cfg, dry_run=False)
        chunk_ids = [r["chunk_id"] for r in db.execute("SELECT chunk_id FROM chunks").fetchall()]
        assert chunk_ids
        for cid in chunk_ids:
            db.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (cid, struct.pack("1024f", *([0.0] * 1024))),
            )
        db.commit()

        wider = cfg.model_copy(
            update={"sensitivity": cfg.sensitivity.model_copy(update={"personal_paths": ["Archive/**"]})}
        )
        await run_ingest(vault, db, wider, dry_run=False)

        surviving = db.execute("SELECT COUNT(*) AS n FROM chunks_vec").fetchone()["n"]
        assert surviving == len(chunk_ids), "reclassification must not force a re-embed"

    async def test_no_change_reports_zero(self, db: sqlite3.Connection, cfg: Settings) -> None:
        vault = FakeVault({"10 raw/Public.md": ("rev-1", "body")})
        await run_ingest(vault, db, cfg, dry_run=False)
        report = await run_ingest(vault, db, cfg, dry_run=False)
        assert report.reclassified == 0


class TestMassDeletionGuard:
    """Deletion is a set-difference, so anything that shortens the listing
    deletes the index. VaultUnavailable covers transport errors and non-200s;
    a *successful* 200 with empty or truncated rows looks exactly like "the
    user deleted everything". Refusing wrongly costs a re-run; proceeding
    wrongly costs a full re-embed of the vault.
    """

    async def _seed_many(self, db: sqlite3.Connection, n: int) -> None:
        for i in range(n):
            _seed_doc(
                db, doc_id=f"10 raw/n{i}.md", path=f"10 raw/n{i}.md", rev="rev-1", markdown="body"
            )

    async def test_empty_listing_against_a_real_index_is_refused(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        await self._seed_many(db, 40)
        with pytest.raises(MassDeletionRefused, match="refusing to delete 40 of 40"):
            await run_ingest(FakeVault({}), db, cfg, dry_run=False)
        # Nothing applied — the index must survive the refusal intact.
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 40

    async def test_refusal_also_fires_on_dry_run(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        """A dry run exists to find out what an apply would do."""
        await self._seed_many(db, 40)
        with pytest.raises(MassDeletionRefused):
            await run_ingest(FakeVault({}), db, cfg, dry_run=True)

    async def test_below_the_fraction_is_allowed(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        await self._seed_many(db, 40)
        keep = {f"10 raw/n{i}.md": ("rev-1", "body") for i in range(36)}  # drop 4 of 40 = 10%
        report = await run_ingest(FakeVault(keep), db, cfg, dry_run=False)
        assert len(report.deleted) == 4
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 36

    async def test_small_index_is_not_guarded(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        """1 of 3 docs is 33% and entirely routine — the fraction only means
        something on an index big enough for a wipe to hurt."""
        await self._seed_many(db, 3)
        report = await run_ingest(FakeVault({}), db, cfg, dry_run=False)
        assert len(report.deleted) == 3
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0

    async def test_explicit_override_permits_it(
        self, db: sqlite3.Connection, cfg: Settings
    ) -> None:
        await self._seed_many(db, 40)
        report = await run_ingest(
            FakeVault({}), db, cfg, dry_run=False, allow_mass_delete=True
        )
        assert len(report.deleted) == 40
        assert db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] == 0
