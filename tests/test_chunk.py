from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from vault_ask.chunk import (
    _sections,  # exercises the heading-split step in isolation, before runt-merging
    build_prelude,
    chunk_id,
    chunk_markdown,
    delete_chunks,
    replace_chunks,
)
from vault_ask.config import ChunkingConfig
from vault_ask.db import connect

CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite")
    yield conn
    conn.close()


class TestSections:
    def test_no_headings_is_one_chunk(self) -> None:
        chunks = chunk_markdown("just a paragraph, no headings at all", CFG)
        assert [c.heading_path for c in chunks] == [""]

    def test_splits_on_h2(self) -> None:
        md = "intro text\n\n## First\n\nfirst body\n\n## Second\n\nsecond body"
        chunks = _sections(md)
        assert [c.heading_path for c in chunks] == ["", "First", "Second"]
        assert chunks[1].text == "first body"
        assert chunks[2].text == "second body"

    def test_h3_nests_under_h2(self) -> None:
        md = "## Parent\n\nparent body\n\n### Child\n\nchild body"
        chunks = _sections(md)
        assert [c.heading_path for c in chunks] == ["Parent", "Parent > Child"]

    def test_h3_without_a_parent_h2_stands_alone(self) -> None:
        md = "### Orphan\n\nbody text here"
        chunks = _sections(md)
        assert chunks[0].heading_path == "Orphan"

    def test_second_h2_resets_the_h3_breadcrumb(self) -> None:
        md = (
            "## A\n\na body\n\n### A1\n\na1 body\n\n"
            "## B\n\nb body\n\n### B1\n\nb1 body"
        )
        chunks = _sections(md)
        assert [c.heading_path for c in chunks] == ["A", "A > A1", "B", "B > B1"]

    def test_h1_is_not_a_split_boundary(self) -> None:
        md = "# Title\n\nintro under the h1\n\n## Real section\n\nbody"
        chunks = _sections(md)
        # The H1 line and the text under it both land in the untitled preamble.
        assert chunks[0].heading_path == ""
        assert "Title" in chunks[0].text
        assert chunks[1].heading_path == "Real section"


class TestRuntMerging:
    def test_short_section_merges_into_previous(self) -> None:
        md = "## Long enough\n\n" + ("word " * 200) + "\n\n## See also\n\n- a link"
        chunks = chunk_markdown(md, CFG)
        assert len(chunks) == 1
        assert chunks[0].heading_path == "Long enough"
        assert "a link" in chunks[0].text

    def test_leading_runt_with_nothing_before_it_stands_alone(self) -> None:
        md = "## Tiny\n\njust a few words"
        chunks = chunk_markdown(md, CFG)
        assert len(chunks) == 1
        assert chunks[0].heading_path == "Tiny"


class TestHardSplit:
    def test_oversized_section_is_split(self) -> None:
        huge_cfg = ChunkingConfig(target_tokens=100, hard_split_tokens=150)
        paragraphs = "\n\n".join(f"paragraph number {i} with some extra words in it" for i in range(40))
        md = f"## Big\n\n{paragraphs}"
        chunks = chunk_markdown(md, huge_cfg)
        assert len(chunks) > 1
        assert all(c.heading_path == "Big" for c in chunks)
        # Nothing was lost: every paragraph's marker text survives somewhere.
        rejoined = " ".join(c.text for c in chunks)
        assert "paragraph number 0 " in rejoined
        assert "paragraph number 39 " in rejoined

    def test_single_giant_paragraph_is_token_cut(self) -> None:
        tiny_cfg = ChunkingConfig(target_tokens=100, hard_split_tokens=100)
        md = "## Dense\n\n" + ("word " * 200)  # one paragraph, no blank lines
        chunks = chunk_markdown(md, tiny_cfg)
        assert len(chunks) > 1
        assert all(c.heading_path == "Dense" for c in chunks)


class TestPrelude:
    def test_includes_title_path_and_heading(self) -> None:
        prelude = build_prelude("Anthropic", "10 raw/Anthropic.md", "History > Founding")
        assert prelude == "Anthropic\nHistory > Founding\n10 raw/Anthropic.md"

    def test_omits_empty_heading_path(self) -> None:
        prelude = build_prelude("Anthropic", "10 raw/Anthropic.md", "")
        assert prelude == "Anthropic\n10 raw/Anthropic.md"


class TestChunkId:
    def test_stable_across_calls(self) -> None:
        assert chunk_id("doc-1", "Heading", 0) == chunk_id("doc-1", "Heading", 0)

    def test_differs_on_ordinal(self) -> None:
        assert chunk_id("doc-1", "Heading", 0) != chunk_id("doc-1", "Heading", 1)

    def test_differs_on_doc(self) -> None:
        assert chunk_id("doc-1", "Heading", 0) != chunk_id("doc-2", "Heading", 0)


def _seed_docs(conn: sqlite3.Connection, *doc_ids: str) -> None:
    """`chunks.doc_id` has a foreign key into `docs` — real callers (ingest.py)
    always upsert the docs row first, so these tests give chunk.py the same.
    """
    for doc_id in doc_ids:
        conn.execute(
            "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
            "VALUES (?, ?, ?, 'rev-1', 'hash', 'open', 0, '{}')",
            (doc_id, doc_id, doc_id),
        )
    conn.commit()


class TestPersistence:
    def test_replace_chunks_writes_chunks_and_fts(self, db: sqlite3.Connection) -> None:
        _seed_docs(db, "10 raw/anthropic.md")
        md = "---\ntitle: Anthropic\n---\n\n## History\n\nfounded in 2021"
        n = replace_chunks(
            db,
            doc_id="10 raw/anthropic.md",
            path="10 raw/Anthropic.md",
            title="Anthropic",
            sensitivity="open",
            markdown=md,
            cfg=CFG,
        )
        db.commit()
        assert n == 1
        chunks = db.execute("SELECT * FROM chunks WHERE doc_id = ?", ("10 raw/anthropic.md",)).fetchall()
        assert len(chunks) == 1
        assert chunks[0]["heading_path"] == "History"
        assert chunks[0]["body"] == "founded in 2021"
        assert "founded" not in chunks[0]["prelude"]  # prelude is title/heading/path only

        fts = db.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH 'founded'"
        ).fetchall()
        assert [r["chunk_id"] for r in fts] == [chunks[0]["chunk_id"]]

    def test_frontmatter_is_not_chunked(self, db: sqlite3.Connection) -> None:
        _seed_docs(db, "d1")
        md = "---\ntitle: Anthropic\nsensitivity: open\n---\n\nbody text"
        replace_chunks(
            db,
            doc_id="d1",
            path="p.md",
            title="Anthropic",
            sensitivity="open",
            markdown=md,
            cfg=CFG,
        )
        db.commit()
        row = db.execute("SELECT body FROM chunks WHERE doc_id = 'd1'").fetchone()
        assert row["body"] == "body text"

    def test_replace_is_delete_then_insert(self, db: sqlite3.Connection) -> None:
        _seed_docs(db, "d1")
        replace_chunks(
            db, doc_id="d1", path="p.md", title="T", sensitivity="open",
            markdown="## One\n\nfirst version", cfg=CFG,
        )
        replace_chunks(
            db, doc_id="d1", path="p.md", title="T", sensitivity="open",
            markdown="## Two\n\nsecond version", cfg=CFG,
        )
        db.commit()
        chunks = db.execute("SELECT heading_path FROM chunks WHERE doc_id = 'd1'").fetchall()
        assert [c["heading_path"] for c in chunks] == ["Two"]
        fts = db.execute("SELECT body FROM chunks_fts WHERE doc_id = 'd1'").fetchall()
        assert [r["body"] for r in fts] == ["second version"]

    def test_delete_chunks_clears_both_tables(self, db: sqlite3.Connection) -> None:
        _seed_docs(db, "d1")
        replace_chunks(
            db, doc_id="d1", path="p.md", title="T", sensitivity="open",
            markdown="## One\n\nbody", cfg=CFG,
        )
        db.commit()
        delete_chunks(db, "d1")
        db.commit()
        assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 0

    def test_delete_chunks_does_not_touch_other_docs(self, db: sqlite3.Connection) -> None:
        _seed_docs(db, "d1", "d2")
        replace_chunks(
            db, doc_id="d1", path="p1.md", title="T1", sensitivity="open",
            markdown="## One\n\nbody one", cfg=CFG,
        )
        replace_chunks(
            db, doc_id="d2", path="p2.md", title="T2", sensitivity="open",
            markdown="## Two\n\nbody two", cfg=CFG,
        )
        db.commit()
        delete_chunks(db, "d1")
        db.commit()
        remaining = db.execute("SELECT doc_id FROM chunks").fetchall()
        assert [r["doc_id"] for r in remaining] == ["d2"]
        remaining_fts = db.execute("SELECT doc_id FROM chunks_fts").fetchall()
        assert [r["doc_id"] for r in remaining_fts] == ["d2"]
