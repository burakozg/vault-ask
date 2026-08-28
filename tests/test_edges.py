from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from vault_ask.db import connect
from vault_ask.edges import (
    delete_edges,
    extract_tags,
    extract_topic_links,
    extract_wikilinks,
    replace_edges,
    resolve,
    resolve_pending,
)


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite")
    yield conn
    conn.close()


def _seed_doc(conn: sqlite3.Connection, path: str, title: str) -> str:
    """doc_id is always the lowercased path — the real convention
    (vault_ask.vault.Entry.doc_id) that vault_ask.edges.resolve depends on.
    Returns the doc_id so callers don't have to repeat the lowering.
    """
    doc_id = path.lower()
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, 'rev-1', 'hash', 'open', 0, '{}')",
        (doc_id, path, title),
    )
    conn.commit()
    return doc_id


class TestExtractWikilinks:
    def test_plain_link(self) -> None:
        assert extract_wikilinks("see [[10 raw/Anthropic.md]] for more") == ["10 raw/Anthropic.md"]

    def test_aliased_link_drops_the_alias(self) -> None:
        assert extract_wikilinks("[[10 raw/Anthropic.md|Anthropic]]") == ["10 raw/Anthropic.md"]

    def test_multiple_links(self) -> None:
        text = "[[A]] and [[B|beta]] and [[C]]"
        assert extract_wikilinks(text) == ["A", "B", "C"]

    def test_no_links(self) -> None:
        assert extract_wikilinks("just plain text") == []

    def test_empty_link_target_is_skipped(self) -> None:
        assert extract_wikilinks("[[]] and [[real]]") == ["real"]


class TestExtractTopicLinks:
    def test_links_inside_marker_region(self) -> None:
        md = (
            "intro\n\n<!-- begin:clippings -->\n"
            "- [[10 raw/A.md|A]]\n- [[10 raw/B.md|B]]\n"
            "<!-- end:clippings -->\n\noutro [[10 raw/C.md]]"
        )
        assert extract_topic_links(md) == ["10 raw/A.md", "10 raw/B.md"]

    def test_no_marker_region_is_empty(self) -> None:
        assert extract_topic_links("[[A]] [[B]]") == []

    def test_multiple_regions(self) -> None:
        md = (
            "<!-- begin:clippings -->[[A]]<!-- end:clippings -->\n"
            "text\n"
            "<!-- begin:clippings -->[[B]]<!-- end:clippings -->"
        )
        assert extract_topic_links(md) == ["A", "B"]


class TestExtractTags:
    def test_list_of_tags(self) -> None:
        assert extract_tags({"tags": ["ai", "vendor"]}) == ["ai", "vendor"]

    def test_single_string_tag(self) -> None:
        assert extract_tags({"tags": "ai"}) == ["ai"]

    def test_no_tags_key(self) -> None:
        assert extract_tags({}) == []

    def test_non_string_tags_coerced(self) -> None:
        assert extract_tags({"tags": [1, "two"]}) == ["1", "two"]


class TestResolve:
    def test_path_qualified_target_resolves(self) -> None:
        dst, ok = resolve(
            "10 raw/Anthropic.md", doc_ids={"10 raw/anthropic.md"}, by_title={}
        )
        assert (dst, ok) == ("10 raw/anthropic.md", True)

    def test_path_qualified_without_extension_resolves(self) -> None:
        dst, ok = resolve("10 raw/Anthropic", doc_ids={"10 raw/anthropic.md"}, by_title={})
        assert (dst, ok) == ("10 raw/anthropic.md", True)

    def test_bare_title_resolves_when_unambiguous(self) -> None:
        dst, ok = resolve(
            "Anthropic",
            doc_ids=set(),
            by_title={"anthropic": ["10 raw/anthropic.md"]},
        )
        assert (dst, ok) == ("10 raw/anthropic.md", True)

    def test_ambiguous_title_stays_unresolved(self) -> None:
        dst, ok = resolve(
            "Overview",
            doc_ids=set(),
            by_title={"overview": ["a/overview.md", "b/overview.md"]},
        )
        assert ok is False
        assert dst == "Overview"

    def test_no_match_stays_unresolved(self) -> None:
        dst, ok = resolve("Nonexistent", doc_ids=set(), by_title={})
        assert (dst, ok) == ("Nonexistent", False)


class TestReplaceEdges:
    def test_wikilink_edge_written(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "10 raw/A.md", "A")
        _seed_doc(db, "10 raw/B.md", "B")
        replace_edges(db, doc_id=d1, markdown="see [[10 raw/B.md|B]]", frontmatter={})
        db.commit()
        rows = db.execute("SELECT src, dst, kind, resolved FROM edges").fetchall()
        assert [dict(r) for r in rows] == [
            {"src": d1, "dst": "10 raw/b.md", "kind": "wikilink", "resolved": 1}
        ]

    def test_unresolved_wikilink_kept(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "10 raw/A.md", "A")
        replace_edges(db, doc_id=d1, markdown="see [[Nowhere]]", frontmatter={})
        db.commit()
        row = db.execute("SELECT dst, resolved FROM edges WHERE src = ?", (d1,)).fetchone()
        assert row["dst"] == "Nowhere"
        assert row["resolved"] == 0

    def test_topic_region_produces_both_topic_and_wikilink_edges(
        self, db: sqlite3.Connection
    ) -> None:
        topic = _seed_doc(db, "99 topics/Anthropic.md", "Anthropic")
        clip = _seed_doc(db, "10 raw/Clip.md", "Clip")
        md = "<!-- begin:clippings -->\n- [[10 raw/Clip.md|Clip]]\n<!-- end:clippings -->"
        replace_edges(db, doc_id=topic, markdown=md, frontmatter={})
        db.commit()
        kinds = {
            row["kind"]
            for row in db.execute(
                "SELECT kind FROM edges WHERE src = ? AND dst = ?", (topic, clip)
            ).fetchall()
        }
        assert kinds == {"topic", "wikilink"}

    def test_tag_edges_never_resolved(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "10 raw/A.md", "A")
        replace_edges(db, doc_id=d1, markdown="body", frontmatter={"tags": ["ai", "vendor"]})
        db.commit()
        rows = db.execute(
            "SELECT dst, resolved FROM edges WHERE src = ? AND kind='tag'", (d1,)
        ).fetchall()
        assert {r["dst"] for r in rows} == {"tag:ai", "tag:vendor"}
        assert all(r["resolved"] == 0 for r in rows)

    def test_replace_is_delete_then_insert(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "p.md", "T")
        replace_edges(db, doc_id=d1, markdown="[[One]]", frontmatter={})
        replace_edges(db, doc_id=d1, markdown="[[Two]]", frontmatter={})
        db.commit()
        rows = db.execute("SELECT dst FROM edges WHERE src = ?", (d1,)).fetchall()
        assert [r["dst"] for r in rows] == ["Two"]


class TestDeleteEdges:
    def test_deletes_outbound_only(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "p1.md", "T1")
        _seed_doc(db, "p2.md", "T2")
        replace_edges(db, doc_id=d1, markdown="[[p2.md]]", frontmatter={})
        db.commit()
        delete_edges(db, d1)
        db.commit()
        assert db.execute("SELECT COUNT(*) FROM edges WHERE src = ?", (d1,)).fetchone()[0] == 0


class TestResolvePending:
    def test_resolves_once_target_exists(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "10 raw/A.md", "A")
        replace_edges(db, doc_id=d1, markdown="[[10 raw/B.md]]", frontmatter={})
        db.commit()
        row = db.execute("SELECT resolved FROM edges WHERE src = ?", (d1,)).fetchone()
        assert row["resolved"] == 0

        # B shows up in a later batch of the same run.
        _seed_doc(db, "10 raw/B.md", "B")
        n = resolve_pending(db)
        db.commit()
        assert n == 1
        row = db.execute("SELECT dst, resolved FROM edges WHERE src = ?", (d1,)).fetchone()
        assert (row["dst"], row["resolved"]) == ("10 raw/b.md", 1)

    def test_no_pending_edges_is_a_noop(self, db: sqlite3.Connection) -> None:
        assert resolve_pending(db) == 0

    def test_tag_edges_are_never_touched(self, db: sqlite3.Connection) -> None:
        d1 = _seed_doc(db, "p.md", "T")
        replace_edges(db, doc_id=d1, markdown="body", frontmatter={"tags": ["ai"]})
        db.commit()
        assert resolve_pending(db) == 0
        row = db.execute("SELECT resolved FROM edges WHERE kind='tag'").fetchone()
        assert row["resolved"] == 0
