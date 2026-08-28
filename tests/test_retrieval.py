from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig
from vault_ask.db import connect
from vault_ask.retrieval import (
    SearchHit,
    build_fts_query,
    expand_graph,
    fuse_rrf,
    search_fts,
    search_vector,
)

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


def _vec(*values: float) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _seed_vector(conn: sqlite3.Connection, chunk_id: str, embedding: tuple[float, ...]) -> None:
    conn.execute(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)", (chunk_id, _vec(*embedding))
    )
    conn.commit()


def _seed_edge(
    conn: sqlite3.Connection, src: str, dst: str, kind: str = "topic", resolved: int = 1
) -> None:
    conn.execute(
        "INSERT INTO edges (src, dst, kind, resolved) VALUES (?, ?, ?, ?)",
        (src, dst, kind, resolved),
    )
    conn.commit()


def _hit_for(conn: sqlite3.Connection, doc_id: str, *, score: float = 1.0) -> SearchHit:
    row = conn.execute(
        "SELECT c.chunk_id, c.heading_path, c.prelude, c.body, c.sensitivity, d.path, d.title "
        "FROM chunks c JOIN docs d ON d.doc_id = c.doc_id WHERE c.doc_id = ? ORDER BY c.ordinal LIMIT 1",
        (doc_id,),
    ).fetchone()
    return SearchHit(
        chunk_id=row["chunk_id"],
        doc_id=doc_id,
        path=row["path"],
        title=row["title"],
        heading_path=row["heading_path"] or "",
        prelude=row["prelude"] or "",
        body=row["body"] or "",
        sensitivity=row["sensitivity"],
        score=score,
    )


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


class TestBuildFtsQuery:
    def test_tokenises_and_ors(self) -> None:
        assert build_fts_query("Entra ID") == '"Entra" OR "ID"'

    def test_strips_operators_and_punctuation(self) -> None:
        # A raw "-word" is FTS5 syntax for NOT; quoting neutralises it, and a
        # bare hyphen disappears rather than raising a MATCH syntax error.
        assert build_fts_query("pre-auth AND flow") == '"pre" OR "auth" OR "AND" OR "flow"'

    def test_empty_input(self) -> None:
        assert build_fts_query("   ") == ""


class TestSearchFts:
    def test_finds_matching_chunk(self, db: sqlite3.Connection) -> None:
        _seed(db, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        hits = search_fts(db, "founded", top_k=10)
        assert [h.path for h in hits] == ["10 raw/Anthropic.md"]
        assert hits[0].title == "Anthropic"
        assert hits[0].heading_path == "History"

    def test_no_match_is_empty(self, db: sqlite3.Connection) -> None:
        _seed(db, "d1", "p.md", "T", "open", "## H\n\nunrelated text")
        assert search_fts(db, "quantum computing", top_k=10) == []

    def test_empty_query_is_empty(self, db: sqlite3.Connection) -> None:
        _seed(db, "d1", "p.md", "T", "open", "## H\n\nfounded in 2021")
        assert search_fts(db, "   ", top_k=10) == []

    def test_sensitivity_filter_excludes_personal(self, db: sqlite3.Connection) -> None:
        _seed(db, "d1", "10 raw/Public.md", "Public", "open", "## H\n\nfounded in 2021")
        _seed(db, "d2", "30 journal/Private.md", "Private", "personal", "## H\n\nfounded in 2021")
        hits = search_fts(db, "founded", top_k=10, sensitivity="open")
        assert [h.doc_id for h in hits] == ["d1"]

    def test_no_filter_returns_both(self, db: sqlite3.Connection) -> None:
        _seed(db, "d1", "10 raw/Public.md", "Public", "open", "## H\n\nfounded in 2021")
        _seed(db, "d2", "30 journal/Private.md", "Private", "personal", "## H\n\nfounded in 2021")
        hits = search_fts(db, "founded", top_k=10, sensitivity=None)
        assert {h.doc_id for h in hits} == {"d1", "d2"}

    def test_top_k_limits_results(self, db: sqlite3.Connection) -> None:
        for i in range(5):
            _seed(db, f"d{i}", f"p{i}.md", f"T{i}", "open", f"## H\n\nfounded in {2000 + i}")
        hits = search_fts(db, "founded", top_k=2)
        assert len(hits) == 2


class TestSearchVector:
    def test_orders_by_distance(self, vec_db: sqlite3.Connection) -> None:
        _seed(vec_db, "d1", "near.md", "Near", "open", "## H\n\nnear text")
        _seed(vec_db, "d2", "far.md", "Far", "open", "## H\n\nfar text")
        chunk_near = vec_db.execute("SELECT chunk_id FROM chunks WHERE doc_id='d1'").fetchone()[0]
        chunk_far = vec_db.execute("SELECT chunk_id FROM chunks WHERE doc_id='d2'").fetchone()[0]
        _seed_vector(vec_db, chunk_near, (1.0, 0.0, 0.0, 0.0))
        _seed_vector(vec_db, chunk_far, (0.0, 0.0, 0.0, 1.0))

        hits = search_vector(vec_db, [1.0, 0.0, 0.0, 0.0], top_k=10)
        assert [h.doc_id for h in hits] == ["d1", "d2"]
        assert hits[0].score < hits[1].score  # distance: smaller is closer

    def test_sensitivity_filter(self, vec_db: sqlite3.Connection) -> None:
        _seed(vec_db, "d1", "public.md", "Public", "open", "## H\n\ntext")
        _seed(vec_db, "d2", "private.md", "Private", "personal", "## H\n\ntext")
        c1 = vec_db.execute("SELECT chunk_id FROM chunks WHERE doc_id='d1'").fetchone()[0]
        c2 = vec_db.execute("SELECT chunk_id FROM chunks WHERE doc_id='d2'").fetchone()[0]
        _seed_vector(vec_db, c1, (1.0, 0.0, 0.0, 0.0))
        _seed_vector(vec_db, c2, (1.0, 0.0, 0.0, 0.0))

        hits = search_vector(vec_db, [1.0, 0.0, 0.0, 0.0], top_k=10, sensitivity="open")
        assert [h.doc_id for h in hits] == ["d1"]

    def test_top_k_limits_results(self, vec_db: sqlite3.Connection) -> None:
        for i in range(5):
            _seed(vec_db, f"d{i}", f"p{i}.md", f"T{i}", "open", f"## H\n\ntext {i}")
            cid = vec_db.execute(f"SELECT chunk_id FROM chunks WHERE doc_id='d{i}'").fetchone()[0]
            _seed_vector(vec_db, cid, (float(i), 0.0, 0.0, 0.0))
        hits = search_vector(vec_db, [0.0, 0.0, 0.0, 0.0], top_k=2)
        assert len(hits) == 2


class TestFuseRrf:
    def _hit(self, chunk_id: str) -> SearchHit:
        return SearchHit(
            chunk_id=chunk_id,
            doc_id=chunk_id,
            path=f"{chunk_id}.md",
            title=chunk_id,
            heading_path="",
            prelude="",
            body="",
            sensitivity="open",
            score=0.0,
        )

    def test_hit_in_both_lists_outranks_a_hit_in_only_one(self) -> None:
        a, b, c = self._hit("a"), self._hit("b"), self._hit("c")
        # "a" is #2 in vector and #2 in fts; "b" is #1 in vector only; "c" is
        # #1 in fts only. Appearing in both should beat leading either list once.
        vector_hits = [b, a]
        fts_hits = [c, a]
        fused = fuse_rrf(vector_hits, fts_hits, top_k=10)
        assert fused[0].chunk_id == "a"

    def test_union_of_both_lists(self) -> None:
        a, b = self._hit("a"), self._hit("b")
        fused = fuse_rrf([a], [b], top_k=10)
        assert {h.chunk_id for h in fused} == {"a", "b"}

    def test_top_k_truncates(self) -> None:
        hits = [self._hit(str(i)) for i in range(10)]
        fused = fuse_rrf(hits, [], top_k=3)
        assert len(fused) == 3

    def test_empty_inputs(self) -> None:
        assert fuse_rrf([], [], top_k=10) == []

    def test_fused_score_replaces_source_score(self) -> None:
        from dataclasses import replace

        a = replace(self._hit("a"), score=999.0)  # whatever bm25/distance happened to be
        fused = fuse_rrf([a], [], top_k=10)
        assert fused[0].score != 999.0


class TestExpandGraph:
    def test_pulls_in_the_topic_note(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/Anthropic.md", "Anthropic", "open", "## H\n\nbody")
        _seed(db, "topic", "99 topics/Anthropic.md", "Anthropic topic", "open", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic")

        hit = _hit_for(db, "hit", score=1.0)
        expanded = expand_graph(db, [hit], max_siblings=5, discount=0.7)
        assert [h.doc_id for h in expanded] == ["topic"]
        assert expanded[0].score == pytest.approx(0.7)

    def test_pulls_in_siblings_sharing_the_topic(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(db, "sibling", "10 raw/B.md", "B", "open", "## H\n\nbody")
        _seed(db, "topic", "99 topics/T.md", "T", "open", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic")
        _seed_edge(db, src="topic", dst="sibling", kind="topic")

        expanded = expand_graph(db, [_hit_for(db, "hit")], max_siblings=5, discount=0.7)
        assert {h.doc_id for h in expanded} == {"topic", "sibling"}

    def test_max_siblings_caps_per_topic(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(db, "topic", "99 topics/T.md", "T", "open", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic")
        for i in range(5):
            doc_id = f"sib{i}"
            _seed(db, doc_id, f"10 raw/S{i}.md", f"S{i}", "open", "## H\n\nbody")
            _seed_edge(db, src="topic", dst=doc_id, kind="topic")

        expanded = expand_graph(db, [_hit_for(db, "hit")], max_siblings=2, discount=0.7)
        sibling_docs = [h.doc_id for h in expanded if h.doc_id != "topic"]
        assert len(sibling_docs) == 2

    def test_docs_already_in_hits_are_not_duplicated(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(db, "sibling", "10 raw/B.md", "B", "open", "## H\n\nbody")
        _seed(db, "topic", "99 topics/T.md", "T", "open", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic")
        _seed_edge(db, src="topic", dst="sibling", kind="topic")

        # `sibling` is already a hit in its own right — expansion should not
        # produce a second, discounted copy of it.
        hits = [_hit_for(db, "hit"), _hit_for(db, "sibling")]
        expanded = expand_graph(db, hits, max_siblings=5, discount=0.7)
        assert "sibling" not in {h.doc_id for h in expanded}

    def test_one_hop_only(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(db, "topic", "99 topics/T.md", "T", "open", "## H\n\nabout")
        _seed(db, "sibling", "10 raw/B.md", "B", "open", "## H\n\nbody")
        _seed(db, "far", "99 topics/Far.md", "Far", "open", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic")
        _seed_edge(db, src="topic", dst="sibling", kind="topic")
        # `far` is a topic note for `sibling` — two hops from `hit`.
        _seed_edge(db, src="far", dst="sibling", kind="topic")

        expanded = expand_graph(db, [_hit_for(db, "hit")], max_siblings=5, discount=0.7)
        assert "far" not in {h.doc_id for h in expanded}

    def test_sensitivity_filter_excludes_personal_expansion(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(db, "topic", "40 people/Private.md", "Private", "personal", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic")

        expanded = expand_graph(db, [_hit_for(db, "hit")], max_siblings=5, discount=0.7, sensitivity="open")
        assert expanded == []

    def test_unresolved_edges_are_ignored(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(db, "topic", "99 topics/T.md", "T", "open", "## H\n\nabout")
        _seed_edge(db, src="topic", dst="hit", kind="topic", resolved=0)

        expanded = expand_graph(db, [_hit_for(db, "hit")], max_siblings=5, discount=0.7)
        assert expanded == []

    def test_no_hits_is_empty(self, db: sqlite3.Connection) -> None:
        assert expand_graph(db, [], max_siblings=5, discount=0.7) == []

    def test_no_edges_is_empty(self, db: sqlite3.Connection) -> None:
        _seed(db, "hit", "10 raw/A.md", "A", "open", "## H\n\nbody")
        assert expand_graph(db, [_hit_for(db, "hit")], max_siblings=5, discount=0.7) == []


class TestProvenance:
    """Graph expansion could not be measured because nothing downstream could
    tell an expanded chunk from a direct hit — `fuse_rrf` discarded origin.
    """

    def test_fts_hits_are_tagged(self, db: sqlite3.Connection) -> None:
        _seed(db, "a.md", "a.md", "A", "open", "## H\n\nwhisky")
        hits = search_fts(db, "whisky", top_k=5)
        assert [h.source for h in hits] == ["fts"]

    def test_vector_hits_are_tagged(self, vec_db: sqlite3.Connection) -> None:
        _seed(vec_db, "a.md", "a.md", "A", "open", "## H\n\nbody")
        cid = vec_db.execute("SELECT chunk_id FROM chunks LIMIT 1").fetchone()["chunk_id"]
        _seed_vector(vec_db, cid, (1.0, 0.0, 0.0, 0.0))
        hits = search_vector(vec_db, [1.0, 0.0, 0.0, 0.0], top_k=5)
        assert [h.source for h in hits] == ["vector"]

    def test_fusion_records_both_arms(self, db: sqlite3.Connection) -> None:
        """Being in both lists is what RRF most rewards, and was exactly the
        thing fusion used to make invisible."""
        _seed(db, "a.md", "a.md", "A", "open", "## H\n\nbody")
        hit = _hit_for(db, "a.md")
        fused = fuse_rrf([replace(hit, source="vector")], [replace(hit, source="fts")], top_k=5)
        assert [h.source for h in fused] == ["vector+fts"]

    def test_fusion_keeps_a_single_arm_unchanged(self, db: sqlite3.Connection) -> None:
        _seed(db, "a.md", "a.md", "A", "open", "## H\n\nbody")
        hit = _hit_for(db, "a.md")
        fused = fuse_rrf([replace(hit, source="vector")], [], top_k=5)
        assert [h.source for h in fused] == ["vector"]

    def test_expanded_hits_are_tagged_graph(self, db: sqlite3.Connection) -> None:
        _seed(db, "99 topics/t.md", "99 topics/t.md", "T", "open", "## H\n\nhub")
        _seed(db, "hit.md", "hit.md", "Hit", "open", "## H\n\nbody")
        _seed_edge(db, "99 topics/t.md", "hit.md")
        expanded = expand_graph(db, [_hit_for(db, "hit.md")], max_siblings=5, discount=0.7)
        assert expanded and all(h.source == "graph" for h in expanded)
