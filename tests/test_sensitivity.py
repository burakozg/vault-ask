"""The sensitivity gate — README "Corpus and sensitivity".

This is the test file the README has named as load-bearing since the project
started ("the test that stops the failure that actually costs something") and
which did not exist until an audit went looking for it.

What makes it different from the sensitivity assertions already scattered
through test_retrieval.py / test_ask.py / test_mcp_adapter.py: those check the
*weak* property, that personal text is absent from the output. That property
holds under a filter-*after* implementation too, so they cannot distinguish a
correct gate from a broken one. The tests here check the properties that only
hold when the filter is applied before the result set is truncated:

* a personal chunk must not **consume a slot** in a top-k;
* the permitted result count must be **preserved** under the filter;
* metadata (path, title, existence) must not escape either.

Every test that matters here therefore seeds *more personal candidates than
the k being asked for*, so the k genuinely binds. The pre-existing vector test
(`test_retrieval.py::test_sensitivity_filter`) uses top_k=10 against a 2-row
corpus, where k never binds and the bug it was meant to catch cannot appear.
"""

from __future__ import annotations

import sqlite3
import struct
from collections.abc import Iterator
from pathlib import Path

import pytest

from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig
from vault_ask.db import connect
from vault_ask.frontmatter import classify_sensitivity
from vault_ask.retrieval import (
    assemble_context,
    expand_graph,
    search_fts,
    search_vector,
)

CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)

#: Embeddings are 4-dimensional here purely so the fixtures stay readable.
DIM = 4


@pytest.fixture
def db(tmp_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(tmp_path / "index.sqlite", embedding_dim=DIM)
    yield conn
    conn.close()


def _seed(
    conn: sqlite3.Connection,
    doc_id: str,
    *,
    sensitivity: str,
    body: str,
    embedding: tuple[float, ...] | None = None,
) -> str:
    """One doc, its chunks, and optionally its vector. Returns the chunk_id."""
    path = doc_id
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, 'rev-1', 'hash', ?, 0, '{}')",
        (doc_id, path, doc_id.removesuffix(".md"), sensitivity),
    )
    replace_chunks(
        conn,
        doc_id=doc_id,
        path=path,
        title=doc_id.removesuffix(".md"),
        sensitivity=sensitivity,
        markdown=body,
        cfg=CFG,
    )
    chunk_id = conn.execute(
        "SELECT chunk_id FROM chunks WHERE doc_id = ? ORDER BY ordinal LIMIT 1", (doc_id,)
    ).fetchone()["chunk_id"]
    if embedding is not None:
        conn.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, struct.pack(f"{len(embedding)}f", *embedding)),
        )
    conn.commit()
    return str(chunk_id)


class TestVectorSearchSlotConsumption:
    """The bug this file exists for.

    `chunks_vec` is `vec0(chunk_id, embedding)` — no sensitivity column — so
    vec0 cannot see the `c.sensitivity` predicate: `k` binds first and the
    filter is applied to the rows vec0 already picked. Without compensation,
    personal chunks eat the whole k.
    """

    def test_personal_chunks_do_not_consume_the_top_k(self, db: sqlite3.Connection) -> None:
        # Ten personal chunks strictly nearer the query than any open one.
        for i in range(10):
            _seed(
                db,
                f"Tastings/p{i}.md",
                sensitivity="personal",
                body="## H\n\npersonal body",
                embedding=(1.0, 0.01 * i, 0.0, 0.0),
            )
        for i in range(3):
            _seed(
                db,
                f"10 raw/o{i}.md",
                sensitivity="open",
                body="## H\n\nopen body",
                embedding=(0.0, 0.0, 1.0, 0.01 * i),
            )

        # k=5 binds hard: the 5 nearest chunks are all personal. A naive
        # `k = top_k` returns zero rows here.
        hits = search_vector(db, [1.0, 0.0, 0.0, 0.0], top_k=5, sensitivity="open")

        assert [h.sensitivity for h in hits] == ["open"] * 3
        assert len(hits) == 3, "open chunks exist and must be reachable despite nearer personal ones"

    def test_count_preserved_when_enough_open_chunks_exist(self, db: sqlite3.Connection) -> None:
        """The stronger property: a full top_k, not merely a non-empty list."""
        for i in range(10):
            _seed(
                db,
                f"Tastings/p{i}.md",
                sensitivity="personal",
                body="## H\n\npersonal body",
                embedding=(1.0, 0.01 * i, 0.0, 0.0),
            )
        for i in range(8):
            _seed(
                db,
                f"10 raw/o{i}.md",
                sensitivity="open",
                body="## H\n\nopen body",
                embedding=(0.0, 0.0, 1.0, 0.01 * i),
            )

        hits = search_vector(db, [1.0, 0.0, 0.0, 0.0], top_k=5, sensitivity="open")
        assert len(hits) == 5
        assert all(h.sensitivity == "open" for h in hits)

    def test_never_exceeds_top_k_after_widening(self, db: sqlite3.Connection) -> None:
        """Widening must not leak extra rows past the caller's contract."""
        for i in range(6):
            _seed(
                db, f"Tastings/p{i}.md", sensitivity="personal",
                body="## H\n\nbody", embedding=(1.0, 0.01 * i, 0.0, 0.0),
            )
        for i in range(20):
            _seed(
                db, f"10 raw/o{i}.md", sensitivity="open",
                body="## H\n\nbody", embedding=(0.0, 0.0, 1.0, 0.01 * i),
            )
        hits = search_vector(db, [1.0, 0.0, 0.0, 0.0], top_k=3, sensitivity="open")
        assert len(hits) == 3

    def test_unfiltered_search_is_unaffected(self, db: sqlite3.Connection) -> None:
        """`sensitivity=None` (allow_web false) must not pay the widening cost
        or change behaviour — it is the common path."""
        for i in range(10):
            _seed(
                db, f"Tastings/p{i}.md", sensitivity="personal",
                body="## H\n\nbody", embedding=(1.0, 0.01 * i, 0.0, 0.0),
            )
        hits = search_vector(db, [1.0, 0.0, 0.0, 0.0], top_k=4, sensitivity=None)
        assert len(hits) == 4


class TestFtsSlotConsumption:
    """FTS filters in the WHERE clause before LIMIT, so it was always correct.
    Pinned so a future "optimisation" that post-filters gets caught.
    """

    def test_personal_chunks_do_not_consume_the_top_k(self, db: sqlite3.Connection) -> None:
        for i in range(10):
            _seed(db, f"Tastings/p{i}.md", sensitivity="personal", body="## H\n\nwhisky whisky whisky")
        for i in range(3):
            _seed(db, f"10 raw/o{i}.md", sensitivity="open", body="## H\n\nwhisky")

        hits = search_fts(db, "whisky", top_k=5, sensitivity="open")
        assert len(hits) == 3
        assert all(h.sensitivity == "open" for h in hits)


class TestGraphExpansionSlotConsumption:
    """Same class of bug as the vector path: `LIMIT :max_siblings` applied
    before the sensitivity predicate would make a topic whose first N members
    are personal expand to nothing, while permitted siblings sit behind them.
    """

    def test_personal_siblings_do_not_consume_max_siblings(self, db: sqlite3.Connection) -> None:
        _seed(db, "99 topics/whisky.md", sensitivity="open", body="## H\n\ntopic hub")
        _seed(db, "10 raw/hit.md", sensitivity="open", body="## H\n\nthe hit")
        # Names chosen so the personal siblings sort *ahead* of the open one
        # under the query's ORDER BY — otherwise a LIMIT applied before the
        # sensitivity filter would happen to pick the open sibling anyway and
        # the test would pass against the broken implementation.
        for i in range(5):
            _seed(db, f"10 raw/aa{i}.md", sensitivity="personal", body="## H\n\npersonal")
        _seed(db, "10 raw/zz-open-sibling.md", sensitivity="open", body="## H\n\nopen sibling")

        for member in (
            "10 raw/hit.md",
            *(f"10 raw/aa{i}.md" for i in range(5)),
            "10 raw/zz-open-sibling.md",
        ):
            db.execute(
                "INSERT INTO edges (src, dst, kind, resolved) VALUES ('99 topics/whisky.md', ?, 'topic', 1)",
                (member,),
            )
        db.commit()

        from vault_ask.retrieval import SearchHit

        hit = SearchHit(
            chunk_id="x", doc_id="10 raw/hit.md", path="10 raw/hit.md", title="hit",
            heading_path="", prelude="", body="the hit", sensitivity="open", score=1.0,
        )
        expanded = expand_graph(db, [hit], max_siblings=2, discount=0.7, sensitivity="open")

        assert all(h.sensitivity == "open" for h in expanded)
        paths = {h.path for h in expanded}
        assert "10 raw/zz-open-sibling.md" in paths, (
            "an open sibling must be reachable even when personal ones sort ahead of it"
        )

    def test_personal_topic_hub_is_not_traversed(self, db: sqlite3.Connection) -> None:
        """A hub the caller may not see must not leak by inference.

        Withholding the hub's text but still pulling its open members in would
        encode "these notes co-belong to a personal topic" into the citations.
        """
        _seed(db, "Tastings/private-hub.md", sensitivity="personal", body="## H\n\nsecret hub")
        _seed(db, "10 raw/hit.md", sensitivity="open", body="## H\n\nthe hit")
        _seed(db, "10 raw/member.md", sensitivity="open", body="## H\n\nco-member")
        for member in ("10 raw/hit.md", "10 raw/member.md"):
            db.execute(
                "INSERT INTO edges (src, dst, kind, resolved) "
                "VALUES ('Tastings/private-hub.md', ?, 'topic', 1)",
                (member,),
            )
        db.commit()

        from vault_ask.retrieval import SearchHit

        hit = SearchHit(
            chunk_id="x", doc_id="10 raw/hit.md", path="10 raw/hit.md", title="hit",
            heading_path="", prelude="", body="the hit", sensitivity="open", score=1.0,
        )
        expanded = expand_graph(db, [hit], max_siblings=5, discount=0.7, sensitivity="open")
        assert expanded == [], "a personal hub must not be traversed, not merely withheld"


class TestAssembleContextGuard:
    """Last line of defence — the invariant otherwise rests on four separate
    call sites each being passed the right argument.
    """

    def test_personal_hit_raises(self) -> None:
        from vault_ask.retrieval import SearchHit

        leaked = SearchHit(
            chunk_id="x", doc_id="Tastings/p.md", path="Tastings/p.md", title="p",
            heading_path="", prelude="", body="secret", sensitivity="personal", score=1.0,
        )
        with pytest.raises(AssertionError, match="sensitivity gate breached"):
            assemble_context([leaked], sensitivity="open")

    def test_open_hits_pass(self) -> None:
        from vault_ask.retrieval import SearchHit

        ok = SearchHit(
            chunk_id="x", doc_id="a.md", path="a.md", title="a",
            heading_path="", prelude="", body="fine", sensitivity="open", score=1.0,
        )
        assert "fine" in assemble_context([ok], sensitivity="open")

    def test_unfiltered_assembly_allows_personal(self) -> None:
        """allow_web=false is the local-only path — personal content is the
        whole point of it, and must not trip the guard."""
        from vault_ask.retrieval import SearchHit

        personal = SearchHit(
            chunk_id="x", doc_id="Tastings/p.md", path="Tastings/p.md", title="p",
            heading_path="", prelude="", body="secret", sensitivity="personal", score=1.0,
        )
        assert "secret" in assemble_context([personal], sensitivity=None)


class TestClassificationMatchesTheRealVault:
    """The classification rule itself, pinned against the folders that actually
    exist. Every one of 2,254 real notes classified `open` because the shipped
    patterns named folders (`30 journal/`, `40 people/`, `50 tastings/`) that
    this vault does not have.
    """

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("Tastings/Laphroaig 10.md", "personal"),
            ("30 projects/vault-ask.md", "personal"),
            ("10 raw/Anthropic.md", "open"),
            ("12 daily-digest/2026/08/x.md", "open"),
            ("99 topics/ai.md", "open"),
            # Case-sensitive: fnmatchcase, against the real path. The
            # lowercased CouchDB doc_id would NOT match, which is the trap.
            ("tastings/laphroaig 10.md", "open"),
        ],
    )
    def test_shipped_defaults_against_real_paths(self, path: str, expected: str) -> None:
        from vault_ask.config import SensitivityConfig

        assert classify_sensitivity(path, {}, SensitivityConfig()) == expected

    def test_frontmatter_override_beats_path(self) -> None:
        from vault_ask.config import SensitivityConfig

        cfg = SensitivityConfig()
        assert classify_sensitivity("Tastings/x.md", {"sensitivity": "open"}, cfg) == "open"
        assert classify_sensitivity("10 raw/x.md", {"sensitivity": "personal"}, cfg) == "personal"
