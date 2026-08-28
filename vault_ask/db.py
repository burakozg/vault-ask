"""The SQLite index. One file — see README "Index": a join beats an operational
dependency for a graph this small.

Schema is declared whole, matching the design in README, even though `edges`
is not populated until a later build step (graph expansion). Declaring the
shape now means a later step adds a writer, not a migration.

`chunks_vec` is the one table whose shape depends on runtime config
(`models.embedding_dim` — vec0 bakes the vector width into the column type),
so it is created here parametrically rather than as a fixed string, and its
lifecycle past creation is `vault_ask.embed.ensure_embedding_space`'s job, not
this module's.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

SCHEMA = """
CREATE TABLE IF NOT EXISTS docs (
  doc_id TEXT PRIMARY KEY,       -- vault path, lowercased (LiveSync's own key)
  path TEXT NOT NULL,            -- real case
  title TEXT,
  rev TEXT NOT NULL,
  content_hash TEXT NOT NULL,
  sensitivity TEXT NOT NULL CHECK (sensitivity IN ('open', 'personal')),
  mtime INTEGER,
  frontmatter TEXT               -- JSON
);

CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY,
  doc_id TEXT NOT NULL REFERENCES docs(doc_id) ON DELETE CASCADE,
  ordinal INTEGER NOT NULL,
  heading_path TEXT,
  prelude TEXT,
  body TEXT,
  -- Denormalised from docs, so a retrieval filter never needs a join to
  -- decide whether a chunk is eligible for a web-augmented answer.
  sensitivity TEXT NOT NULL CHECK (sensitivity IN ('open', 'personal'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- doc_id is UNINDEXED (not part of the full-text index, just carried along) so
-- a whole document's rows can be cleared by equality without a join back to
-- `chunks` — see vault_ask.chunk.delete_chunks. FTS5 has no foreign keys, so
-- this table is kept in sync procedurally, on the same delete-then-insert
-- rhythm as `chunks` itself.
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id UNINDEXED, doc_id UNINDEXED, prelude, body, tokenize='unicode61'
);

CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL,             -- doc_id
  dst TEXT NOT NULL,             -- doc_id, or an unresolved target
  kind TEXT NOT NULL CHECK (kind IN ('wikilink', 'topic', 'tag')),
  resolved INTEGER NOT NULL,
  PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
-- keys: embedding_model, embedding_dim, schema_version, last_run
"""

#: Same reasoning as chunks_fts's doc_id column: vec0 has no foreign keys, so
#: chunk.delete_chunks needs a way to clear a document's vectors directly. vec0
#: does not support UNINDEXED metadata columns the way fts5 does, so this one
#: is deleted by chunk_id list instead — see delete_chunks.
_CHUNKS_VEC_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0("
    "chunk_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
)


def connect(path: Path, *, embedding_dim: int = 1024) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    # Off by default in sqlite3; ON DELETE CASCADE above is a no-op without it,
    # and a doc's chunks/edges would outlive the doc that owns them.
    conn.execute("PRAGMA foreign_keys = ON")
    # The API server (vault_ask.api.app) reads from one connection while a
    # background loop indexes from another, both against this same file. WAL
    # lets readers proceed without waiting on — or seeing a torn view from —
    # a writer's in-progress transaction; the default rollback-journal mode
    # would serialize them and risk exactly the interleaving this is for.
    conn.execute("PRAGMA journal_mode = WAL")

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.executescript(SCHEMA)
    # IF NOT EXISTS: on a file that already has chunks_vec at a different
    # width, this is correctly a no-op — reconciling that mismatch is
    # ensure_embedding_space's job (it may need to DROP and recreate, which
    # this constructor-time call must never do on its own).
    conn.execute(_CHUNKS_VEC_DDL.format(dim=embedding_dim))
    conn.commit()
    return conn
