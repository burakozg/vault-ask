"""Embeddings — always local (README "Models"): it is the one call that touches
every note in the vault, including the `personal` ones, on every rebuild.
`models.embedding_base_url` names a machine on the LAN running Ollama, not this
process (see OLLAMA-SETUP.md); nothing here hardcodes an address.
"""

from __future__ import annotations

import logging
import sqlite3
import struct

import litellm

from .config import Settings

log = logging.getLogger("vault_ask.embed")

# Same reasoning as vault_ask.ask: litellm chatters on import and phones home
# for version checks by default.
litellm.telemetry = False
litellm.suppress_debug_info = True


class EmbeddingSpaceChanged(Exception):
    """`models.embedding`/`embedding_dim` no longer match what's in the index.

    Mixing vector spaces silently produces retrieval that is wrong in a way no
    test catches (README "Index") — this is the loud failure instead.
    """


def _serialize(vector: list[float]) -> bytes:
    return struct.pack(f"{len(vector)}f", *vector)


def _meta_get(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _meta_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def ensure_embedding_space(conn: sqlite3.Connection, cfg: Settings, *, rebuild: bool) -> None:
    """Reconcile `chunks_vec`'s shape with `models.embedding`/`embedding_dim`.

    A fresh index (no meta recorded yet) just records the current model/dim.
    A match is a no-op. A mismatch without ``--rebuild`` raises — silently
    querying a vec0 column of the wrong width for the vectors already in it is
    the failure this exists to prevent. A mismatch with ``--rebuild`` drops and
    recreates `chunks_vec` at the new width (old vectors are gone regardless of
    doc-level change detection — there is no partial migration between vector
    spaces) and records the new model/dim.
    """
    model = cfg.models.embedding
    dim = cfg.models.embedding_dim
    stored_model = _meta_get(conn, "embedding_model")
    stored_dim_raw = _meta_get(conn, "embedding_dim")
    stored_dim = int(stored_dim_raw) if stored_dim_raw is not None else None

    if stored_model is None:
        _meta_set(conn, "embedding_model", model)
        _meta_set(conn, "embedding_dim", str(dim))
        conn.commit()
        return

    if stored_model == model and stored_dim == dim:
        return

    if not rebuild:
        raise EmbeddingSpaceChanged(
            f"embedding model/dimension changed ({stored_model}/{stored_dim} -> {model}/{dim}) "
            "— run `vault_ask index --rebuild`"
        )

    log.warning(
        "embed.space_reset old_model=%s old_dim=%s new_model=%s new_dim=%s",
        stored_model,
        stored_dim,
        model,
        dim,
    )
    conn.execute("DROP TABLE IF EXISTS chunks_vec")
    conn.execute(
        "CREATE VIRTUAL TABLE chunks_vec USING vec0"
        f"(chunk_id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
    )
    _meta_set(conn, "embedding_model", model)
    _meta_set(conn, "embedding_dim", str(dim))
    conn.commit()


async def embed_texts(cfg: Settings, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    response = await litellm.aembedding(
        model=cfg.models.embedding,
        input=texts,
        api_base=cfg.models.embedding_base_url,
    )
    return [item["embedding"] for item in response.data]


async def embed_missing_chunks(conn: sqlite3.Connection, cfg: Settings, doc_id: str) -> int:
    """Embed whichever of a document's chunks have no vector yet.

    Prelude and body together, the same text a human reader would use to judge
    relevance (README "Chunking") — the prelude alone is what makes a chunk
    from the middle of a clipping retrievable at all.
    """
    rows = conn.execute(
        "SELECT c.chunk_id, c.prelude, c.body FROM chunks c "
        "LEFT JOIN chunks_vec v ON v.chunk_id = c.chunk_id "
        "WHERE c.doc_id = ? AND v.chunk_id IS NULL",
        (doc_id,),
    ).fetchall()
    if not rows:
        return 0

    texts = [f"{row['prelude']}\n\n{row['body']}" for row in rows]
    vectors = await embed_texts(cfg, texts)
    for row, vector in zip(rows, vectors, strict=True):
        conn.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (row["chunk_id"], _serialize(vector)),
        )
    return len(rows)
