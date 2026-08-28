"""One-off: compute edges for docs that were indexed before edges existed.

Not part of the CLI — normal `index` runs compute edges for new/changed docs
as part of chunking (see ingest.py). This exists only for the one-time
transition of an index built before graph expansion landed, where every doc
is already `unchanged` and would never re-enter that path. Re-reads each doc
fresh from the vault (cheap: no chunking, no embedding, chunks_vec is never
touched) and writes edges for it directly.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from vault_ask.config import load_settings
from vault_ask.db import connect
from vault_ask.edges import replace_edges, resolve_pending
from vault_ask.frontmatter import parse_frontmatter
from vault_ask.vault import LiveSyncVault, VaultUnavailable

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("backfill_edges")


async def main() -> int:
    cfg = load_settings()
    conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
    known_doc_ids = {row["doc_id"] for row in conn.execute("SELECT doc_id FROM docs").fetchall()}
    log.info("known docs: %d", len(known_doc_ids))

    vault = LiveSyncVault(
        cfg.vault,
        cfg.vault_couchdb_password.get_secret_value() if cfg.vault_couchdb_password else None,
    )
    try:
        entries = await vault.list_prefix("")
        done = 0
        for entry in entries:
            if entry.doc_id not in known_doc_ids:
                continue
            markdown = await vault.read(entry)
            if markdown is None:
                continue
            frontmatter = parse_frontmatter(markdown)
            replace_edges(conn, doc_id=entry.doc_id, markdown=markdown, frontmatter=frontmatter)
            done += 1
            if done % 200 == 0:
                conn.commit()
                log.info("backfilled %d/%d", done, len(known_doc_ids))
        conn.commit()
        log.info("backfilled %d docs total", done)

        resolved = resolve_pending(conn)
        conn.commit()
        log.info("resolved %d edges on catch-up pass", resolved)
    except VaultUnavailable as exc:
        log.error("vault unavailable: %s", exc)
        return 1
    finally:
        await vault.close()
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
