"""Link-graph edges: wikilinks, topic-note membership, tags (README "Index").

Extracted from what the writers already put in the vault — no new authoring
surface, no marker pair of this project's own. Three sources:

* `[[path/to/note|alias]]` anywhere in a note body -> a `wikilink` edge.
* the same links, when they additionally fall inside a
  `<!-- begin:clippings --> ... <!-- end:clippings -->` region (the marker
  `clippings-topics` writes into `99 topics/*.md`) -> a `topic` edge from the
  topic note to each clipping. Both edges get written for such a link — a
  wikilink is still a wikilink even where it also happens to sit inside a
  curated membership region, and `topic` edges are what graph expansion
  (`vault_ask.retrieval.expand_graph`) specifically follows.
* frontmatter `tags:` -> `tag` edges. A tag is never a document, so its `dst`
  is a synthetic `tag:<value>` key and it is never "resolved".

Recomputed whole per document on every (re)chunk, same delete-then-insert
rhythm as chunks: an edit can remove a link, and only a full replacement makes
its edge disappear.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

_WIKILINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]")
_TOPIC_REGION_RE = re.compile(
    r"<!--\s*begin:clippings\s*-->(.*?)<!--\s*end:clippings\s*-->", re.DOTALL
)


def extract_wikilinks(body: str) -> list[str]:
    return [m.group(1).strip() for m in _WIKILINK_RE.finditer(body) if m.group(1).strip()]


def extract_topic_links(body: str) -> list[str]:
    links: list[str] = []
    for region in _TOPIC_REGION_RE.finditer(body):
        links.extend(extract_wikilinks(region.group(1)))
    return links


def extract_tags(frontmatter: dict[str, Any]) -> list[str]:
    tags = frontmatter.get("tags")
    if isinstance(tags, str):
        return [tags] if tags else []
    if isinstance(tags, list):
        return [str(t) for t in tags if t]
    return []


def resolve(
    target: str, *, doc_ids: set[str], by_title: dict[str, list[str]]
) -> tuple[str, bool]:
    """A wikilink target -> (dst, resolved).

    Tries, in order: the target as a path-qualified doc_id (with or without
    `.md`), then its basename against note titles/filenames. Ambiguous (more
    than one note shares that basename) or no match at all both come back
    unresolved — a wrong guess is worse than admitting the link is ambiguous,
    the same reasoning clippings-topics uses for writing links path-qualified
    in the first place.
    """
    lowered = target.strip().lower()
    candidate = lowered if lowered.endswith(".md") else f"{lowered}.md"
    if candidate in doc_ids:
        return candidate, True
    if lowered in doc_ids:
        return lowered, True

    basename = target.rsplit("/", 1)[-1].strip().lower()
    matches = by_title.get(basename, [])
    if len(matches) == 1:
        return matches[0], True
    return target, False


def _lookup_tables(conn: sqlite3.Connection) -> tuple[set[str], dict[str, list[str]]]:
    doc_ids: set[str] = set()
    by_title: dict[str, list[str]] = {}
    for row in conn.execute("SELECT doc_id, path, title FROM docs").fetchall():
        doc_ids.add(row["doc_id"])
        filename = row["path"].rsplit("/", 1)[-1].removesuffix(".md").strip().lower()
        title = (row["title"] or "").strip().lower()
        for key in {filename, title}:
            if key:
                by_title.setdefault(key, []).append(row["doc_id"])
    return doc_ids, by_title


def delete_edges(conn: sqlite3.Connection, doc_id: str) -> None:
    """Outbound edges only — README "Ingestion": a deleted doc's own links go
    with it, but other docs' links *to* it are left (they are not this doc's
    to delete, and a dangling `dst` is harmless: nothing joins to a `docs` row
    that no longer exists).
    """
    conn.execute("DELETE FROM edges WHERE src = ?", (doc_id,))


def replace_edges(
    conn: sqlite3.Connection, *, doc_id: str, markdown: str, frontmatter: dict[str, Any]
) -> int:
    delete_edges(conn, doc_id)
    doc_ids, by_title = _lookup_tables(conn)

    rows: dict[tuple[str, str], int] = {}  # (dst, kind) -> resolved
    for target in extract_topic_links(markdown):
        dst, resolved = resolve(target, doc_ids=doc_ids, by_title=by_title)
        rows[(dst, "topic")] = int(resolved)
    for target in extract_wikilinks(markdown):
        dst, resolved = resolve(target, doc_ids=doc_ids, by_title=by_title)
        rows[(dst, "wikilink")] = int(resolved)
    for tag in extract_tags(frontmatter):
        rows[(f"tag:{tag}", "tag")] = 0

    for (dst, kind), is_resolved in rows.items():
        conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst, kind, resolved) VALUES (?, ?, ?, ?)",
            (doc_id, dst, kind, is_resolved),
        )
    return len(rows)


def resolve_pending(conn: sqlite3.Connection) -> int:
    """Retry resolution for every still-unresolved wikilink/topic edge, not
    just the docs touched this run. A link's target may not have existed yet
    when the linking note was indexed (see the ordering note in
    vault_ask.ingest); this is what lets it resolve later, once the target
    itself gets indexed, without the linking note needing to change again.
    """
    doc_ids, by_title = _lookup_tables(conn)
    pending = conn.execute(
        "SELECT src, dst, kind FROM edges WHERE resolved = 0 AND kind IN ('wikilink', 'topic')"
    ).fetchall()

    resolved_count = 0
    for row in pending:
        new_dst, ok = resolve(row["dst"], doc_ids=doc_ids, by_title=by_title)
        if not ok or new_dst == row["dst"]:
            continue
        conn.execute(
            "DELETE FROM edges WHERE src = ? AND dst = ? AND kind = ?",
            (row["src"], row["dst"], row["kind"]),
        )
        conn.execute(
            "INSERT OR IGNORE INTO edges (src, dst, kind, resolved) VALUES (?, ?, ?, 1)",
            (row["src"], new_dst, row["kind"]),
        )
        resolved_count += 1
    return resolved_count
