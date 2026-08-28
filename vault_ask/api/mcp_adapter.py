"""MCP adapter — exposes retrieval as tools, not one `ask` tool (README
"Architecture"): the calling model can do its own multi-hop. Only
`vault_search` is the RAG entry point; `vault_read`, `vault_neighbors` and
`vault_topics` are why the graph exists — a model that gets a `vault_search`
hit can follow it to the whole note, its neighbours, or the topic it belongs
to, the same way a person would in Obsidian.

``allow_web`` defaults **false** here, same reasoning as the OpenAI shim
(``vault_ask/api/openai_shim.py``): a model this application does not control
is holding the transcript. Every tool takes it, not just `vault_search` — the
core invariant (README "Corpus and sensitivity": a `personal` chunk is never
placed in a context that might also carry a web tool) does not stop applying
just because a note was reached by `vault_read` instead of a search hit.

`conn`/`cfg` are set once, at startup (`configure`), not per-call: a tool
function's signature is the JSON schema shown to the calling model, so they
cannot be ordinary parameters the way they are everywhere else in this
codebase — there is exactly one connection and one config for the process's
whole life, so module state costs nothing a request-scoped mechanism would
have bought back.
"""

from __future__ import annotations

import sqlite3
from typing import Any, cast

from mcp.server.mcpserver import MCPServer

from ..ask import retrieve
from ..config import Settings

server = MCPServer("vault-ask")

_conn: sqlite3.Connection | None = None
_cfg: Settings | None = None


def configure(conn: sqlite3.Connection, cfg: Settings) -> None:
    global _conn, _cfg
    _conn = conn
    _cfg = cfg


def _connection() -> sqlite3.Connection:
    assert _conn is not None, "mcp_adapter.configure() was never called"
    return _conn


def _settings() -> Settings:
    assert _cfg is not None, "mcp_adapter.configure() was never called"
    return _cfg


def _sensitivity_filter(allow_web: bool) -> str | None:
    return "open" if allow_web else None


def _resolve_doc(
    conn: sqlite3.Connection, path_or_slug: str, *, sensitivity: str | None = None
) -> sqlite3.Row | list[str] | None:
    """A path, a bare filename, or a note title -> its `docs` row.

    Returns ``None`` for no match, a list of candidate paths for an ambiguous
    bare-title match, or the row itself once resolution is unambiguous. Same
    resolution order as `vault_ask.edges.resolve`, reimplemented rather than
    shared — that one resolves *wikilink targets found while indexing*, keyed
    off a lookup table built once per run over the whole corpus; this
    resolves *one name a caller just typed*, a single real-time lookup where
    building that table would be pure overhead.

    ``sensitivity`` gates **resolution itself**, which is what makes this the
    single choke point for the whole tool surface: a note the caller may not
    see does not resolve, so it cannot be confirmed to exist, cannot have its
    real path echoed back by an "ambiguous — matches: ..." message, and cannot
    have its title or link structure returned by `vault_neighbors`. Filtering
    later — at the point of returning the body — would have left every one of
    those metadata channels open, which is exactly what it did before.
    """
    needle = path_or_slug.strip().lower()
    candidate = needle if needle.endswith(".md") else f"{needle}.md"

    row = conn.execute(
        "SELECT * FROM docs WHERE doc_id = :doc_id "
        "  AND (:sensitivity IS NULL OR sensitivity = :sensitivity)",
        {"doc_id": candidate, "sensitivity": sensitivity},
    ).fetchone()
    if row is not None:
        return cast(sqlite3.Row, row)

    matches = conn.execute(
        "SELECT * FROM docs WHERE (lower(title) = :needle OR lower(path) LIKE :like) "
        "  AND (:sensitivity IS NULL OR sensitivity = :sensitivity)",
        {"needle": needle, "like": f"%/{needle}.md", "sensitivity": sensitivity},
    ).fetchall()
    if len(matches) == 1:
        return cast(sqlite3.Row, matches[0])
    if len(matches) > 1:
        return [m["path"] for m in matches]
    return None


def _resolved_neighbor(
    conn: sqlite3.Connection, doc_id: str, kind: str, sensitivity: str | None
) -> dict[str, Any] | None:
    if kind == "tag":
        return {"kind": "tag", "value": doc_id.removeprefix("tag:")}
    target = conn.execute(
        "SELECT path, title, sensitivity FROM docs WHERE doc_id = ?", (doc_id,)
    ).fetchone()
    if target is None or (sensitivity and target["sensitivity"] != sensitivity):
        return None
    return {"kind": kind, "path": target["path"], "title": target["title"]}


@server.tool()
async def vault_search(query: str, k: int = 8, allow_web: bool = False) -> list[dict[str, Any]]:
    """Hybrid search over the vault — FTS5 + vector + one-hop graph expansion,
    the same pipeline `python -m vault_ask ask` uses. The RAG entry point; the
    other three tools exist to let you follow a hit further than a keyword
    match can reach.

    allow_web: set true only if you (the calling model/agent) may also use a
    web-search tool in this same conversation — it narrows results to `open`
    chunks. Personal notes (journal, people, tastings) are excluded whenever
    this is true, by design, not by omission.
    """
    hits = await retrieve(_connection(), _settings(), query, allow_web=allow_web, top_k=k)
    return [
        {
            "path": hit.path,
            "title": hit.title,
            "heading_path": hit.heading_path,
            "citation": f"[[{hit.path}|{hit.title}]]",
            "sensitivity": hit.sensitivity,
            "score": hit.score,
            "text": f"{hit.prelude}\n\n{hit.body}" if hit.prelude else hit.body,
        }
        for hit in hits
    ]


@server.tool()
async def vault_read(path: str, allow_web: bool = False) -> str:
    """The full text of one note, reassembled from its indexed chunks (headings
    restored). Use the `path` a `vault_search` hit gave you, or a note title.

    allow_web: same meaning as in `vault_search` — a `personal` note is not
    readable when true.
    """
    conn = _connection()
    sensitivity = _sensitivity_filter(allow_web)
    doc = _resolve_doc(conn, path, sensitivity=sensitivity)
    if doc is None:
        # Deliberately the same answer a genuinely absent note gets. The old
        # message named the note ("<path> is a `personal` note and allow_web is
        # true — refusing...") which withheld the body but confirmed the note
        # exists at that exact path, to a caller explicitly told it may not see
        # it. Existence is the thing worth hiding here; the hint below points a
        # legitimate local caller at the fix without asserting anything.
        return (
            f"No note found matching {path!r}. (If this note is `personal`, it is "
            "not visible while allow_web is true — call again with allow_web=false "
            "for a local-only conversation.)"
        )
    if isinstance(doc, list):
        return f"{path!r} is ambiguous — matches: " + ", ".join(doc)

    rows = conn.execute(
        "SELECT heading_path, body FROM chunks WHERE doc_id = ? ORDER BY ordinal",
        (doc["doc_id"],),
    ).fetchall()

    parts = [f"# {doc['title']}", f"[[{doc['path']}|{doc['title']}]]"]
    last_heading = None
    for row in rows:
        heading = row["heading_path"] or ""
        if heading and heading != last_heading:
            parts.append(f"## {heading}")
        last_heading = heading
        parts.append(row["body"])
    return "\n\n".join(parts)


@server.tool()
async def vault_neighbors(path: str, allow_web: bool = False) -> dict[str, Any]:
    """What this note links to, and what links to it — wikilinks, topic-note
    membership, and shared tags (README "Index"). This is the graph
    `vault_search`'s expansion walks automatically for a hit; use this tool to
    walk it yourself, in either direction, from a note you already have.

    allow_web: same meaning as elsewhere — when true, `personal` notes are left
    out of the result *and* a `personal` note cannot itself be the subject.
    """
    conn = _connection()
    sensitivity = _sensitivity_filter(allow_web)
    # Gating resolution covers the subject too. Previously only *neighbours*
    # were filtered, so asking about a `personal` note returned its real path,
    # its title, and its full open-neighbour link structure — the body was
    # never the only thing worth withholding.
    doc = _resolve_doc(conn, path, sensitivity=sensitivity)
    if doc is None:
        return {"error": f"No note found matching {path!r}."}
    if isinstance(doc, list):
        return {"error": f"{path!r} is ambiguous — matches: " + ", ".join(doc)}

    outbound: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT dst, kind, resolved FROM edges WHERE src = ?", (doc["doc_id"],)
    ).fetchall():
        if not row["resolved"] and row["kind"] != "tag":
            # Not sensitivity-filtered, deliberately: `resolved = 0` means this
            # target matched no doc in the corpus, so there is no `personal`
            # note being disclosed — only the raw link text an *open* note
            # already carries in its body, which vault_read hands over anyway.
            # Filtering it would suggest a protection that isn't real.
            outbound.append({"kind": row["kind"], "unresolved_target": row["dst"]})
            continue
        neighbor = _resolved_neighbor(conn, row["dst"], row["kind"], sensitivity)
        if neighbor is not None:
            outbound.append(neighbor)

    inbound: list[dict[str, Any]] = []
    for row in conn.execute(
        "SELECT src, kind FROM edges WHERE dst = ? AND resolved = 1", (doc["doc_id"],)
    ).fetchall():
        neighbor = _resolved_neighbor(conn, row["src"], row["kind"], sensitivity)
        if neighbor is not None:
            inbound.append(neighbor)

    return {"path": doc["path"], "title": doc["title"], "outbound": outbound, "inbound": inbound}


@server.tool()
async def vault_topics(allow_web: bool = False) -> list[dict[str, Any]]:
    """Every note that functions as a topic page — one with at least one
    `topic` edge pointing *from* it (README "Index": membership in a
    `<!-- begin:clippings -->` region). Not hardcoded to `99 topics/`: any note
    that actually aggregates others qualifies, wherever it lives.

    allow_web: same meaning as elsewhere — `personal` topic pages are left out
    when true (uncommon, but the vault's sensitivity rules make no exception
    for topic notes specifically).
    """
    sensitivity = _sensitivity_filter(allow_web)
    rows = _connection().execute(
        """
        SELECT d.path, d.title, COUNT(*) AS member_count
        FROM edges e
        JOIN docs d ON d.doc_id = e.src
        WHERE e.kind = 'topic' AND e.resolved = 1
          AND (:sensitivity IS NULL OR d.sensitivity = :sensitivity)
        GROUP BY e.src
        ORDER BY member_count DESC
        """,
        {"sensitivity": sensitivity},
    ).fetchall()
    return [
        {"path": row["path"], "title": row["title"], "member_count": row["member_count"]}
        for row in rows
    ]
