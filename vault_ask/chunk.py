"""Heading-aware chunking (README "Chunking") and its SQLite persistence.

Split on ``##``/``###`` boundaries, not fixed windows — a fixed window cuts a
paragraph in half as readily as a heading, and a chunk that starts mid-sentence
is a worse retrieval unit than one that starts at a heading. Runts (small
sections) merge into the previous chunk; anything over ``hard_split_tokens``
is split again at paragraph boundaries so no single chunk is too large for the
embedding model's window (step 3) or the answer's context budget.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from .config import ChunkingConfig
from .frontmatter import strip_frontmatter

#: Below this, a section merges into the previous one rather than standing
#: alone — a lone "## See also" with three links is not worth its own
#: retrieval unit. Not user-configurable: it is a floor on chunk usefulness,
#: not a sizing knob like target/hard_split_tokens.
_RUNT_TOKENS = 80

_HEADING_RE = re.compile(r"^(#{2,3})[ \t]+(.+?)\s*$", re.MULTILINE)


@lru_cache(maxsize=1)
def _encoding() -> object:
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    enc = _encoding()
    return len(enc.encode(text))  # type: ignore[attr-defined]


@dataclass(frozen=True)
class RawChunk:
    heading_path: str
    text: str


def _sections(body: str) -> list[RawChunk]:
    """The note split at H2/H3 boundaries, breadcrumbed (`H2 > H3`)."""
    matches = list(_HEADING_RE.finditer(body))
    if not matches:
        text = body.strip()
        return [RawChunk("", text)] if text else []

    sections: list[RawChunk] = []
    preamble = body[: matches[0].start()].strip()
    if preamble:
        sections.append(RawChunk("", preamble))

    stack: list[str] = []
    for index, match in enumerate(matches):
        level = len(match.group(1))  # 2 or 3
        title = match.group(2).strip()
        # A new H2 starts a fresh breadcrumb; an H3 nests under whatever H2 (if
        # any) currently leads the stack.
        stack = [title] if level == 2 else ([stack[0], title] if stack else [title])
        heading_path = " > ".join(stack)

        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
        text = body[start:end].strip()
        if text:
            sections.append(RawChunk(heading_path, text))

    return sections


def _merge_runts(sections: list[RawChunk]) -> list[RawChunk]:
    merged: list[RawChunk] = []
    for section in sections:
        if merged and _tokens(section.text) < _RUNT_TOKENS:
            prior = merged[-1]
            merged[-1] = RawChunk(prior.heading_path, f"{prior.text}\n\n{section.text}")
        else:
            merged.append(section)
    return merged


def _split_paragraph(text: str, hard_split_tokens: int) -> list[str]:
    """A single paragraph too large to keep whole, cut at a token boundary.

    Last resort: everything above this tries to split on document structure
    (headings, then paragraphs) first. A token-boundary cut can land mid-word,
    which is an acceptable cost for a paragraph that was already too dense to
    split any other way.
    """
    enc = _encoding()
    tokens = enc.encode(text)  # type: ignore[attr-defined]
    pieces = []
    for start in range(0, len(tokens), hard_split_tokens):
        pieces.append(enc.decode(tokens[start : start + hard_split_tokens]))  # type: ignore[attr-defined]
    return pieces


def _hard_split(section: RawChunk, cfg: ChunkingConfig) -> list[RawChunk]:
    if _tokens(section.text) <= cfg.hard_split_tokens:
        return [section]

    paragraphs = [p for p in re.split(r"\n{2,}", section.text) if p.strip()]
    pieces: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for paragraph in paragraphs:
        para_tokens = _tokens(paragraph)
        if para_tokens > cfg.hard_split_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            pieces.extend(_split_paragraph(paragraph, cfg.hard_split_tokens))
            continue
        if current and current_tokens + para_tokens > cfg.target_tokens:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(paragraph)
        current_tokens += para_tokens
    if current:
        pieces.append("\n\n".join(current))

    return [RawChunk(section.heading_path, piece) for piece in pieces] or [section]


def chunk_markdown(body: str, cfg: ChunkingConfig) -> list[RawChunk]:
    sections = _merge_runts(_sections(body))
    chunks: list[RawChunk] = []
    for section in sections:
        chunks.extend(_hard_split(section, cfg))
    return chunks


def build_prelude(title: str, path: str, heading_path: str) -> str:
    """What is embedded alongside the body, stripped before display (README
    "Chunking"). Topic memberships join this once edges exist (later work).
    """
    lines = [title]
    if heading_path:
        lines.append(heading_path)
    lines.append(path)
    return "\n".join(lines)


def chunk_id(doc_id: str, heading_path: str, ordinal: int) -> str:
    digest = hashlib.sha256(f"{doc_id}{heading_path}{ordinal}".encode()).hexdigest()
    return digest[:16]


def delete_chunks(conn: sqlite3.Connection, doc_id: str) -> None:
    """Clear a document's chunks from `chunks`, `chunks_fts` and `chunks_vec`.

    `chunks` cascades from a `docs` delete via its foreign key; the other two
    do not (neither FTS5 nor vec0 support foreign keys), so all three need
    clearing explicitly whenever a document's chunks are about to be dropped
    or replaced.

    This is also what stops a subtler bug: `chunk_id` is derived from
    doc_id + heading_path + ordinal, not from content (see chunk_id above), so
    an edit that changes a section's text without changing its heading or
    position produces the *same* chunk_id. Without this cleanup a stale vector
    from the old text would sit in `chunks_vec` under that id forever —
    invisible to `embed_missing_chunks`, which only looks for chunk_ids with
    no vector at all, so it would never be re-embedded.
    """
    rows = conn.execute("SELECT chunk_id FROM chunks WHERE doc_id = ?", (doc_id,)).fetchall()
    chunk_ids = [row["chunk_id"] for row in rows]
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
    conn.execute("DELETE FROM chunks_fts WHERE doc_id = ?", (doc_id,))
    if chunk_ids:
        # `placeholders` is only ever "?, ?, ..." — the values themselves are
        # bound as parameters below, never interpolated.
        placeholders = ",".join("?" * len(chunk_ids))
        conn.execute(
            f"DELETE FROM chunks_vec WHERE chunk_id IN ({placeholders})",  # noqa: S608
            chunk_ids,
        )


def replace_chunks(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    path: str,
    title: str,
    sensitivity: str,
    markdown: str,
    cfg: ChunkingConfig,
) -> int:
    """Whole-document replacement: delete, then insert what the note is now.

    Never a per-chunk diff — an edit can remove a section, and only a full
    replacement makes its chunk disappear (README "Ingestion"). ``markdown`` is
    the whole note, frontmatter included — stripped here so callers never have
    to remember to do it themselves.
    """
    delete_chunks(conn, doc_id)
    raw_chunks = chunk_markdown(strip_frontmatter(markdown), cfg)
    for ordinal, raw in enumerate(raw_chunks):
        cid = chunk_id(doc_id, raw.heading_path, ordinal)
        prelude = build_prelude(title, path, raw.heading_path)
        conn.execute(
            "INSERT INTO chunks"
            " (chunk_id, doc_id, ordinal, heading_path, prelude, body, sensitivity) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (cid, doc_id, ordinal, raw.heading_path, prelude, raw.text, sensitivity),
        )
        conn.execute(
            "INSERT INTO chunks_fts (chunk_id, doc_id, prelude, body) VALUES (?, ?, ?, ?)",
            (cid, doc_id, prelude, raw.text),
        )
    return len(raw_chunks)
