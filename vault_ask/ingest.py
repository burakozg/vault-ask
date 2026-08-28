"""Change detection against the vault, plus chunking and embedding on top of it.

The core classification (README "Build order", step 1) deciding, for every
candidate note, which of five things is true:

| in listing | in cache | | read? |
|---|---|---|---|
| rev differs, hash differs | yes | **changed** — re-chunk, re-embed | yes |
| rev differs, hash same    | yes | **touched** — cache rev bump only | yes |
| —                         | no  | **new** — chunk and embed        | yes |
| gone / deleted            | yes | **deleted** — drop the doc and its chunks | no |
| rev same                  | yes | **unchanged**                    | no |

Two stages deliberately, same reasoning as clippings-topics: a changed `rev`
costs a read, but only a changed **content hash** costs the expensive work
(chunking, embedding) — LiveSync rewrites `rev` for its own reasons unrelated
to content.

``--rebuild`` disables the `touched`/`unchanged` shortcuts (every candidate is
read and reclassified as `new` or `changed`) but still diffs against the real
cache for `deleted` — a rebuild re-embeds everything live, it does not stop
noticing that something was removed.

Embedding is a separate pass after the doc/chunk transaction commits (see
`_embed_pending`), not folded into it: it is a network call to another machine
(OLLAMA-SETUP.md), and a machine being briefly unreachable must not roll back
an otherwise-successful chunking run, nor abandon it — the next `index` run
picks up wherever embedding left off, doc-by-doc, independent of whether that
doc's content changed this run.

Edge extraction (wikilinks, topic-note membership, tags — see
`vault_ask.edges`) happens inline with chunking, since it needs nothing the
per-doc loop doesn't already have. *Resolving* an edge's target to a real
doc_id is a separate pass after commit, for the opposite reason from
embedding: it is cheap, local and synchronous, but a note processed early in
a big batch can link to one processed later in the same batch — the target
does not exist in `docs` yet at extraction time, so resolution is retried,
whole-table, once every doc in this run has actually landed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass, field

from .chunk import delete_chunks, replace_chunks
from .config import Settings
from .edges import delete_edges as delete_doc_edges
from .edges import replace_edges, resolve_pending
from .embed import embed_missing_chunks, ensure_embedding_space
from .frontmatter import classify_sensitivity, glob_match, in_corpus, parse_frontmatter
from .vault import Entry, VaultReader

log = logging.getLogger("vault_ask.ingest")


@dataclass(frozen=True)
class _CachedDoc:
    rev: str
    content_hash: str


@dataclass
class IngestReport:
    new: list[Entry] = field(default_factory=list)
    changed: list[Entry] = field(default_factory=list)
    touched: list[Entry] = field(default_factory=list)
    unchanged: list[Entry] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    #: Notes whose chunk list pointed at a missing chunk document — skipped
    #: rather than guessed at; retried automatically next run.
    skipped_torn: list[str] = field(default_factory=list)
    #: Chunks embedded this run — includes catch-up for docs chunked in a
    #: previous run whose embedding call failed or wasn't yet configured.
    embedded: int = 0
    #: Docs whose embedding call failed this run; their chunks remain
    #: unembedded and are retried automatically next run.
    embed_failed: int = 0
    #: Previously-unresolved wikilink/topic edges that resolved this run —
    #: not limited to this run's docs; see edges.resolve_pending.
    edges_resolved: int = 0
    #: Docs whose `sensitivity` changed this run *without* their content
    #: changing — i.e. because config did. See _reclassify_sensitivity.
    reclassified: int = 0

    def reads(self) -> int:
        return len(self.new) + len(self.changed) + len(self.touched)

    def summary(self) -> str:
        lines = [
            f"  new       {len(self.new):>5}",
            f"  changed   {len(self.changed):>5}",
            f"  touched   {len(self.touched):>5}  (rev bump only, no re-embed)",
            f"  unchanged {len(self.unchanged):>5}  (not read)",
            f"  deleted   {len(self.deleted):>5}",
            f"  embedded  {self.embedded:>5} chunk(s)"
            + (f", {self.embed_failed} doc(s) failed" if self.embed_failed else ""),
            f"  edges     {self.edges_resolved:>5} resolved this run",
        ]
        if self.reclassified:
            lines.append(
                f"  reclassed {self.reclassified:>5}  (sensitivity changed by config, not content)"
            )
        if self.skipped_torn:
            lines.append(f"  skipped   {len(self.skipped_torn):>5}  (torn note — missing chunk)")
        return "\n".join(lines)


def _content_hash(markdown: str) -> str:
    return hashlib.sha256(markdown.encode("utf-8")).hexdigest()


def _title_of(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".md")


def _load_cache(conn: sqlite3.Connection) -> dict[str, _CachedDoc]:
    rows = conn.execute("SELECT doc_id, rev, content_hash FROM docs").fetchall()
    return {
        row["doc_id"]: _CachedDoc(rev=row["rev"], content_hash=row["content_hash"]) for row in rows
    }


#: Fraction of the index that may be deleted in one run before it needs to be
#: asked for explicitly. Vault edits arrive continuously via LiveSync, so a
#: normal run deletes single digits; a fifth of the corpus disappearing at once
#: is a listing problem far more often than a real one.
MASS_DELETE_FRACTION = 0.2

#: Below this the fraction is meaningless — 1 of 3 docs is 33% and entirely
#: routine. The guard only has teeth on an index big enough for a wipe to hurt.
MASS_DELETE_MIN = 25


class MassDeletionRefused(RuntimeError):
    """A single run tried to delete an implausible share of the index.

    Deletion is a set-difference: anything that makes the listing come back
    short deletes the local index (see run_ingest). `VaultUnavailable` covers
    transport errors and non-200s, but a *successful* HTTP 200 carrying empty
    or truncated `rows` is indistinguishable from "the user deleted everything"
    — a wrong `db` name, a collation or endkey mismatch, a proxy returning a
    cached empty body, or a future CouchDB that paginates by default all land
    here. The cost is asymmetric: refusing wrongly costs one re-run, proceeding
    wrongly costs a full re-embed of the entire vault.
    """


def _check_deletion_floor(
    deleted: list[str], cached_count: int, *, allow_mass_delete: bool
) -> None:
    if allow_mass_delete or cached_count < MASS_DELETE_MIN:
        return
    if len(deleted) < MASS_DELETE_FRACTION * cached_count:
        return
    raise MassDeletionRefused(
        f"refusing to delete {len(deleted)} of {cached_count} indexed docs in one run "
        f"({len(deleted) / cached_count:.0%} — the limit is {MASS_DELETE_FRACTION:.0%}). "
        "The vault listing came back far shorter than the index; check the vault is "
        "serving the right database before assuming these notes are really gone. "
        "Re-run with --allow-mass-delete if this deletion is genuine."
    )


def _reclassify_sensitivity(conn: sqlite3.Connection, cfg: Settings) -> int:
    """Recompute `sensitivity` for every indexed doc. Returns how many changed.

    Sensitivity is derived from path + frontmatter + **config**, and config can
    change while the vault does not. Change detection is rev-based, so after
    editing `sensitivity.personal_paths` every doc is `unchanged`, nothing is
    re-read, and the new rule silently does not apply — a privacy control that
    appears to have been fixed and has not been. That is the same failure mode
    as the fictional paths themselves, one level up.

    Deliberately *not* done via replace_chunks: that deletes the doc's rows
    from `chunks_vec` (see chunk.py — chunk_id is positional, not content
    derived), so routing this through the normal rewrite path would re-embed
    all 5,664 chunks to change one column. Sensitivity is denormalised onto
    `chunks` precisely so it can be updated in place; `chunks_fts` does not
    carry it and needs no touch.
    """
    rows = conn.execute("SELECT doc_id, path, frontmatter, sensitivity FROM docs").fetchall()
    changed = 0
    for row in rows:
        try:
            frontmatter = json.loads(row["frontmatter"] or "{}")
        except json.JSONDecodeError:
            frontmatter = {}
        if not isinstance(frontmatter, dict):
            frontmatter = {}
        current = classify_sensitivity(row["path"], frontmatter, cfg.sensitivity)
        if current == row["sensitivity"]:
            continue
        conn.execute(
            "UPDATE docs SET sensitivity = ? WHERE doc_id = ?", (current, row["doc_id"])
        )
        conn.execute(
            "UPDATE chunks SET sensitivity = ? WHERE doc_id = ?", (current, row["doc_id"])
        )
        log.info(
            "ingest.reclassified path=%s %s -> %s", row["path"], row["sensitivity"], current
        )
        changed += 1
    return changed


def _warn_unmatched_personal_paths(candidates: list[Entry], cfg: Settings) -> None:
    """Warn about any `sensitivity.personal_paths` pattern matching no note.

    A pattern naming a folder that does not exist classifies nothing and fails
    *silently* — the corpus simply comes out all-`open` and the whole allow_web
    gate guards an empty set. That is not hypothetical: this shipped with
    "30 journal/**", "40 people/**" and "50 tastings/**" against a vault whose
    real folders are "30 projects/" and "Tastings/", so all 2,254 notes were
    `open`. Matching is case-sensitive (fnmatchcase), which is the easiest way
    to get this wrong. Loud on every run is the right volume for a privacy
    control that is invisible when broken.
    """
    for pattern in cfg.sensitivity.personal_paths:
        if not any(glob_match(e.path, pattern) for e in candidates):
            log.warning(
                "ingest.personal_path_matches_nothing pattern=%r — notes under it will be "
                "classified `open` and visible when allow_web is true. Patterns are "
                "case-sensitive and match the real path, not the lowercased CouchDB id.",
                pattern,
            )


async def run_ingest(
    vault: VaultReader,
    conn: sqlite3.Connection,
    cfg: Settings,
    *,
    dry_run: bool = False,
    rebuild: bool = False,
    allow_mass_delete: bool = False,
) -> IngestReport:
    entries = await vault.list_prefix("")
    candidates = [e for e in entries if in_corpus(e.path, cfg.corpus.include, cfg.corpus.exclude)]
    _warn_unmatched_personal_paths(candidates, cfg)
    cached = _load_cache(conn)

    report = IngestReport()
    content: dict[str, str] = {}

    for entry in candidates:
        prior = cached.get(entry.doc_id)
        if prior is not None and not rebuild and entry.rev == prior.rev:
            report.unchanged.append(entry)
            continue

        markdown = await vault.read(entry)
        if markdown is None:
            log.warning("ingest.torn path=%s", entry.path)
            report.skipped_torn.append(entry.path)
            continue
        content[entry.doc_id] = markdown
        content_hash = _content_hash(markdown)

        if prior is None:
            report.new.append(entry)
        elif not rebuild and content_hash == prior.content_hash:
            report.touched.append(entry)
        else:
            report.changed.append(entry)

    present_ids = {e.doc_id for e in candidates}
    report.deleted = sorted(doc_id for doc_id in cached if doc_id not in present_ids)

    log.info(
        "ingest.planned candidates=%d new=%d changed=%d touched=%d unchanged=%d deleted=%d",
        len(candidates),
        len(report.new),
        len(report.changed),
        len(report.touched),
        len(report.unchanged),
        len(report.deleted),
    )

    # Checked before the dry-run return so `--dry-run` surfaces the refusal
    # too — the point of a dry run is to find out what an apply would do.
    _check_deletion_floor(report.deleted, len(cached), allow_mass_delete=allow_mass_delete)

    if dry_run:
        return report

    with conn:
        for doc_id in report.deleted:
            # `chunks` cascades from the docs delete below via its foreign key;
            # `chunks_fts` does not (FTS5 has no foreign keys) and needs its
            # own delete. `edges` too — its outbound rows only (delete_edges),
            # never the ones other docs still point at it with.
            delete_chunks(conn, doc_id)
            delete_doc_edges(conn, doc_id)
            conn.execute("DELETE FROM docs WHERE doc_id = ?", (doc_id,))

        for entry in (*report.new, *report.changed):
            markdown = content[entry.doc_id]
            frontmatter = parse_frontmatter(markdown)
            sensitivity = classify_sensitivity(entry.path, frontmatter, cfg.sensitivity)
            # str(): a frontmatter `title` isn't guaranteed to already be one —
            # YAML happily parses an unquoted `title: 2026-08-28` into a
            # datetime.date, which sqlite3 cannot bind as a parameter at all
            # (unlike the JSON column above, this isn't even a silent-mangling
            # risk, it's an outright InterfaceError).
            raw_title = frontmatter.get("title")
            title = str(raw_title) if raw_title else _title_of(entry.path)
            conn.execute(
                """
                INSERT INTO docs
                    (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter)
                VALUES
                    (:doc_id, :path, :title, :rev, :content_hash,
                     :sensitivity, :mtime, :frontmatter)
                ON CONFLICT(doc_id) DO UPDATE SET
                    path = excluded.path,
                    title = excluded.title,
                    rev = excluded.rev,
                    content_hash = excluded.content_hash,
                    sensitivity = excluded.sensitivity,
                    mtime = excluded.mtime,
                    frontmatter = excluded.frontmatter
                """,
                {
                    "doc_id": entry.doc_id,
                    "path": entry.path,
                    "title": title,
                    "rev": entry.rev,
                    "content_hash": _content_hash(markdown),
                    "sensitivity": sensitivity,
                    "mtime": entry.mtime,
                    # default=str: yaml.safe_load turns an unquoted frontmatter
                    # value like `date: 2026-08-28` into a real datetime.date,
                    # which json.dumps otherwise refuses outright — and this
                    # column exists to carry frontmatter through, not to
                    # validate it, so coercing to its string form beats
                    # crashing the whole ingest run over one journal note.
                    "frontmatter": json.dumps(frontmatter, default=str),
                },
            )
            replace_chunks(
                conn,
                doc_id=entry.doc_id,
                path=entry.path,
                title=title,
                sensitivity=sensitivity,
                markdown=markdown,
                cfg=cfg.chunking,
            )
            replace_edges(conn, doc_id=entry.doc_id, markdown=markdown, frontmatter=frontmatter)

        for entry in report.touched:
            conn.execute("UPDATE docs SET rev = ? WHERE doc_id = ?", (entry.rev, entry.doc_id))

    with conn:
        # After the writes above, so docs added this run are classified once by
        # the write path and then simply agree here. Its real job is the docs
        # that were *not* rewritten — `unchanged` is the whole corpus on a
        # normal run, and a config edit has to reach them.
        report.reclassified = _reclassify_sensitivity(conn, cfg)

    with conn:
        report.edges_resolved = resolve_pending(conn)

    if cfg.models.embedding_base_url:
        report.embedded, report.embed_failed = await _embed_pending(conn, cfg, rebuild=rebuild)
    else:
        log.info("ingest.embedding_skipped reason=embedding_base_url_unset")

    return report


async def _embed_pending(
    conn: sqlite3.Connection, cfg: Settings, *, rebuild: bool
) -> tuple[int, int]:
    """Embed every chunk lacking a vector — not just ones from this run's
    new/changed docs. Decoupled from doc-level change detection on purpose:
    a doc chunked while embedding was unconfigured, or whose embedding call
    failed last run, has no rev/hash change to trigger a retry through the
    normal path, so this instead asks the index itself what it's missing.
    """
    ensure_embedding_space(conn, cfg, rebuild=rebuild)

    pending_sql = (
        "SELECT DISTINCT doc_id FROM chunks "
        "WHERE chunk_id NOT IN (SELECT chunk_id FROM chunks_vec)"
    )
    doc_ids = [row["doc_id"] for row in conn.execute(pending_sql).fetchall()]

    embedded = 0
    failed = 0
    for doc_id in doc_ids:
        try:
            embedded += await embed_missing_chunks(conn, cfg, doc_id)
            conn.commit()
        except Exception:
            conn.rollback()
            failed += 1
            log.warning("ingest.embed_failed doc_id=%s", doc_id, exc_info=True)

    log.info(
        "ingest.embedded chunks=%d docs_pending=%d docs_failed=%d", embedded, len(doc_ids), failed
    )
    return embedded, failed
