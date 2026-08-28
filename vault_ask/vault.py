"""The vault's CouchDB, in Self-hosted LiveSync's document format — read-only.

**Ported from podcast-digest** (`podcast_agent/vault.py`) / `clippings-topics`
(`clippings_topics/vault.py`), which both also carry `LiveSyncVault.list_prefix`
for the same reason this one needs it: discovering what to read rather than
being told a path. The read path (`list_prefix`, `read`, `_markdown_from`,
`_get`) is the same code as those two; a bug found there is a bug here too, and
should be ported to all three. What this copy drops, deliberately: every write
method (`project`, `soft_delete`, `_put*`). vault-ask is the first application
on this vault that does not write to it (see README) — there is no merge path,
no marker pair, and reintroducing a write method here would be adding back a
capability the whole project exists to not have.

Two documents per file:

* a **chunk**, content-addressed, so identical text is stored once;
* an **entry**, keyed by the *lowercased* vault path, listing its chunks.

Two plugin settings must stay off on the vault side, and both break this
silently: ``encrypt`` (E2EE) and ``usePathObfuscation``. Chunks are plaintext,
keyed by path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

import httpx

from .config import VaultConfig

log = logging.getLogger("vault_ask.vault")


class VaultUnavailable(Exception):
    """The vault database cannot be reached, or refused a read."""


def _q(doc_id: str) -> str:
    from urllib.parse import quote

    # Entry ids are vault paths and contain "/" — left unencoded, CouchDB parses
    # them as db/doc/attachment segments and the read lands somewhere else.
    return quote(doc_id, safe="")


@dataclass(frozen=True)
class Entry:
    """One file in the vault, as the listing sees it."""

    doc_id: str  # lowercased vault path — the CouchDB _id
    path: str  # real case
    rev: str
    children: tuple[str, ...]
    mtime: int


class VaultReader(Protocol):
    """What `vault_ask.ingest` needs from a vault — satisfied by `LiveSyncVault`
    and, in tests, by a fake with no network underneath (matching
    podcast-digest's `FakeLLM`/`FakeASR` pattern: production code depends on
    the Protocol, not the concrete client).
    """

    async def list_prefix(self, prefix: str) -> list[Entry]: ...
    async def read(self, entry: Entry) -> str | None: ...


class LiveSyncVault:
    def __init__(self, cfg: VaultConfig, password: str | None) -> None:
        self._cfg = cfg
        base = (cfg.couchdb_url or "").rstrip("/")
        self._client = (
            httpx.AsyncClient(
                base_url=base,
                auth=(cfg.user, password or ""),
                timeout=cfg.timeout_s,
            )
            if base
            else None
        )

    @property
    def name(self) -> str:
        return f"vault:{(self._cfg.couchdb_url or '').rstrip('/')}/{self._cfg.db}"

    async def list_prefix(self, prefix: str) -> list[Entry]:
        """Every live file under ``prefix`` (``""`` lists the whole vault).

        Entries are keyed by lowercased path, so a folder is a key range:
        ``startkey="10 raw/"`` to ``endkey="10 raw0"`` — ``0`` being the byte
        after ``/``. An empty prefix has no such byte to bump, so it spans to
        the highest possible CouchDB key instead.

        **``include_docs=true`` is not an optimisation, it is the correctness
        requirement.** LiveSync does not tombstone a deleted note: it keeps a
        live CouchDB document and sets ``deleted: true`` in the *body*. So
        ``_all_docs`` lists deleted notes exactly like present ones, and the
        flag is only visible in the document — trusting the row list means
        answering questions from notes that exist on no device.
        """
        if self._client is None:
            raise VaultUnavailable("vault.couchdb_url is not set")
        end = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else "￰"
        try:
            response = await self._client.get(
                f"/{self._cfg.db}/_all_docs",
                params={
                    "startkey": f'"{prefix}"',
                    "endkey": f'"{end}"',
                    "include_docs": "true",
                },
            )
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        if response.status_code != 200:
            raise VaultUnavailable(
                f"{self.name} refused a listing: HTTP {response.status_code} {response.text[:200]}"
            )

        entries: list[Entry] = []
        skipped = 0
        non_entry = 0
        for row in response.json().get("rows") or []:
            # A `?keys=` lookup for something absent yields {"key", "error"}
            # rather than a doc; a range GET should not produce these, but a
            # missing `id` would KeyError below rather than being ignored.
            if "id" not in row:
                continue
            doc = row.get("doc") or {}
            if doc.get("deleted"):
                skipped += 1
                continue
            # `_all_docs` spans *every* id in the database, so this listing also
            # carries LiveSync's chunk docs ("h:t<hash>", ~21.8k of them on this
            # vault), CouchDB's own `_design/*`, and singletons like
            # `obsydian_livesync_version`. They were previously excluded only
            # incidentally, downstream, by the `**/*.md` corpus glob failing to
            # match them — so `corpus.include: ["**"]` would have started
            # ingesting chunk documents as if they were notes. Entry docs are
            # the ones carrying a type; filter on it here where it is known.
            if str(doc.get("type") or "") not in ("plain", "newnote"):
                non_entry += 1
                continue
            entries.append(
                Entry(
                    doc_id=str(row["id"]),
                    path=str(doc.get("path") or row["id"]),
                    rev=str((row.get("value") or {}).get("rev") or ""),
                    children=tuple(str(c) for c in (doc.get("children") or [])),
                    mtime=int(doc.get("mtime") or 0),
                )
            )
        # The soft-delete filter is the whole reason include_docs is requested,
        # and it is invisible when it silently stops working — this count is the
        # only signal that the body-`deleted` assumption still holds. (The line
        # exists in clippings-topics' equivalent and was dropped in the port,
        # leaving `skipped` incremented and never read.)
        log.info(
            "vault.listed prefix=%s live=%d soft_deleted=%d non_entry=%d",
            prefix or "(all)",
            len(entries),
            skipped,
            non_entry,
        )
        # Sorted so anything downstream that iterates is stable run to run.
        return sorted(entries, key=lambda e: e.doc_id)

    async def read(self, entry: Entry) -> str | None:
        """A note's markdown, reassembled from its chunks. None if torn."""
        return await self._markdown_from(entry.children)

    async def _markdown_from(self, children: tuple[str, ...]) -> str | None:
        # An entry doc with no chunk list is torn, not empty. `"".join([])` is
        # `""`, which would sail through as real content: the note would be
        # indexed with an empty body and `sha256("")` as its hash. That matters
        # well beyond the odd malformed doc — drop `include_docs=true` from the
        # listing above and *every* row loses its `children`, so the whole
        # vault would re-ingest as empty and quietly blank the index instead of
        # failing. A genuinely empty note has one chunk holding "".
        if not children:
            return None
        parts = []
        for chunk_id in children:
            chunk = await self._get(chunk_id)
            if chunk is None:
                return None  # torn note; safer to skip than to answer from half of it
            parts.append(str(chunk.get("data") or ""))
        return "".join(parts)

    async def _get(self, doc_id: str) -> dict[str, object] | None:
        assert self._client is not None
        try:
            response = await self._client.get(f"/{self._cfg.db}/{_q(doc_id)}")
        except httpx.HTTPError as exc:
            raise VaultUnavailable(f"{self.name} unreachable: {type(exc).__name__}: {exc}") from exc
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise VaultUnavailable(
                f"{self.name} refused a read: HTTP {response.status_code} {response.text[:200]}"
            )
        doc: dict[str, object] = response.json()
        return doc

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
