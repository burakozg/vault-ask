"""`LiveSyncVault` against real CouchDB response shapes.

This file did not exist. `respx` was a declared dev dependency that no test
imported, so `list_prefix` — the only code in the project that parses a real
CouchDB response, and the sole enforcement point for LiveSync's soft-delete —
was executed by nothing. Every one of these mutations left the old suite fully
green: deleting the `deleted` check, dropping `include_docs=true`, reading
`deleted` off `row["value"]` instead of the body, or misspelling the key.

The payloads below are **captured from the live vault**, not invented, because
the assumption under test is precisely "what does CouchDB actually return":

    GET /the_brain                      -> doc_del_count: 0
    GET /the_brain/_all_docs?limit=5    -> rows[].value == {"rev": "3-<hash>"}
                                          (no `deleted` key, ever)
    GET /the_brain/<a deleted note>     -> {"deleted": true, "type": "plain", ...}

That combination is the whole point: 1,227 notes are deleted, CouchDB reports
`doc_del_count: 0`, and none of them carry `deleted` at the row level. LiveSync
soft-deletes — a live document with a body flag — so a listing that trusts
`rows[].value` sees every deleted note as present.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from vault_ask.config import VaultConfig
from vault_ask.vault import LiveSyncVault, VaultUnavailable

COUCH = "http://couch.local:5984"
DB = "the_brain"


def _cfg() -> VaultConfig:
    return VaultConfig(couchdb_url=COUCH, db=DB, user="admin")


def _vault() -> LiveSyncVault:
    return LiveSyncVault(_cfg(), "password")


def _live_row(doc_id: str, rev: str = "3-502b7708f51d488d9cb3326cab92631c") -> dict[str, Any]:
    """A present note, exactly as `_all_docs?include_docs=true` returns it."""
    return {
        "id": doc_id,
        "key": doc_id,
        "value": {"rev": rev},  # NB: no `deleted` key. There never is one here.
        "doc": {
            "_id": doc_id,
            "_rev": rev,
            "path": doc_id,
            "children": ["h:tfe36dbceda632f2b23696b39"],
            "ctime": 1787694801469,
            "mtime": 1787743028493,
            "size": 4918,
            "type": "plain",
            "eden": {},
        },
    }


def _soft_deleted_row(doc_id: str, rev: str = "2-f0aedd1edec69ed4fb3f25411c3c2d1e") -> dict[str, Any]:
    """A LiveSync-deleted note. Captured verbatim from the real vault.

    Note what this is NOT: a CouchDB tombstone. The row looks completely
    ordinary — `value` carries only a rev — and the document is live. The only
    evidence is `doc["deleted"]`.
    """
    return {
        "id": doc_id,
        "key": doc_id,
        "value": {"rev": rev},
        "doc": {
            "_id": doc_id,
            "_rev": rev,
            "path": doc_id,
            "children": ["h:tabc123"],
            "ctime": 1787694801469,
            "deleted": True,
            "eden": {},
            "mtime": 1787743028493,
            "size": 4918,
            "type": "plain",
        },
    }


def _chunk_row(chunk_id: str = "h:tfe36dbceda632f2b23696b39") -> dict[str, Any]:
    """A LiveSync content chunk. ~21,800 of these share the database."""
    return {
        "id": chunk_id,
        "key": chunk_id,
        "value": {"rev": "1-92e00fe5f3a531b2cffb0d4d5a6512dc"},
        "doc": {"_id": chunk_id, "data": "# Some note\n\nbody text", "type": "leaf"},
    }


def _design_row() -> dict[str, Any]:
    doc_id = "_design/096ebad658ffb0aee9fa53459163a85b5994b811"
    return {
        "id": doc_id,
        "key": doc_id,
        "value": {"rev": "1-abc"},
        "doc": {"_id": doc_id, "views": {}, "language": "javascript"},
    }


def _mock_listing(rows: list[dict[str, Any]]) -> None:
    respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
        return_value=httpx.Response(200, json={"total_rows": len(rows), "offset": 0, "rows": rows})
    )


@respx.mock
async def test_soft_deleted_note_is_excluded() -> None:
    """The assumption the whole design rests on."""
    _mock_listing([_live_row("10 raw/present.md"), _soft_deleted_row("10 raw/gone.md")])
    entries = await _vault().list_prefix("")
    assert [e.doc_id for e in entries] == ["10 raw/present.md"]


@respx.mock
async def test_deleted_flag_is_read_from_the_body_not_the_row() -> None:
    """A row-level `value.deleted` must NOT be what the filter keys on.

    `_all_docs` in its range-GET form never emits `value.deleted` — only the
    `POST ?keys=[...]` form and `_changes` do, and neither is used. A filter
    reading it there would therefore never fire, and every soft-deleted note
    would be indexed. Here the row says nothing and the body says deleted; the
    note must still be dropped.
    """
    row = _soft_deleted_row("10 raw/gone.md")
    assert "deleted" not in row["value"], "fixture must mirror real CouchDB"
    assert row["doc"]["deleted"] is True
    _mock_listing([row])
    assert await _vault().list_prefix("") == []


@respx.mock
async def test_include_docs_is_requested() -> None:
    """Without it there is no body, so the soft-delete flag is invisible and
    every note also loses its `children` (see test below for that half).
    """
    _mock_listing([_live_row("10 raw/a.md")])
    await _vault().list_prefix("")
    request = respx.calls.last.request
    assert request.url.params["include_docs"] == "true"


@respx.mock
async def test_chunk_and_design_docs_are_not_entries() -> None:
    """`_all_docs` spans every id in the database, not just notes."""
    _mock_listing([_live_row("10 raw/a.md"), _chunk_row(), _design_row()])
    entries = await _vault().list_prefix("")
    assert [e.doc_id for e in entries] == ["10 raw/a.md"]


@respx.mock
async def test_rev_is_taken_from_the_row_value() -> None:
    _mock_listing([_live_row("10 raw/a.md", rev="7-deadbeef")])
    entries = await _vault().list_prefix("")
    assert entries[0].rev == "7-deadbeef"


@respx.mock
async def test_path_and_mtime_come_from_the_body() -> None:
    """The doc `_id` is lowercased by LiveSync; `path` preserves real case, and
    that distinction is what `sensitivity.personal_paths` matches against.
    """
    row = _live_row("tastings/laphroaig 10.md")
    row["doc"]["path"] = "Tastings/Laphroaig 10.md"
    _mock_listing([row])
    entries = await _vault().list_prefix("")
    assert entries[0].doc_id == "tastings/laphroaig 10.md"
    assert entries[0].path == "Tastings/Laphroaig 10.md"
    assert entries[0].mtime == 1787743028493


@respx.mock
async def test_entries_are_sorted_by_doc_id() -> None:
    _mock_listing([_live_row("10 raw/z.md"), _live_row("10 raw/a.md")])
    entries = await _vault().list_prefix("")
    assert [e.doc_id for e in entries] == ["10 raw/a.md", "10 raw/z.md"]


@respx.mock
async def test_rows_without_an_id_are_skipped() -> None:
    """`{"key": ..., "error": "not_found"}` — the real shape for an absent id."""
    _mock_listing([{"key": "nope", "error": "not_found"}, _live_row("10 raw/a.md")])
    entries = await _vault().list_prefix("")
    assert [e.doc_id for e in entries] == ["10 raw/a.md"]


class TestListPrefixFailures:
    @respx.mock
    async def test_non_200_raises(self) -> None:
        respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            return_value=httpx.Response(401, text="unauthorized")
        )
        with pytest.raises(VaultUnavailable):
            await _vault().list_prefix("")

    @respx.mock
    async def test_transport_error_raises(self) -> None:
        respx.get(url__startswith=f"{COUCH}/{DB}/_all_docs").mock(
            side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(VaultUnavailable):
            await _vault().list_prefix("")

    async def test_unset_url_raises(self) -> None:
        vault = LiveSyncVault(VaultConfig(couchdb_url=None, db=DB), None)
        with pytest.raises(VaultUnavailable):
            await vault.list_prefix("")


class TestRead:
    @respx.mock
    async def test_reassembles_chunks_in_order(self) -> None:
        respx.get(f"{COUCH}/{DB}/h%3Ata").mock(
            return_value=httpx.Response(200, json={"_id": "h:ta", "data": "first "})
        )
        respx.get(f"{COUCH}/{DB}/h%3Atb").mock(
            return_value=httpx.Response(200, json={"_id": "h:tb", "data": "second"})
        )
        _mock_listing([])
        from vault_ask.vault import Entry

        entry = Entry(doc_id="a.md", path="a.md", rev="1-x", children=("h:ta", "h:tb"), mtime=0)
        assert await _vault().read(entry) == "first second"

    @respx.mock
    async def test_missing_chunk_is_torn_not_partial(self) -> None:
        """Half a note is worse than no note — it would be cited as complete."""
        respx.get(f"{COUCH}/{DB}/h%3Ata").mock(
            return_value=httpx.Response(200, json={"_id": "h:ta", "data": "first "})
        )
        respx.get(f"{COUCH}/{DB}/h%3Atb").mock(return_value=httpx.Response(404))
        from vault_ask.vault import Entry

        entry = Entry(doc_id="a.md", path="a.md", rev="1-x", children=("h:ta", "h:tb"), mtime=0)
        assert await _vault().read(entry) is None

    async def test_no_children_is_torn_not_empty(self) -> None:
        """`"".join([])` is `""`, which would index as real, empty content.

        This is the failure mode that makes dropping `include_docs=true`
        catastrophic rather than merely wrong: without bodies, every row loses
        its `children`, so the entire vault would re-ingest as empty notes with
        `sha256("")` and silently blank the index.
        """
        from vault_ask.vault import Entry

        entry = Entry(doc_id="a.md", path="a.md", rev="1-x", children=(), mtime=0)
        assert await _vault().read(entry) is None
