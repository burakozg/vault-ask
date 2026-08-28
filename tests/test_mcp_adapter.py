from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from vault_ask.api.app import create_app
from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig, Settings
from vault_ask.db import connect

CHUNK_CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    return Settings(index={"db_path": tmp_path / "index.sqlite"})


@pytest.fixture
def client(cfg: Settings) -> Iterator[TestClient]:
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def _seed(
    cfg: Settings, doc_id: str, path: str, title: str, sensitivity: str, body: str
) -> None:
    conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
    conn.execute(
        "INSERT INTO docs (doc_id, path, title, rev, content_hash, sensitivity, mtime, frontmatter) "
        "VALUES (?, ?, ?, 'rev-1', 'hash', ?, 0, '{}')",
        (doc_id, path, title, sensitivity),
    )
    replace_chunks(
        conn, doc_id=doc_id, path=path, title=title, sensitivity=sensitivity, markdown=body, cfg=CHUNK_CFG
    )
    conn.commit()
    conn.close()


def _seed_edge(cfg: Settings, src: str, dst: str, kind: str, resolved: int = 1) -> None:
    conn = connect(cfg.index.db_path, embedding_dim=cfg.models.embedding_dim)
    conn.execute(
        "INSERT INTO edges (src, dst, kind, resolved) VALUES (?, ?, ?, ?)", (src, dst, kind, resolved)
    )
    conn.commit()
    conn.close()


def _mcp_session(client: TestClient) -> dict[str, str]:
    resp = client.post(
        "/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {
                "protocolVersion": "2026-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test", "version": "0"},
            },
        },
        headers={"Accept": "application/json, text/event-stream"},
    )
    assert resp.status_code == 200, resp.text
    session_id = resp.headers["mcp-session-id"]
    return {"Accept": "application/json, text/event-stream", "mcp-session-id": session_id}


def _call_tool(client: TestClient, headers: dict[str, str], name: str, arguments: dict[str, Any]) -> Any:
    resp = client.post(
        "/mcp/",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    # SSE framing: "event: message\ndata: {...}\n\n" — take the JSON payload.
    line = next(line for line in resp.text.splitlines() if line.startswith("data: "))
    envelope = json.loads(line.removeprefix("data: "))
    result = envelope["result"]
    assert result["isError"] is False, result
    # The SDK wraps str/list returns as structuredContent={"result": ...} but
    # (this version, at least) leaves plain-dict returns unwrapped — only the
    # text content carries them, as a JSON string. Handle both rather than
    # assume one.
    structured = result.get("structuredContent")
    if structured is not None and "result" in structured:
        return structured["result"]
    text = result["content"][0]["text"]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


class TestToolsList:
    def test_all_four_tools_present(self, client: TestClient) -> None:
        headers = _mcp_session(client)
        resp = client.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=headers,
        )
        line = next(line for line in resp.text.splitlines() if line.startswith("data: "))
        tools = {t["name"] for t in json.loads(line.removeprefix("data: "))["result"]["tools"]}
        assert tools == {"vault_search", "vault_read", "vault_neighbors", "vault_topics"}


class TestVaultSearch:
    def test_finds_seeded_chunk(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/anthropic.md", "10 raw/Anthropic.md", "Anthropic", "open", "## H\n\nfounded in 2021")
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_search", {"query": "founded"})
        assert [r["path"] for r in result] == ["10 raw/Anthropic.md"]
        assert result[0]["citation"] == "[[10 raw/Anthropic.md|Anthropic]]"

    def test_allow_web_true_excludes_personal(self, client: TestClient, cfg: Settings) -> None:
        _seed(
            cfg, "30 journal/2026-08-28.md", "30 journal/2026-08-28.md", "Journal", "personal",
            "## H\n\nfounded a company today",
        )
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_search", {"query": "founded", "allow_web": True})
        assert result == []

    def test_k_limits_results(self, client: TestClient, cfg: Settings) -> None:
        for i in range(3):
            _seed(cfg, f"p{i}.md", f"p{i}.md", f"T{i}", "open", f"## H\n\nfounded in {2000 + i}")
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_search", {"query": "founded", "k": 1})
        assert len(result) == 1


class TestVaultRead:
    def test_reads_full_note_with_headings_restored(self, client: TestClient, cfg: Settings) -> None:
        # Each section padded well past the runt-merge threshold (80 tokens,
        # see chunk.py) — short sections would legitimately merge into one
        # chunk during indexing, which would make this test's own assumption
        # wrong, not vault_read.
        history = "founded in 2021. " + ("word " * 100)
        funding = "raised a lot. " + ("word " * 100)
        _seed(
            cfg, "10 raw/anthropic.md", "10 raw/Anthropic.md", "Anthropic", "open",
            f"## History\n\n{history}\n\n## Funding\n\n{funding}",
        )
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_read", {"path": "10 raw/Anthropic.md"})
        assert "## History" in result
        assert "## Funding" in result
        assert "founded in 2021" in result
        assert "[[10 raw/Anthropic.md|Anthropic]]" in result

    def test_not_found(self, client: TestClient) -> None:
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_read", {"path": "nope.md"})
        assert "No note found" in result

    def test_personal_note_indistinguishable_from_absent_when_allow_web_true(
        self, client: TestClient, cfg: Settings
    ) -> None:
        """The refusal must not confirm the note exists.

        The previous message named it ("<path> is a `personal` note ... —
        refusing") which withheld the body but handed a caller explicitly told
        it may not see the note both its existence and its exact real path.
        """
        _seed(
            cfg, "tastings/laphroaig 10.md", "Tastings/Laphroaig 10.md", "Laphroaig 10", "personal", "## H\n\ndear diary"
        )
        headers = _mcp_session(client)
        result = _call_tool(
            client, headers, "vault_read", {"path": "Tastings/Laphroaig 10.md", "allow_web": True}
        )
        assert "dear diary" not in result
        # Says nothing the caller did not already supply: no real path, no
        # title, no confirmation that anything is there.
        assert "No note found" in result
        assert "Laphroaig 10" not in result.replace("Tastings/Laphroaig 10.md", "")

        absent = _call_tool(
            client, headers, "vault_read", {"path": "Tastings/Laphroaig 10.md", "allow_web": True}
        )
        missing = _call_tool(
            client, headers, "vault_read", {"path": "nope.md", "allow_web": True}
        )
        assert absent.split("(")[0].replace("Tastings/Laphroaig 10.md", "X") == missing.split("(")[
            0
        ].replace("nope.md", "X")

    def test_personal_note_readable_when_allow_web_false(self, client: TestClient, cfg: Settings) -> None:
        _seed(
            cfg, "tastings/laphroaig 10.md", "Tastings/Laphroaig 10.md", "Laphroaig 10", "personal", "## H\n\ndear diary"
        )
        headers = _mcp_session(client)
        result = _call_tool(
            client, headers, "vault_read", {"path": "Tastings/Laphroaig 10.md", "allow_web": False}
        )
        assert "dear diary" in result

    def test_resolves_by_title(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/anthropic.md", "10 raw/Anthropic.md", "Anthropic", "open", "## H\n\nbody")
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_read", {"path": "Anthropic"})
        assert "body" in result


class TestVaultNeighbors:
    def test_outbound_and_inbound(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/a.md", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(cfg, "10 raw/b.md", "10 raw/B.md", "B", "open", "## H\n\nbody")
        _seed_edge(cfg, "10 raw/a.md", "10 raw/b.md", "wikilink")

        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_neighbors", {"path": "10 raw/A.md"})
        assert result["outbound"] == [{"kind": "wikilink", "path": "10 raw/B.md", "title": "B"}]

        result_b = _call_tool(client, headers, "vault_neighbors", {"path": "10 raw/B.md"})
        assert result_b["inbound"] == [{"kind": "wikilink", "path": "10 raw/A.md", "title": "A"}]

    def test_tag_edges(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/a.md", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed_edge(cfg, "10 raw/a.md", "tag:ai", "tag")
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_neighbors", {"path": "10 raw/A.md"})
        assert result["outbound"] == [{"kind": "tag", "value": "ai"}]

    def test_unresolved_wikilink_kept_as_target_string(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/a.md", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed_edge(cfg, "10 raw/a.md", "Nowhere", "wikilink", resolved=0)
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_neighbors", {"path": "10 raw/A.md"})
        assert result["outbound"] == [{"kind": "wikilink", "unresolved_target": "Nowhere"}]

    def test_allow_web_excludes_personal_neighbor(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/a.md", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(cfg, "40 people/priv.md", "40 people/Priv.md", "Priv", "personal", "## H\n\nbody")
        _seed_edge(cfg, "10 raw/a.md", "40 people/priv.md", "wikilink")
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_neighbors", {"path": "10 raw/A.md", "allow_web": True})
        assert result["outbound"] == []

    def test_not_found(self, client: TestClient) -> None:
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_neighbors", {"path": "nope.md"})
        assert "error" in result


class TestVaultTopics:
    def test_lists_topic_notes_by_member_count(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "99 topics/ai.md", "99 topics/ai.md", "AI", "open", "## H\n\nbody")
        _seed(cfg, "10 raw/a.md", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed(cfg, "10 raw/b.md", "10 raw/B.md", "B", "open", "## H\n\nbody")
        _seed_edge(cfg, "99 topics/ai.md", "10 raw/a.md", "topic")
        _seed_edge(cfg, "99 topics/ai.md", "10 raw/b.md", "topic")

        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_topics", {})
        assert result == [{"path": "99 topics/ai.md", "title": "AI", "member_count": 2}]

    def test_not_hardcoded_to_99_topics_folder(self, client: TestClient, cfg: Settings) -> None:
        _seed(cfg, "10 raw/hub.md", "10 raw/Hub.md", "Hub", "open", "## H\n\nbody")
        _seed(cfg, "10 raw/a.md", "10 raw/A.md", "A", "open", "## H\n\nbody")
        _seed_edge(cfg, "10 raw/hub.md", "10 raw/a.md", "topic")
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_topics", {})
        assert [r["path"] for r in result] == ["10 raw/Hub.md"]

    def test_no_topics_is_empty(self, client: TestClient) -> None:
        headers = _mcp_session(client)
        result = _call_tool(client, headers, "vault_topics", {})
        assert result == []
