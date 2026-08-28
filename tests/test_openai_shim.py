from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import litellm
import pytest
from fastapi.testclient import TestClient

from vault_ask.api.app import create_app
from vault_ask.chunk import replace_chunks
from vault_ask.config import ChunkingConfig, Settings
from vault_ask.db import connect

CHUNK_CFG = ChunkingConfig(target_tokens=1000, hard_split_tokens=2000)


class _FakeCompletion:
    """Answers both plain and `stream=True` litellm.acompletion calls."""

    def __init__(self, text: str) -> None:
        self.calls: list[dict[str, Any]] = []
        self._text = text

    async def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._stream()
        message = SimpleNamespace(content=self._text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    async def _stream(self) -> AsyncIterator[Any]:
        for word in self._text.split(" "):
            delta = SimpleNamespace(content=word + " ")
            yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    # vault.couchdb_url left unset: create_app's background index loop logs a
    # warning and returns immediately (see api/app.py::_index_loop) rather
    # than trying to reach a real vault during a test.
    return Settings(index={"db_path": tmp_path / "index.sqlite"})


@pytest.fixture
def client(cfg: Settings) -> Iterator[TestClient]:
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def _seed(cfg: Settings, doc_id: str, path: str, title: str, sensitivity: str, body: str) -> None:
    """Writes through a *separate* connection to the same (WAL-mode) db file.

    `TestClient` runs the app's lifespan on a different thread than the test
    body, and sqlite3 connections are thread-bound — `app.state.conn` cannot
    be touched from here directly. Opening our own connection to the same
    file and relying on WAL to make the write visible is simpler than
    threading a cross-thread handoff through the fixture.
    """
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


class TestHealthz:
    def test_ok(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModels:
    def test_lists_vault_ask(self, client: TestClient) -> None:
        response = client.get("/v1/models")
        assert response.status_code == 200
        body = response.json()
        assert [m["id"] for m in body["data"]] == ["vault-ask"]


class TestChatCompletionsNonStreaming:
    def test_answers_from_the_vault(
        self, client: TestClient, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(cfg, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        fake = _FakeCompletion("Anthropic was founded in 2021 [[10 raw/Anthropic.md|Anthropic]].")
        monkeypatch.setattr(litellm, "acompletion", fake)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "vault-ask", "messages": [{"role": "user", "content": "founded"}]},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["object"] == "chat.completion"
        assert body["model"] == "vault-ask"
        assert body["choices"][0]["message"]["content"] == fake._text
        assert fake.calls[0].get("stream") is None or fake.calls[0]["stream"] is False

    def test_uses_the_last_user_message(
        self, client: TestClient, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(cfg, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        fake = _FakeCompletion("ok")
        monkeypatch.setattr(litellm, "acompletion", fake)

        client.post(
            "/v1/chat/completions",
            json={
                "model": "vault-ask",
                "messages": [
                    {"role": "user", "content": "irrelevant first turn"},
                    {"role": "assistant", "content": "some reply"},
                    {"role": "user", "content": "founded"},
                ],
            },
        )
        assert "founded in 2021" in fake.calls[0]["messages"][1]["content"]

    def test_no_user_message_is_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat/completions",
            json={"model": "vault-ask", "messages": [{"role": "system", "content": "hi"}]},
        )
        assert response.status_code == 400

    def test_allow_web_defaults_false_sees_personal_chunks(
        self, client: TestClient, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(
            cfg, "d1", "30 journal/2026-08-28.md", "Journal", "personal", "## Entry\n\nfounded a company today"
        )
        fake = _FakeCompletion("You founded a company today.")
        monkeypatch.setattr(litellm, "acompletion", fake)

        response = client.post(
            "/v1/chat/completions",
            json={"model": "vault-ask", "messages": [{"role": "user", "content": "founded"}]},
        )
        assert response.json()["choices"][0]["message"]["content"] == "You founded a company today."

    def test_allow_web_true_excludes_personal_chunks(
        self, client: TestClient, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(
            cfg, "d1", "30 journal/2026-08-28.md", "Journal", "personal", "## Entry\n\nfounded a company today"
        )
        fake = _FakeCompletion("should never be reached")
        monkeypatch.setattr(litellm, "acompletion", fake)

        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "vault-ask",
                "messages": [{"role": "user", "content": "founded"}],
                "allow_web": True,
            },
        )
        # No `open` chunk matches, so the vault-silent answer short-circuits
        # before the model is ever called.
        assert response.json()["choices"][0]["message"]["content"] == "The vault is silent on this."
        assert fake.calls == []


def _sse_content(lines: list[str]) -> str:
    """The concatenated `delta.content` across a stream's chunks.

    Each SSE line is a *separate* JSON envelope (id/object/created/model/...),
    so naively concatenating the raw lines interleaves that boilerplate with
    the text — joining the parsed `delta.content` values is what actually
    reconstructs the streamed answer.
    """
    text = ""
    for line in lines:
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        payload = json.loads(line.removeprefix("data: "))
        text += payload["choices"][0]["delta"].get("content", "")
    return text


class TestChatCompletionsStreaming:
    def test_streams_sse_chunks_ending_in_done(
        self, client: TestClient, cfg: Settings, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _seed(cfg, "d1", "10 raw/Anthropic.md", "Anthropic", "open", "## History\n\nfounded in 2021")
        fake = _FakeCompletion("Anthropic was founded in 2021.")
        monkeypatch.setattr(litellm, "acompletion", fake)

        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "vault-ask",
                "messages": [{"role": "user", "content": "founded"}],
                "stream": True,
            },
        ) as response:
            assert response.status_code == 200
            lines = [line for line in response.iter_lines() if line]

        assert lines[-1] == "data: [DONE]"
        assert lines[0].startswith("data: ")
        assert '"role": "assistant"' in lines[0]
        # _FakeCompletion._stream splits on " " and re-appends it to every
        # word including the last, so the reconstructed text carries one
        # trailing space the source sentence didn't have.
        assert _sse_content(lines).strip() == "Anthropic was founded in 2021."
        assert fake.calls[0]["stream"] is True

    def test_streaming_vault_silent_is_one_chunk(self, client: TestClient) -> None:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "vault-ask",
                "messages": [{"role": "user", "content": "nothing matches this"}],
                "stream": True,
            },
        ) as response:
            lines = [line for line in response.iter_lines() if line]
        assert _sse_content(lines) == "The vault is silent on this."
        assert lines[-1] == "data: [DONE]"


class TestChatPage:
    """The chat UI vault-ask serves itself, replacing Open WebUI.

    Open WebUI cost a 5.09 GB image and ~15m35s of cold Python imports per
    recreate for a RAG stack that was never used here — vault-ask is the
    retrieval layer. What it actually supplied was a chat box.
    """

    def test_served_and_self_contained(self, client: TestClient) -> None:
        resp = client.get("/chat")
        assert resp.status_code == 200
        body = resp.text
        assert "Ask the vault" in body
        # Same constraint as admin.html: no external requests. A CDN reference
        # would make a LAN-only service depend on the internet to render.
        assert "https://" not in body.replace("https://github.com", "")
        assert "<script src=" not in body
        assert "<link" not in body

    def test_root_redirects_to_chat(self, client: TestClient) -> None:
        """A bare visit used to 404 — the first thing anyone tries."""
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (307, 308)
        assert resp.headers["location"] == "/chat"

    def test_posts_the_inverse_of_the_switch(self, client: TestClient) -> None:
        """`allow_web` is the inverse of "include personal notes".

        Pinned because the two names mean opposite things: allow_web=true
        *narrows* retrieval to `open` chunks. A UI that got this backwards
        would leak personal content into a web-capable context, silently.
        """
        body = client.get("/chat").text
        assert "allow_web: !personal" in body

    def test_healthz_and_admin_still_routed(self, client: TestClient) -> None:
        """Adding a catch-all-looking "/" route must not shadow anything —
        mounting at "/" once swallowed every other route on this app."""
        assert client.get("/healthz").status_code == 200
        assert client.get("/admin").status_code == 200
        assert client.get("/v1/models").status_code == 200
