from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from vault_ask.api.app import create_app
from vault_ask.config import Settings

ADMIN_KEY = "test-admin-key"


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    return Settings(
        index={"db_path": tmp_path / "index.sqlite"},
        admin_api_key=ADMIN_KEY,
    )


@pytest.fixture
def client(cfg: Settings) -> Iterator[TestClient]:
    app = create_app(cfg)
    with TestClient(app) as c:
        yield c


def _auth(key: str = ADMIN_KEY) -> dict[str, str]:
    return {"X-API-Key": key}


class TestAdminPage:
    def test_served_unauthenticated(self, client: TestClient) -> None:
        resp = client.get("/admin")
        assert resp.status_code == 200
        assert "vault-ask admin" in resp.text


class TestAuth:
    def test_no_key_configured_is_503(self, tmp_path: Path) -> None:
        cfg = Settings(index={"db_path": tmp_path / "index.sqlite"})
        with TestClient(create_app(cfg)) as c:
            resp = c.get("/admin/config", headers=_auth())
        assert resp.status_code == 503

    def test_missing_key_is_401(self, client: TestClient) -> None:
        resp = client.get("/admin/config")
        assert resp.status_code == 401

    def test_wrong_key_is_401(self, client: TestClient) -> None:
        resp = client.get("/admin/config", headers=_auth("nope"))
        assert resp.status_code == 401

    def test_repeated_failures_are_throttled(self, client: TestClient) -> None:
        for _ in range(10):
            client.get("/admin/config", headers=_auth("nope"))
        resp = client.get("/admin/config", headers=_auth("nope"))
        assert resp.status_code == 429

    def test_correct_key_after_failures_clears_throttle(self, client: TestClient) -> None:
        for _ in range(5):
            client.get("/admin/config", headers=_auth("nope"))
        resp = client.get("/admin/config", headers=_auth())
        assert resp.status_code == 200


class TestReadConfig:
    def test_defaults_have_no_overrides(self, client: TestClient) -> None:
        resp = client.get("/admin/config", headers=_auth())
        assert resp.status_code == 200
        body = resp.json()
        assert body["models"]["generation"]["override"] is None
        assert body["models"]["generation"]["active"] == "openrouter/google/gemini-2.5-flash"
        assert body["retrieval"]["final_top_k"] == {"active": 8, "override": None}
        assert body["pending_restart"] is False
        assert body["editable_keys"]["models"] == ["generation"]
        assert "rerank_top_k" not in body["retrieval"]


class TestWriteConfig:
    def test_valid_generation_override(self, client: TestClient) -> None:
        resp = client.put(
            "/admin/config", headers=_auth(), json={"models": {"generation": "openrouter/x/y"}}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["pending_restart"] is True

        again = client.get("/admin/config", headers=_auth()).json()
        assert again["models"]["generation"]["override"] == "openrouter/x/y"
        # Active is unchanged: overrides apply on restart, not live.
        assert again["models"]["generation"]["active"] == "openrouter/google/gemini-2.5-flash"
        assert again["pending_restart"] is True

    def test_valid_retrieval_override(self, client: TestClient) -> None:
        resp = client.put(
            "/admin/config", headers=_auth(), json={"retrieval": {"final_top_k": 12}}
        )
        assert resp.status_code == 200, resp.text
        again = client.get("/admin/config", headers=_auth()).json()
        assert again["retrieval"]["final_top_k"]["override"] == 12

    def test_rerank_top_k_not_overridable(self, client: TestClient) -> None:
        resp = client.put(
            "/admin/config", headers=_auth(), json={"retrieval": {"rerank_top_k": 40}}
        )
        assert resp.status_code == 400
        assert "rerank_top_k" in resp.json()["detail"]

    def test_cross_field_validation_uses_active_rerank_top_k(self, client: TestClient) -> None:
        # active rerank_top_k is 30 (file default); final_top_k > that violates
        # RetrievalConfig's rerank_top_k >= final_top_k invariant.
        resp = client.put(
            "/admin/config", headers=_auth(), json={"retrieval": {"final_top_k": 40}}
        )
        assert resp.status_code == 400
        assert "rerank_top_k" in resp.json()["detail"]

    def test_unauthenticated_write_rejected(self, client: TestClient) -> None:
        resp = client.put("/admin/config", json={"models": {"generation": "x"}})
        assert resp.status_code == 401


class TestResetConfig:
    def test_clears_stored_overrides(self, client: TestClient) -> None:
        client.put("/admin/config", headers=_auth(), json={"models": {"generation": "openrouter/x/y"}})
        resp = client.delete("/admin/config", headers=_auth())
        assert resp.status_code == 200, resp.text

        again = client.get("/admin/config", headers=_auth()).json()
        assert again["models"]["generation"]["override"] is None

    def test_pending_restart_false_when_matches_boot_snapshot(self, client: TestClient) -> None:
        # Nothing was ever written, so resetting (to {}) matches what this
        # process booted with ({}) — no restart needed to reach that state.
        resp = client.delete("/admin/config", headers=_auth())
        assert resp.json()["pending_restart"] is False


class TestOverridesFileHardening:
    """The allowlist has to hold for a hand-edited file, not only for what the
    admin API writes — overrides.json is bind-mounted next to the index and is
    the obvious thing to reach for when the console is unreachable.
    """

    def test_non_overridable_section_is_dropped_at_load(self, tmp_path: Path) -> None:
        from vault_ask.overrides import read_overrides

        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"vault": {"couchdb_url": "http://evil.example"}}))
        assert read_overrides(path) == {}

    def test_non_overridable_key_is_dropped_at_load(self, tmp_path: Path) -> None:
        from vault_ask.overrides import read_overrides

        path = tmp_path / "overrides.json"
        path.write_text(
            json.dumps({"models": {"generation": "openrouter/x/y", "embedding_base_url": "http://evil"}})
        )
        assert read_overrides(path) == {"models": {"generation": "openrouter/x/y"}}

    def test_hand_edited_topology_cannot_reach_settings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The end-to-end version: a malicious/mistaken file must not move the
        vault connection when Settings is actually built."""
        import vault_ask.config as config_module

        path = tmp_path / "overrides.json"
        path.write_text(json.dumps({"vault": {"couchdb_url": "http://evil.example:5984"}}))
        monkeypatch.setattr(config_module, "_active_overrides_path", path)
        assert Settings().vault.couchdb_url is None

    def test_malformed_json_does_not_prevent_startup(self, tmp_path: Path) -> None:
        """Raising here would mean a hand-edit typo bricks the process."""
        from vault_ask.overrides import read_overrides

        path = tmp_path / "overrides.json"
        path.write_text("{not json at all")
        assert read_overrides(path) == {}

    def test_non_object_top_level_is_ignored(self, tmp_path: Path) -> None:
        from vault_ask.overrides import read_overrides

        path = tmp_path / "overrides.json"
        path.write_text("[1, 2, 3]")
        assert read_overrides(path) == {}

    def test_malformed_file_does_not_500_the_admin_api(
        self, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vault_ask.config as config_module

        path = tmp_path / "overrides.json"
        path.write_text("{broken")
        monkeypatch.setattr(config_module, "_active_overrides_path", path)
        with TestClient(create_app(cfg)) as c:
            resp = c.get("/admin/config", headers=_auth())
        assert resp.status_code == 200


class TestGenerationValidation:
    def test_empty_generation_is_rejected(self, client: TestClient) -> None:
        """Previously saved fine and failed at the next restart's first call."""
        resp = client.put("/admin/config", headers=_auth(), json={"models": {"generation": ""}})
        assert resp.status_code == 400

    def test_graph_enabled_is_overridable(self, client: TestClient) -> None:
        resp = client.put(
            "/admin/config", headers=_auth(), json={"retrieval": {"graph_enabled": False}}
        )
        assert resp.status_code == 200, resp.text
        again = client.get("/admin/config", headers=_auth()).json()
        assert again["retrieval"]["graph_enabled"]["override"] is False


class TestWebSection:
    """The console can turn the web fallback off without a redeploy — it is
    the one component that talks to the outside world, so the kill switch
    should not require shipping an image."""

    def test_web_section_is_readable(self, client: TestClient) -> None:
        body = client.get("/admin/config", headers=_auth()).json()
        assert body["web"]["enabled"] == {"active": False, "override": None}
        assert body["editable_keys"]["web"] == ["enabled", "max_results", "provider"]

    def test_can_be_toggled(self, client: TestClient) -> None:
        resp = client.put("/admin/config", headers=_auth(), json={"web": {"enabled": True}})
        assert resp.status_code == 200, resp.text
        again = client.get("/admin/config", headers=_auth()).json()
        assert again["web"]["enabled"]["override"] is True

    def test_thresholds_are_not_console_editable(self, client: TestClient) -> None:
        """thin_distance was measured; a browser is the wrong place to nudge
        the number that decides when the vault stops being the only source."""
        resp = client.put("/admin/config", headers=_auth(), json={"web": {"thin_distance": 5.0}})
        assert resp.status_code == 400
        assert "thin_distance" in resp.json()["detail"]


class TestProviderChoices:
    """The console offers real choices, and refuses one it knows cannot work."""

    def test_reports_every_provider_with_availability(self, client: TestClient) -> None:
        body = client.get("/admin/config", headers=_auth()).json()
        by_name = {p["name"]: p for p in body["web_providers"]}
        assert set(by_name) == {"duckduckgo", "tavily", "brave"}
        assert by_name["duckduckgo"]["available"] is True
        assert by_name["duckduckgo"]["needs_key"] is False
        # No key configured in the test settings.
        assert by_name["tavily"]["available"] is False
        assert by_name["tavily"]["env_var"] == "VAULTASK_TAVILY_API_KEY"

    def test_offers_model_suggestions(self, client: TestClient) -> None:
        body = client.get("/admin/config", headers=_auth()).json()
        suggestions = body["generation_suggestions"]
        assert len(suggestions) > 3
        assert body["models"]["generation"]["active"] in suggestions
        assert all(s.startswith("openrouter/") for s in suggestions)

    def test_selecting_a_keyless_provider_is_rejected(self, client: TestClient) -> None:
        """Saving it would look fine and then silently return no web results
        after the next restart — the failure would point nowhere."""
        resp = client.put("/admin/config", headers=_auth(), json={"web": {"provider": "tavily"}})
        assert resp.status_code == 400
        detail = resp.json()["detail"]
        assert "VAULTASK_TAVILY_API_KEY" in detail

    def test_selecting_a_keyed_provider_works_once_the_key_exists(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import vault_ask.config as config_module

        monkeypatch.setattr(config_module, "_active_overrides_path", tmp_path / "o.json")
        cfg = Settings(
            index={"db_path": tmp_path / "index.sqlite"},
            admin_api_key=ADMIN_KEY,
            tavily_api_key="tvly-x",
        )
        with TestClient(create_app(cfg)) as c:
            resp = c.put("/admin/config", headers=_auth(), json={"web": {"provider": "tavily"}})
            assert resp.status_code == 200, resp.text
            again = c.get("/admin/config", headers=_auth()).json()
            assert again["web"]["provider"]["override"] == "tavily"

    def test_unknown_provider_is_rejected(self, client: TestClient) -> None:
        resp = client.put("/admin/config", headers=_auth(), json={"web": {"provider": "altavista"}})
        assert resp.status_code == 400
