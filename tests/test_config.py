from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vault_ask.config import ChunkingConfig, RetrievalConfig, Settings, load_settings


class TestShippedConfig:
    def test_repo_config_yaml_is_valid(self) -> None:
        """The shipped config.yaml must load — it is the deployment default."""
        settings = load_settings(Path(__file__).parent.parent / "config.yaml")
        assert settings.corpus.include == ["**/*.md"]
        assert settings.vault.db == "the_brain"
        assert settings.retrieval.final_top_k == 8

    def test_unknown_key_is_a_startup_crash(self, tmp_path: Path) -> None:
        bad = tmp_path / "config.yaml"
        bad.write_text("corpus:\n  includ: ['**/*.md']\n")  # typo'd key
        with pytest.raises(ValidationError):
            load_settings(bad)


class TestVaultCredentials:
    def test_url_without_password_refuses_to_boot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VAULTASK_VAULT_COUCHDB_PASSWORD", raising=False)
        with pytest.raises(ValidationError):
            Settings(vault={"couchdb_url": "http://couchdb.local:5984"})

    def test_url_with_password_boots(self) -> None:
        settings = Settings(
            vault={"couchdb_url": "http://couchdb.local:5984"},
            vault_couchdb_password="secret",
        )
        assert settings.vault.couchdb_url == "http://couchdb.local:5984"

    def test_no_url_needs_no_password(self) -> None:
        settings = Settings()
        assert settings.vault.couchdb_url is None


class TestValidators:
    def test_chunking_target_over_hard_split_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChunkingConfig(target_tokens=3000, hard_split_tokens=2000)

    def test_retrieval_fusion_wider_than_inputs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(vector_top_k=10, fts_top_k=10, fusion_top_k=30)

    def test_retrieval_rerank_narrower_than_final_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RetrievalConfig(rerank_top_k=5, final_top_k=8)
