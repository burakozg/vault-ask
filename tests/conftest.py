from __future__ import annotations

import os
from pathlib import Path

import pytest

import vault_ask.config as config_module
from vault_ask.api.admin_auth import reset_throttle


@pytest.fixture(autouse=True)
def _forget_failed_admin_auth() -> None:
    """The admin auth throttle counts failures per address, in module state.

    Left alone it would leak between tests: a file that exercises a few 401s
    would make a later test's 401 a 429, and which test broke would depend on
    the order they ran in.
    """
    reset_throttle()


@pytest.fixture(autouse=True)
def _isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test gets Settings() built from pure field defaults.

    Two separate leaks to close, not one:

    * `Settings()` would fall back to reading this repo's real config.yaml
      (module-global `_active_yaml_path` defaults to "config.yaml"), making
      tests depend on whatever a developer happens to have edited there.
      chdir to an empty tmp_path, plus repointing `_active_yaml_path` there
      too, closes this.
    * Likewise `_active_overrides_path` defaults to `/data/overrides.json` —
      harmless on a dev machine (the path just doesn't exist), but a test run
      as root inside a container with a real `/data` would read a real
      admin-console override into every test's Settings(). Repointed into
      tmp_path for the same reason as the yaml path.
    * `uv run` auto-loads a real `.env` from the project root as actual
      process environment variables *before Python even starts* — chdir does
      nothing about that, since they're already in `os.environ` by the time
      this fixture runs. A developer with real vault/OpenRouter credentials in
      `.env` (exactly the setup used to run this live against the real vault)
      would otherwise make every test's `Settings()` pick them up, which is
      how `embedding_base_url` in particular went from "unset" to "the real
      Ollama host" mid-session and broke every test asserting the unconfigured
      case. Stripped explicitly here rather than relying on chdir to prevent it.
    """
    monkeypatch.chdir(tmp_path)
    config_module._active_yaml_path = tmp_path / "does-not-exist.yaml"
    config_module._active_overrides_path = tmp_path / "does-not-exist-overrides.json"
    for key in list(os.environ):
        if key.startswith("VAULTASK_"):
            monkeypatch.delenv(key, raising=False)
