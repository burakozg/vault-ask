"""Layered configuration: config.yaml (non-secret) + environment (secrets).

Mirrors podcast-digest's `config.py`: the whole design is modelled up front as
strict pydantic settings, even sections not yet wired to code, so a later step
is a matter of *reading* an already-validated field rather than inventing one
under time pressure. Invalid configuration is a startup crash, never a
half-configured run — every model uses ``extra="forbid"`` so a typo'd key
fails loudly instead of being silently ignored.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

from .overrides import read_overrides

DEFAULT_CONFIG_FILE = "config.yaml"
DEFAULT_OVERRIDES_FILE = "/data/overrides.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_http_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"must be an http(s) URL, got {value!r}")
    if not parsed.netloc:
        raise ValueError(f"missing host in URL {value!r}")
    return value


class CorpusConfig(StrictModel):
    """What counts as the corpus, before any note is read."""

    include: list[str] = Field(default_factory=lambda: ["**/*.md"])
    #: fnmatch-style, evaluated against the vault-relative path. `*` already
    #: crosses `/` under `fnmatch` (it is not a path-aware glob), so `**` reads
    #: as intent rather than doing anything a single `*` would not.
    exclude: list[str] = Field(
        default_factory=lambda: ["00 inbox/**", "**/.trash/**"]
    )


class SensitivityConfig(StrictModel):
    """Classifies every chunk as `open` or `personal` — config, not inference.

    See vault_ask.frontmatter.classify_sensitivity for the rule this feeds:
    the per-note frontmatter override wins over the path match.

    Patterns are matched **case-sensitively** (fnmatchcase) against the note's
    real `path`, not its lowercased CouchDB `_id`. A pattern naming a folder
    that does not exist silently classifies nothing — which is how this shipped
    guarding an empty set — so `ingest.run_ingest` warns on any pattern that
    matches zero candidates.
    """

    personal_paths: list[str] = Field(
        default_factory=lambda: ["Tastings/**", "30 projects/**"]
    )
    frontmatter_key: str = "sensitivity"


class VaultConfig(StrictModel):
    """The vault's CouchDB — the same one Self-hosted LiveSync replicates against.

    Read-only: this application opens no write path into the vault (see
    vault_ask.vault.LiveSyncVault, which is a read-only subset of the
    projecting client the other four apps use).
    """

    couchdb_url: str | None = None
    db: str = Field(default="the_brain", pattern=r"^[a-z][a-z0-9_$()+/-]*$")
    user: str = "vaultask"
    timeout_s: float = Field(default=30.0, ge=1.0, le=300.0)

    @field_validator("couchdb_url")
    @classmethod
    def _check_url(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        return _require_http_url(value)


class ModelsConfig(StrictModel):
    """Everything through litellm — nothing else in the codebase knows a provider name."""

    # Bounded because the admin console can set it: an empty string saved
    # from a browser validates fine and then fails at the next restart's first
    # generation call, which is a long way from the edit that caused it.
    generation: str = Field(
        default="openrouter/google/gemini-2.5-flash", min_length=1, max_length=200
    )
    rerank: str = "openrouter/google/gemini-2.5-flash-lite"
    # Local, always — embedding is the one call that touches every note in the
    # vault, including the `personal` ones, on every --rebuild (see README
    # "Models"). litellm's ollama embedding route wants the bare "ollama/"
    # prefix (not "ollama_chat/", which is only for chat completions), hitting
    # /api/embeddings on the host named by embedding_base_url.
    embedding: str = "ollama/bge-m3"
    # Deployment topology, like podcast-digest's asr.remote_url — a LAN machine
    # running Ollama, not this container. No default: startup refuses to boot
    # rather than run indexing against a host nobody chose. Set via
    # VAULTASK_MODELS__EMBEDDING_BASE_URL. See OLLAMA-SETUP.md for getting a
    # machine to point this at.
    embedding_base_url: str | None = None
    embedding_dim: int = Field(default=1024, ge=1, le=8192)

    @field_validator("embedding_base_url")
    @classmethod
    def _check_base_url(cls, value: str | None) -> str | None:
        return None if value is None else _require_http_url(value)

    # Deliberately NOT a Settings-level validator: config is shared by every
    # subcommand, including ones that never embed anything (index's own
    # change-detection pass, `ask` once an index already exists). Checked at
    # the point of use instead, the same way main.py checks vault.couchdb_url
    # only inside the `index` command — see the embedding step (build order
    # step 3).


#: Suggestions for `models.generation`, offered by the admin console as a
#: type-ahead. NOT a validation allowlist — the field stays free text, because
#: any litellm-routable id is legitimate and a closed list would be wrong the
#: week it shipped.
#:
#: Every id here was checked against OpenRouter's live catalogue rather than
#: recalled, which is the failure mode a hand-written list invites. It will
#: still go stale: re-check with
#:   curl -s https://openrouter.ai/api/v1/models | jq -r '.data[].id'
#: A live fetch was the alternative and was deliberately not taken — it puts a
#: network call and an outage mode into a page whose job is to work when
#: things are broken.
GENERATION_SUGGESTIONS: tuple[str, ...] = (
    # Fast and cheap, with a context window big enough that retrieval width is
    # never the constraint. The shipped default.
    "openrouter/google/gemini-2.5-flash",
    "openrouter/google/gemini-2.5-flash-lite",
    "openrouter/google/gemini-2.5-pro",
    "openrouter/anthropic/claude-haiku-4.5",
    "openrouter/anthropic/claude-fable-5",
    "openrouter/deepseek/deepseek-chat-v3.1",
    "openrouter/meta-llama/llama-3.3-70b-instruct",
    "openrouter/mistralai/mistral-medium-3-5",
    "openrouter/qwen/qwen-plus",
)


class ChunkingConfig(StrictModel):
    """Heading-aware chunking targets, in approximate tokens."""

    target_tokens: int = Field(default=1000, ge=100, le=8000)
    hard_split_tokens: int = Field(default=2000, ge=100, le=16000)

    @model_validator(mode="after")
    def _target_below_hard_split(self) -> ChunkingConfig:
        if self.target_tokens > self.hard_split_tokens:
            raise ValueError("chunking.target_tokens must be <= chunking.hard_split_tokens")
        return self


class RetrievalConfig(StrictModel):
    """Retrieval pipeline widths — see README "Retrieval" for the shape this drives."""

    vector_top_k: int = Field(default=40, ge=1, le=500)
    fts_top_k: int = Field(default=40, ge=1, le=500)
    fusion_top_k: int = Field(default=20, ge=1, le=200)
    #: A real off switch. `graph_max_siblings=0` looks like one and is not:
    #: the topic-note pull in `expand_graph` happens before, and independently
    #: of, the sibling query, so zeroing siblings still injects every topic
    #: note — exactly the docs README "Open questions" suspects of crowding out
    #: primaries. Needed to A/B expansion at all (README "Build order", step 5:
    #: still unmeasured).
    graph_enabled: bool = True
    #: One hop only — see README: the reason the graph hop does not swamp
    #: results is that it is bounded both in depth and in per-doc fan-out.
    graph_max_siblings: int = Field(default=5, ge=0, le=50)
    #: Expanded chunks enter at a fixed discount off the hit that reached them.
    #:
    #: This alone does NOT keep them from crowding out direct hits, despite
    #: what the README used to claim. With `_RRF_K = 60`, a hit ranked 1 in
    #: both arms scores 2/61 = 0.0328, so the chunk it pulls in scores 0.0230 —
    #: above a rank-1 *single-arm* hit at 1/61 = 0.0164. Measured over 20 real
    #: questions: 36 expanded chunks entered final answers and displaced 36
    #: direct hits, 1:1, taking up to 6 of 8 slots. See `graph_max_slots`,
    #: which is what actually bounds it, and
    #: `scripts/measure_graph_expansion.py`.
    graph_discount: float = Field(default=0.7, ge=0.0, le=1.0)
    #: How many of `final_top_k` an expanded chunk may occupy.
    #:
    #: Expansion is worth keeping but not worth trusting with the whole answer.
    #: The same measurement showed it genuinely helps on broad thematic
    #: questions ("what have I read about AI agents?" — it surfaced the topic
    #: note and four on-topic notes, displacing a tangential security survey)
    #: and genuinely hurts on specific ones ("Claude and Anthropic" — it evicted
    #: three digest items actually about Anthropic for a GPT-5.5 evaluation).
    #: A quota keeps the upside and caps the downside, which score tuning
    #: cannot: every expanded chunk inherits the *same* score from the hit that
    #: reached it, so they tie exactly and their order among themselves is
    #: arbitrary — there is no ranking signal in there to tune.
    graph_max_slots: int = Field(default=2, ge=0, le=50)
    rerank_top_k: int = Field(default=30, ge=1, le=200)
    final_top_k: int = Field(default=8, ge=1, le=50)

    @model_validator(mode="after")
    def _widths_narrow(self) -> RetrievalConfig:
        if self.fusion_top_k > self.vector_top_k + self.fts_top_k:
            raise ValueError("retrieval.fusion_top_k cannot exceed vector_top_k + fts_top_k")
        if self.rerank_top_k < self.final_top_k:
            raise ValueError("retrieval.rerank_top_k must be >= retrieval.final_top_k")
        return self


class WebConfig(StrictModel):
    """Web fallback — see vault_ask/web.py for the invariants this feeds.

    Off by default. This is the one component that can send anything outside
    the house, so it is opt-in rather than opt-out: an operator who has not
    thought about it gets a vault-only system, which is the safe reading of
    "the vault is the authoritative source".
    """

    enabled: bool = False
    #: Which search backend. Changeable from the admin console; its API key is
    #: not — keys are secrets, and secrets here are environment-only (see
    #: vault_ask/overrides.py). `duckduckgo` needs none and is the default so
    #: the feature works with nothing to sign up for; the other two need a key
    #: and the console refuses to select an unusable one.
    provider: Literal["duckduckgo", "tavily", "brave"] = "duckduckgo"
    #: Deliberately small. Web material is a supplement to the vault, and a
    #: long list of snippets is exactly how it stops being one — the model has
    #: more web text than vault text to work with and the answer tilts.
    max_results: int = Field(default=3, ge=1, le=10)
    region: str = "wt-wt"  # no regional weighting
    #: 25s, from measurement rather than taste: from the NAS container a
    #: search took 20.2s cold and 8.4s warm, so the original 8.0 default
    #: timed out on every first query and silently degraded every answer to
    #: vault-only. Generous because of what the alternative is — this only
    #: runs when the vault is already thin, so the choice is a slow useful
    #: answer or a fast useless one.
    timeout_s: float = Field(default=25.0, ge=1.0, le=120.0)

    # --- when to search: "the vault's own answer looks thin" ----------------
    #
    # Not "when the vault is silent": with ~2,250 notes, FTS returns something
    # for almost any question, so a silence trigger would essentially never
    # fire. Thin coverage is the honest reading of "the vault did not really
    # answer this".
    #
    # Both thresholds are guesses until measured — see
    # scripts/measure_web_trigger.py, the same treatment graph expansion got.
    #: Search when the vault returned fewer than this many hits.
    thin_hits: int = Field(default=3, ge=0, le=50)
    #: Search when the nearest chunk is further than this from the question in
    #: embedding space. Higher = further = worse match.
    #:
    #: Deliberately NOT a threshold on the fused RRF score, which was the first
    #: attempt and is unworkable: RRF is rank-based, so the top hit scores
    #: 1/61 = 0.0164 whether it is a perfect match or nonsense. Measured, a
    #: question the vault cannot answer at all ("airspeed velocity of an
    #: unladen swallow") scored 0.0164 — indistinguishable from a good answer.
    #: The fused score carries no relevance information; vector distance does.
    #:
    #: 1.0 sits in a clean gap measured over 8 questions on this vault
    #: (bge-m3, L2): covered 0.815-0.955, not covered 1.022-1.129. Eight
    #: questions is a small sample — re-run scripts/measure_web_trigger.py
    #: after a big corpus change or an embedding model swap.
    thin_distance: float = Field(default=1.0, ge=0.0, le=10.0)


class IndexConfig(StrictModel):
    db_path: Path = Path("/data/index.sqlite")
    run_on_startup: bool = True
    #: Cheap by design (README "Deployment"): an hour of no vault edits costs
    #: one _all_docs listing and no LLM calls.
    refresh_interval_s: int = Field(default=3600, ge=60, le=86400)


class APIConfig(StrictModel):
    host: str = "0.0.0.0"  # noqa: S104 — bound to the container's LAN IP by compose
    port: int = Field(default=8080, ge=1, le=65535)


class LoggingConfig(StrictModel):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "console"] = "console"


def _yaml_settings(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        loaded = yaml.safe_load(fh)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path}: top level of the config file must be a mapping")
    return loaded


class _YamlSource(PydanticBaseSettingsSource):
    """Feeds config.yaml into the settings chain below environment variables."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError  # unused: the whole document is supplied via __call__

    def __call__(self) -> dict[str, Any]:
        return _yaml_settings(self._path)


class _OverridesSource(PydanticBaseSettingsSource):
    """Feeds the admin console's overrides (vault_ask/overrides.py) into the
    settings chain — below the environment (deployment topology must not be
    displaceable by a browser edit) but above config.yaml (the whole point of
    an override is to beat the shipped default without editing the image).
    """

    def __init__(self, settings_cls: type[BaseSettings], path: Path) -> None:
        super().__init__(settings_cls)
        self._path = path

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        raise NotImplementedError  # unused: the whole document is supplied via __call__

    def __call__(self) -> dict[str, Any]:
        return read_overrides(self._path)


#: Set by load_settings() before the model is constructed — module state is the
#: only way to parameterise a pydantic-settings source at class level.
_active_yaml_path: Path = Path(DEFAULT_CONFIG_FILE)
_active_overrides_path: Path = Path(DEFAULT_OVERRIDES_FILE)


class Settings(BaseSettings):
    """Fully-resolved application configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VAULTASK_",
        env_nested_delimiter="__",
        extra="forbid",
        env_file=".env",
        env_file_encoding="utf-8",
        dotenv_filtering="match_prefix",
    )

    corpus: CorpusConfig = Field(default_factory=CorpusConfig)
    sensitivity: SensitivityConfig = Field(default_factory=SensitivityConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    web: WebConfig = Field(default_factory=WebConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # --- Secrets: environment only, never YAML, never logged ----------------
    vault_couchdb_password: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    #: Gates the admin portal (vault_ask/api/admin.py). Unset means "closed",
    #: never "open to everyone" — the same fail-closed stance podcast-digest's
    #: own admin key takes.
    admin_api_key: SecretStr | None = None
    #: Search-provider keys. Environment-only like every other secret here:
    #: never in config.yaml, never in overrides.json, never settable from the
    #: admin console. The console reports whether each is *present* so a
    #: provider that cannot work is visibly unselectable.
    tavily_api_key: SecretStr | None = None
    brave_api_key: SecretStr | None = None

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            _OverridesSource(settings_cls, _active_overrides_path),
            _YamlSource(settings_cls, _active_yaml_path),
            file_secret_settings,
        )

    @model_validator(mode="after")
    def _vault_needs_credentials(self) -> Settings:
        if self.vault.couchdb_url and not self.vault_couchdb_password:
            raise ValueError(
                "vault.couchdb_url is set but VAULTASK_VAULT_COUCHDB_PASSWORD is unset"
            )
        return self


def load_settings(config_file: str | Path | None = None) -> Settings:
    """Build Settings from ``config_file`` (default: $VAULTASK_CONFIG_FILE or ./config.yaml)."""
    global _active_yaml_path, _active_overrides_path
    path = Path(config_file or os.environ.get("VAULTASK_CONFIG_FILE", DEFAULT_CONFIG_FILE))
    _active_yaml_path = path
    _active_overrides_path = Path(
        os.environ.get("VAULTASK_OVERRIDES_FILE", DEFAULT_OVERRIDES_FILE)
    )
    return Settings()


def overrides_path() -> Path:
    """The overrides.json path this process is using — set by load_settings().

    Read by the admin API (vault_ask/api/admin.py), which needs the path
    without a dependency on the Settings singleton it is itself validating
    changes against.
    """
    return _active_overrides_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (used by FastAPI dependencies)."""
    return load_settings()
