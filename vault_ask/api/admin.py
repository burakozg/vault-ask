"""Admin console API: read/write vault-ask's console-editable configuration.

Overrides apply at the next restart, not immediately — ``Settings`` is built
once at process start (``vault_ask.config.load_settings``), and every reader
holding a reference to it (``app.state.cfg``, the retrieval defaults baked
into a running request) would need to notice a change mid-request for a live
update to mean anything. This endpoint reports what is *stored* in
overrides.json versus what this process actually *booted* with, rather than
pretending an edit took effect immediately — see ``pending_restart`` below.

A write is validated by merging it onto the process's currently active values
for the fields that stay file/environment-only, then re-running the same
pydantic model (``ModelsConfig`` / ``RetrievalConfig``) config.py itself uses
— so a change that would fail validation (e.g. ``retrieval.rerank_top_k <
final_top_k``, checked against the active, non-overridable ``rerank_top_k``)
is rejected here, not discovered as a startup crash after a restart.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import GENERATION_SUGGESTIONS, Settings, overrides_path
from ..overrides import OVERRIDABLE_KEYS, read_overrides, write_overrides
from ..web import PROVIDER_ENV, PROVIDER_KEYS, provider_available
from .admin_auth import require_api_key

log = logging.getLogger("vault_ask.api.admin")

router = APIRouter(prefix="/admin/config", dependencies=[Depends(require_api_key)])


class ConfigIn(BaseModel):
    """Everything the console may change. Omitted sections are left alone."""

    model_config = ConfigDict(extra="forbid")

    models: dict[str, Any] | None = None
    retrieval: dict[str, Any] | None = None
    web: dict[str, Any] | None = None


def _cfg(request: Request) -> Settings:
    return request.app.state.cfg  # type: ignore[no-any-return]


def _boot_overrides(request: Request) -> dict[str, dict[str, Any]]:
    return request.app.state.overrides_at_boot  # type: ignore[no-any-return]


def _readable(exc: ValidationError) -> str:
    """Config errors should read like the startup message, not a stack trace."""
    parts = []
    for error in exc.errors():
        location = ".".join(str(p) for p in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts[:6])


def _validate_section(section: str, active: BaseModel, override: dict[str, Any]) -> None:
    unknown = set(override) - OVERRIDABLE_KEYS[section]
    if unknown:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"{section}: not overridable from the console: {', '.join(sorted(unknown))}. "
                f"Editable keys are {', '.join(sorted(OVERRIDABLE_KEYS[section]))}."
            ),
        )
    # Merged onto the *active* values (not the shipped defaults) so a field
    # this console cannot edit — models.embedding, retrieval.rerank_top_k —
    # is checked against what is really running, e.g. the cross-field
    # rerank_top_k >= final_top_k rule in RetrievalConfig.
    merged = {**active.model_dump(mode="json"), **override}
    try:
        type(active)(**merged)
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=_readable(exc)
        ) from exc


@router.get("", summary="Editable configuration, as this process is running it")
async def read_config(request: Request) -> dict[str, Any]:
    cfg = _cfg(request)
    stored = read_overrides(overrides_path())
    return {
        "models": {
            key: {
                "active": getattr(cfg.models, key),
                "override": stored.get("models", {}).get(key),
            }
            for key in sorted(OVERRIDABLE_KEYS["models"])
        },
        "retrieval": {
            key: {
                "active": getattr(cfg.retrieval, key),
                "override": stored.get("retrieval", {}).get(key),
            }
            for key in sorted(OVERRIDABLE_KEYS["retrieval"])
        },
        "web": {
            key: {
                "active": getattr(cfg.web, key),
                "override": stored.get("web", {}).get(key),
            }
            for key in sorted(OVERRIDABLE_KEYS["web"])
        },
        # Choices the page offers, so the UI has no list of its own to drift
        # from. `available` is what makes an unusable provider visibly
        # unselectable instead of a save that quietly returns nothing.
        "web_providers": [
            {
                "name": name,
                "available": provider_available(cfg, name),
                "needs_key": PROVIDER_KEYS[name] is not None,
                "env_var": PROVIDER_ENV.get(name),
            }
            for name in PROVIDER_KEYS
        ],
        "generation_suggestions": list(GENERATION_SUGGESTIONS),
        "editable_keys": {k: sorted(v) for k, v in OVERRIDABLE_KEYS.items()},
        "pending_restart": stored != _boot_overrides(request),
    }


@router.put("", summary="Replace the console-editable configuration")
async def write_config(request: Request, body: Annotated[ConfigIn, Body()]) -> dict[str, Any]:
    cfg = _cfg(request)
    stored = read_overrides(overrides_path())
    new_overrides: dict[str, dict[str, Any]] = {k: dict(v) for k, v in stored.items()}

    if body.models is not None:
        _validate_section("models", cfg.models, body.models)
        new_overrides["models"] = {**new_overrides.get("models", {}), **body.models}
    if body.retrieval is not None:
        _validate_section("retrieval", cfg.retrieval, body.retrieval)
        new_overrides["retrieval"] = {**new_overrides.get("retrieval", {}), **body.retrieval}
    if body.web is not None:
        _validate_section("web", cfg.web, body.web)
        chosen = body.web.get("provider")
        if chosen is not None and not provider_available(cfg, str(chosen)):
            # Rejected here rather than saved and discovered after a restart,
            # when the only symptom would be web results silently vanishing.
            # The key itself is not settable from here by design — secrets are
            # environment-only (vault_ask/overrides.py).
            env = PROVIDER_ENV.get(str(chosen), "its API key")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    f"{chosen!r} has no API key configured, so selecting it would "
                    f"silently return no results. Set {env} in .env and redeploy, "
                    "then choose it here."
                ),
            )
        new_overrides["web"] = {**new_overrides.get("web", {}), **body.web}

    write_overrides(overrides_path(), new_overrides)
    log.info("admin.config_updated sections=%s", sorted(new_overrides))
    return {
        "saved": True,
        "pending_restart": new_overrides != _boot_overrides(request),
        "detail": "Saved. Restart vault-ask to apply.",
    }


@router.delete("", summary="Discard all overrides and return to config.yaml")
async def reset_config(request: Request) -> dict[str, Any]:
    write_overrides(overrides_path(), {})
    log.info("admin.config_reset")
    return {
        "saved": True,
        "pending_restart": _boot_overrides(request) != {},
        "detail": "Overrides cleared. Restart vault-ask to return to config.yaml.",
    }

