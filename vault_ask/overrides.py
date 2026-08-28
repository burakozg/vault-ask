"""Console-editable configuration overrides — read by `vault_ask.config` (as a
settings source, ranked below the environment but above `config.yaml`) and
written by the admin API (`vault_ask/api/admin.py`).

Deliberately a plain JSON file, not a table in `index.sqlite`: `config.py`
must not depend on `db.py` to load settings (that would invert the dependency
direction everywhere else in this codebase — config is what db.py itself
depends on), and a file is simpler than opening a second connection just to
read one small blob before the real database connection exists.

**Applies on restart, not live** — same reasoning as podcast-digest's
settings_store.py: `Settings` is built once, at process start, and treating it
as hot-reloadable would need every reader (`app.state.cfg`, closures already
holding a reference to it) to notice a change mid-request. The admin page
reports what is pending vs what is currently active rather than pretending an
edit took effect immediately.

Only a deliberate subset of fields is overridable — see `OVERRIDABLE_KEYS`.
Deployment topology (`vault.*`, `models.embedding*`, `api.*`, `index.*`) and
every secret stay file/environment-only: a typo in a browser form must not be
able to point the vault connection somewhere wrong or lock the container out
of its own port.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("vault_ask.overrides")

#: section -> the keys within it a console may override. Everything else in
#: that section (deployment topology, machine-protecting limits) stays
#: file/environment-only.
OVERRIDABLE_KEYS: dict[str, frozenset[str]] = {
    # rerank is excluded: no rerank step is wired into the pipeline yet
    # (README "Build order") — exposing a knob that currently does nothing
    # would be a UI that lies about what changing it does.
    "models": frozenset({"generation"}),
    # The kill switch for the one component that talks to the outside world.
    # Thresholds stay file-only: they were measured (see WebConfig) and a
    # browser is the wrong place to nudge a number that decides when the vault
    # stops being the only source.
    "web": frozenset({"enabled", "max_results"}),
    # rerank_top_k is excluded for the same reason as models.rerank above.
    "retrieval": frozenset(
        {
            "vector_top_k",
            "fts_top_k",
            "fusion_top_k",
            "graph_enabled",
            "graph_max_siblings",
            "graph_discount",
            "graph_max_slots",
            "final_top_k",
        }
    ),
}


def read_overrides(path: Path) -> dict[str, dict[str, Any]]:
    """The stored overrides, or `{}` if the file does not exist yet — a fresh
    deploy has none, and that is not an error.

    **Filtered against OVERRIDABLE_KEYS on the way in**, not only on the way
    out. The admin API validates what it writes, but that only constrains this
    file while the API is the sole writer — and the guarantee this module
    claims ("a typo in a browser form must not be able to point the vault
    connection somewhere wrong") has to hold for a hand-edited file too, since
    it is bind-mounted next to the index and is the obvious thing to reach for
    when the console is unreachable. Anything outside the allowlist is dropped
    with a warning rather than honoured, so an unknown key can never reach
    `Settings` through this path.

    Malformed content is likewise dropped rather than raised: this is called
    while *building* Settings, so raising here means the process cannot start
    at all — a hand-edit typo that bricks boot is a worse failure than one
    that is ignored loudly.
    """
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        try:
            loaded = json.load(fh)
        except json.JSONDecodeError as exc:
            log.error("overrides.unreadable path=%s error=%s — ignoring the file", path, exc)
            return {}
    if not isinstance(loaded, dict):
        log.error("overrides.not_an_object path=%s — ignoring the file", path)
        return {}

    filtered: dict[str, dict[str, Any]] = {}
    for section, values in loaded.items():
        allowed = OVERRIDABLE_KEYS.get(section)
        if allowed is None:
            log.warning("overrides.section_not_overridable section=%r — ignored", section)
            continue
        if not isinstance(values, dict):
            log.warning("overrides.section_not_an_object section=%r — ignored", section)
            continue
        kept = {k: v for k, v in values.items() if k in allowed}
        for key in set(values) - allowed:
            log.warning("overrides.key_not_overridable key=%s.%s — ignored", section, key)
        if kept:
            filtered[section] = kept
    return filtered


def write_overrides(path: Path, overrides: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(overrides, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)  # atomic on the same filesystem — no reader ever sees a half-written file
