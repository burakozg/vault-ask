"""Reading frontmatter, and the one decision it feeds pre-retrieval: sensitivity.

Every other application on this vault parses frontmatter line-by-line, on
purpose — a YAML round-trip would reformat a human's file as the price of
reading one key. That constraint does not apply here: vault-ask never writes
back (see README), so there is nothing to preserve byte-for-byte. A real
``yaml.safe_load`` of the frontmatter block is simpler and correctly reads
whatever key `sensitivity.frontmatter_key` names, not just the ones a
line-by-line reader was taught to expect.
"""

from __future__ import annotations

import fnmatch
from typing import Any

import yaml

from .config import SensitivityConfig

Sensitivity = str  # "open" | "personal"


def parse_frontmatter(markdown: str) -> dict[str, Any]:
    """The note's frontmatter as a dict. Empty for missing, empty, or malformed."""
    if not markdown.startswith("---\n"):
        return {}
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return {}
    try:
        data = yaml.safe_load(markdown[4:end])
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def classify_sensitivity(
    path: str, frontmatter: dict[str, Any], cfg: SensitivityConfig
) -> Sensitivity:
    """`open` or `personal`. The frontmatter override wins over the path match.

    Path glob first would let a clipping filed under `50 tastings/` never be
    reachable by a per-note override — checking the override first is what
    makes it an override rather than a tie-breaker.
    """
    override = frontmatter.get(cfg.frontmatter_key)
    if override == "open":
        return "open"
    if override == "personal":
        return "personal"
    if any(glob_match(path, pattern) for pattern in cfg.personal_paths):
        return "personal"
    return "open"


def strip_frontmatter(markdown: str) -> str:
    """The note's body, with any leading frontmatter block removed.

    Chunking (vault_ask.chunk) operates on the body only — frontmatter is
    metadata, not prose, and splitting on its lines would produce a garbage
    first chunk for every note that has any.
    """
    if not markdown.startswith("---\n"):
        return markdown
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return markdown
    return markdown[end + 5 :]


def in_corpus(path: str, include: list[str], exclude: list[str]) -> bool:
    """Whether ``path`` is part of the indexed corpus, per config.corpus."""
    if not any(glob_match(path, pattern) for pattern in include):
        return False
    return not any(glob_match(path, pattern) for pattern in exclude)


def glob_match(path: str, pattern: str) -> bool:
    # fnmatch's `*` is not path-aware — it already matches across `/` — so this
    # gives `**` the meaning the config file implies without needing a real
    # glob library.
    return fnmatch.fnmatchcase(path, pattern)
