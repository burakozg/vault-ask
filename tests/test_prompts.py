"""Pins the properties of the answering contract that actually matter
(README "Answering contract") — not a snapshot of the exact wording, which is
free to change, but the guarantees a wording change must not lose.
"""

from __future__ import annotations

from vault_ask.prompts import SYSTEM_PROMPT


def test_citations_are_required_and_path_qualified() -> None:
    assert "[[path/to/note|Note Title]]" in SYSTEM_PROMPT
    assert "never invent, guess, or abbreviate a path" in SYSTEM_PROMPT


def test_citations_are_not_required_after_every_single_sentence() -> None:
    """Regression: the model was repeating the identical [[wikilink]] after
    almost every sentence of a multi-paragraph answer that drew from one note
    — technically compliant with "every claim needs a citation" but unreadable.
    """
    assert "cite economically" in SYSTEM_PROMPT.lower()
    assert "do not repeat" in SYSTEM_PROMPT.lower()


def test_web_and_vault_content_stay_separated() -> None:
    assert "cite it by url" in SYSTEM_PROMPT.lower()
    assert "under its own \nheading" in SYSTEM_PROMPT or "own heading" in SYSTEM_PROMPT


def test_vault_silence_is_a_valid_answer() -> None:
    assert "The vault is silent on this" in SYSTEM_PROMPT


def test_fabrication_is_forbidden() -> None:
    assert "never fabricate" in SYSTEM_PROMPT.lower()


def test_web_citations_are_forbidden_from_using_wikilink_syntax() -> None:
    """Caught in live testing, not in review.

    With web fallback on, the model produced
    `[[Ulaanbaatar - Wikipedia](https://en.wikipedia.org/...)]` — wikilink
    brackets wrapped around a markdown link, for a *web* source. Rule 2 said
    "cite it by URL, never as a wikilink" and that was not specific enough to
    stop syntax blending. `[[` is the vault's marker: using it for a web page
    tells the reader they have a note they do not have.
    """
    assert "NEVER start with `[[`" in SYSTEM_PROMPT
    assert "markdown link" in SYSTEM_PROMPT
