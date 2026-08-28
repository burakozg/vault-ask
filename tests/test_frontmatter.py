from __future__ import annotations

from vault_ask.config import SensitivityConfig
from vault_ask.frontmatter import classify_sensitivity, in_corpus, parse_frontmatter

SENS = SensitivityConfig(
    personal_paths=["30 journal/**", "40 people/**"],
    frontmatter_key="sensitivity",
)


class TestParseFrontmatter:
    def test_no_frontmatter(self) -> None:
        assert parse_frontmatter("# just a heading\n\nbody text") == {}

    def test_basic(self) -> None:
        md = "---\ntitle: Anthropic\ntags: [ai, vendor]\n---\n\nbody"
        assert parse_frontmatter(md) == {"title": "Anthropic", "tags": ["ai", "vendor"]}

    def test_unterminated_block_is_empty(self) -> None:
        assert parse_frontmatter("---\ntitle: Anthropic\n\nno closing fence") == {}

    def test_malformed_yaml_is_empty_not_a_crash(self) -> None:
        md = "---\ntitle: [unclosed\n---\nbody"
        assert parse_frontmatter(md) == {}

    def test_non_mapping_frontmatter_is_empty(self) -> None:
        md = "---\n- just\n- a\n- list\n---\nbody"
        assert parse_frontmatter(md) == {}


class TestClassifySensitivity:
    def test_default_open(self) -> None:
        assert classify_sensitivity("10 raw/Anthropic.md", {}, SENS) == "open"

    def test_personal_path(self) -> None:
        assert classify_sensitivity("30 journal/2026-08-28.md", {}, SENS) == "personal"

    def test_frontmatter_override_wins_over_path(self) -> None:
        # A note filed under a personal path but marked open in its own frontmatter.
        fm = {"sensitivity": "open"}
        assert classify_sensitivity("40 people/Alex.md", fm, SENS) == "open"

    def test_frontmatter_override_the_other_direction(self) -> None:
        fm = {"sensitivity": "personal"}
        assert classify_sensitivity("10 raw/Anthropic.md", fm, SENS) == "personal"

    def test_invalid_override_value_falls_back_to_path_rule(self) -> None:
        fm = {"sensitivity": "top-secret"}
        assert classify_sensitivity("10 raw/Anthropic.md", fm, SENS) == "open"


class TestInCorpus:
    def test_included(self) -> None:
        assert in_corpus("10 raw/Anthropic.md", ["**/*.md"], [])

    def test_excluded_inbox(self) -> None:
        assert not in_corpus(
            "00 inbox/draft.md", ["**/*.md"], ["00 inbox/**", "**/.trash/**"]
        )

    def test_excluded_trash_anywhere(self) -> None:
        assert not in_corpus(
            "99 topics/.trash/old.md", ["**/*.md"], ["00 inbox/**", "**/.trash/**"]
        )

    def test_non_markdown_excluded_by_include(self) -> None:
        assert not in_corpus("attachments/photo.png", ["**/*.md"], [])
