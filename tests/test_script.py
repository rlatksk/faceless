"""Checks for script parsing.

Providers wrap JSON in markdown fences often enough that it broke a real run,
so the tolerant parse gets a check.
"""

import pytest

from pipeline.script import _parse_json


class TestParseJson:
    def test_plain_json(self):
        assert _parse_json('{"a": 1}', "Kenari") == {"a": 1}

    def test_json_in_a_markdown_fence(self):
        """What deepseek-v4-1-flash actually returns on Kenari."""
        content = '```json\n{"narration": "hi", "image_prompts": ["a"]}\n```'
        assert _parse_json(content, "Kenari")["narration"] == "hi"

    def test_bare_fence_without_a_language(self):
        assert _parse_json('```\n{"a": 1}\n```', "Kenari") == {"a": 1}

    def test_surrounding_whitespace(self):
        assert _parse_json('  \n{"a": 1}\n  ', "Kenari") == {"a": 1}

    def test_character_sheet_survives_the_round_trip(self):
        content = ('```json\n{"narration": "n", "characters": [{"name": "Mother", '
                   '"description": "36, dark brown hair"}], "image_prompts": ["p"]}\n```')
        out = _parse_json(content, "Kenari")
        assert out["characters"][0]["name"] == "Mother"

    def test_unparseable_content_names_the_provider(self):
        with pytest.raises(RuntimeError, match="Kenari returned invalid JSON"):
            _parse_json("I'm sorry, I can't help with that.", "Kenari")
