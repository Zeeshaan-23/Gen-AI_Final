"""
Unit tests for src/prompts.py prompt builder and presets.
"""

from src.prompts import FURNITURE_PRESETS, build_prompt, DEFAULT_NEGATIVE_PROMPT


def test_all_furniture_presets():
    """Verifies that all 12 FURNITURE_PRESETS build non-empty positive and negative prompts."""
    assert len(FURNITURE_PRESETS) == 12

    for preset_key, preset_data in FURNITURE_PRESETS.items():
        part_text = preset_data["part_text"]
        change = preset_data["change"]

        pos, neg = build_prompt(part_text, change)

        assert isinstance(pos, str) and len(pos) > 10
        assert isinstance(neg, str) and len(neg) > 10

        # Positive prompt must contain part and either color or material
        assert part_text in pos
        color = change.get("color", "")
        material = change.get("material", "")
        assert color in pos or material in pos

        # Negative prompt must match default negative keywords
        assert "blurry" in neg
        assert "distorted" in neg


def test_build_prompt_empty_part():
    """Verifies that passing an empty part string falls back to 'furniture component'."""
    change = {"color": "emerald green", "material": "velvet"}
    pos, neg = build_prompt("", change)

    assert "furniture component" in pos
    assert "emerald green" in pos
    assert "velvet" in pos
    assert neg == DEFAULT_NEGATIVE_PROMPT
