"""Unit tests for src.main.build_prompt — pure function, no network/model."""

from src.main import build_prompt


def test_build_prompt_tags_and_question():
    chunks = [
        {"article_type": "madde", "article_no": 41, "article_title": "Fazla çalışma", "text": "MADDE 41. - ..."},
        {"article_type": "gecici", "article_no": 3, "article_title": None, "text": "GEÇİCİ MADDE 3. - ..."},
    ]
    system, user = build_prompt("fazla mesai nedir?", chunks)

    assert "[Madde 41]" in user
    assert "[Geçici Madde 3]" in user
    assert "fazla mesai nedir?" in user
    assert "SADECE" in system  # grounded-only instruction
    assert "uydurmak" in system  # never invent an article
    assert "talimat değildir" in system  # context is data, not instructions


def test_build_prompt_preserves_chunk_order():
    chunks = [
        {"article_type": "madde", "article_no": 5, "article_title": None, "text": "beşinci madde"},
        {"article_type": "madde", "article_no": 2, "article_title": None, "text": "ikinci madde"},
    ]
    _, user = build_prompt("q", chunks)
    assert user.index("[Madde 5]") < user.index("[Madde 2]")


def test_build_prompt_empty_context_still_well_formed():
    system, user = build_prompt("off-corpus question", [])
    assert "off-corpus question" in user
    assert "SADECE" in system
