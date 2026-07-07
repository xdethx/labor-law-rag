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


def test_build_prompt_tags_mixed_law_and_contract_chunks():
    chunks = [
        {"article_type": "madde", "article_no": 15, "article_title": "Deneme süresi", "text": "MADDE 15. - ..."},
        {"clause_no": 4, "text": "Deneme süresi dört aydır.", "source": "contract"},
    ]
    system, user = build_prompt("deneme süresi kanuna uygun mu?", chunks)

    assert "[Madde 15]" in user
    assert "[Sözleşme 4]" in user
    assert "Sözleşme" in system  # system prompt explains the mixed-source case
