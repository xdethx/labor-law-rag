"""Unit tests for src.contracts — PDF text extraction + clause splitting.
Pure functions, no Qdrant/model/network (encryption test uses pypdf's own
in-memory PdfWriter, no network involved).
"""

from io import BytesIO

import pytest
from pypdf import PdfWriter

from src.contracts import EncryptedPDFError, extract_contract_text, split_clauses


def test_split_by_numbered_heading():
    text = (
        "İŞ SÖZLEŞMESİ\n"
        "1. Taraflar\n"
        "İşbu sözleşme A ile B arasında akdedilmiştir.\n"
        "2. Deneme süresi\n"
        "Deneme süresi 4 aydır.\n"
        "3. Ücret\n"
        "Aylık brüt ücret 40.000 TL'dir.\n"
    )
    clauses = split_clauses(text)

    assert [c["clause_no"] for c in clauses] == [1, 2, 3]
    assert "Deneme süresi 4 aydır" in clauses[1]["text"]
    assert "Ücret" in clauses[2]["text"]


def test_split_by_madde_style_heading():
    text = (
        "MADDE 1 - Taraflar\n"
        "Bu sözleşme aşağıdaki taraflar arasında yapılmıştır.\n"
        "MADDE 2 - Çalışma süresi\n"
        "Haftalık çalışma süresi 45 saattir.\n"
    )
    clauses = split_clauses(text)

    assert len(clauses) == 2
    assert clauses[0]["clause_no"] == 1
    assert "45 saattir" in clauses[1]["text"]


def test_split_falls_back_to_paragraphs_when_too_few_headings():
    text = (
        "Bu bir giriş paragrafıdır, numaralı madde yoktur.\n"
        "\n"
        "İkinci paragraf burada yer almaktadır ve ayrı bir fikir içerir.\n"
        "\n"
        "Üçüncü ve son paragraf.\n"
    )
    clauses = split_clauses(text)

    assert len(clauses) == 3
    assert clauses[0]["clause_no"] == 1
    assert "giriş paragrafıdır" in clauses[0]["text"]
    assert "Üçüncü ve son paragraf" in clauses[2]["text"]


def test_split_drops_empty_clauses():
    text = "1. İlk madde\nMetin var.\n\n\n2. İkinci madde\nBaşka metin.\n"
    clauses = split_clauses(text)
    assert all(c["text"].strip() for c in clauses)


def test_extract_contract_text_returns_text_and_page_count():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    buf = BytesIO()
    writer.write(buf)

    text, page_count = extract_contract_text(buf.getvalue())

    assert page_count == 2
    assert isinstance(text, str)


def test_extract_contract_text_raises_on_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt(user_password="secret")
    buf = BytesIO()
    writer.write(buf)

    with pytest.raises(EncryptedPDFError):
        extract_contract_text(buf.getvalue())
