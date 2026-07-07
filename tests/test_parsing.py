"""Unit tests for src.parsing.split_articles on pasted text fixtures.

No PDF dependency — fixtures mirror the heading forms verified in the source PDF
plus the consolidated-text forms (EK MADDE, Mülga) we want to stay compatible with.
"""

from src.parsing import clean_text, split_articles

BASIC = """\
BİRİNCİ BÖLÜM
Genel Hükümler
Amaç ve kapsam
MADDE 1. - Bu Kanunun amacı işverenler ile işçilerin hak ve sorumluluklarını düzenlemektir.
Bu Kanun, 4 üncü maddedeki istisnalar dışında kalan bütün işyerlerine uygulanır.
Tanımlar
MADDE 2. - Bir iş sözleşmesine dayanarak çalışan gerçek kişiye işçi denir.
"""

GECICI = """\
Süre
MADDE 120. - Bir önceki maddenin metni burada biter.
GEÇİCİ MADDE  3. - 1475 sayılı Kanuna göre alınmış bulunan asgari ücret kararı,
yürürlükte kalır.
GEÇİCİ MADDE 6. – Kıdem tazminatı için bir kıdem tazminatı fonu kurulur.
"""

CONSOLIDATED = """\
EK MADDE 2 – (Ek: 20/4/2015-6645/35 md.) Mazeret izni bu maddede düzenlenir.
Askeri ve kanuni ödev
MADDE 34 – (Mülga: 12/9/2010-6009/48 md.)
"""

SECTIONS = """\
BİRİNCİ BÖLÜM
Genel Hükümler
MADDE 1. - Birinci bölümün maddesi.
İKİNCİ BÖLÜM
İş Sözleşmesi, Türleri ve Feshi
MADDE 8. - İkinci bölümün maddesi.
"""


def test_basic_split_titles_and_bodies():
    articles = split_articles(BASIC)
    assert [a.article_id for a in articles] == ["madde-1", "madde-2"]
    a1, a2 = articles
    assert a1.article_no == 1 and a1.article_type == "madde"
    assert a1.article_title == "Amaç ve kapsam"
    assert a1.section == "Genel Hükümler"
    assert a1.text.startswith("MADDE 1.")
    # body wraps onto the next line; next article's title is NOT in the body
    assert "bütün işyerlerine uygulanır" in a1.text
    assert "Tanımlar" not in a1.text
    assert a2.article_title == "Tanımlar"
    assert not a1.repealed and not a2.repealed


def test_gecici_madde_spacing_and_en_dash():
    articles = split_articles(GECICI)
    assert [a.article_id for a in articles] == ["madde-120", "gecici-3", "gecici-6"]
    g3, g6 = articles[1], articles[2]
    assert g3.article_type == "gecici" and g3.article_no == 3
    assert g3.article_title is None  # geçici articles have no title line
    assert "yürürlükte kalır." in g3.text
    assert g6.article_no == 6  # en-dash heading
    assert "kıdem tazminatı fonu" in g6.text


def test_consolidated_forms_ek_and_mulga():
    articles = split_articles(CONSOLIDATED)
    assert [a.article_id for a in articles] == ["ek-2", "madde-34"]
    ek, mulga = articles
    assert ek.article_type == "ek" and not ek.repealed
    assert mulga.repealed
    assert mulga.article_title == "Askeri ve kanuni ödev"


def test_section_tracking():
    articles = split_articles(SECTIONS)
    assert articles[0].section == "Genel Hükümler"
    assert articles[1].section == "İş Sözleşmesi, Türleri ve Feshi"


def test_clean_text_strips_rg_header():
    raw = " 10 Haziran 2003 Tarihli Resmi Gazete      Sayı: 25134 \nMADDE 1. - Metin.  "
    cleaned = clean_text(raw)
    assert "Resmi Gazete" not in cleaned
    assert cleaned == "MADDE 1. - Metin."
