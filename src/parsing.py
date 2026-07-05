"""Parse the Labor Law PDF (İş Kanunu No. 4857) into structured articles.

Offline step, run manually:  python -m src.parsing
Reads  data/raw/is_kanunu.pdf  and writes  data/processed/articles.json.

The chunking unit is one whole article (madde) — the natural, citable legal unit.
Heading forms verified against the source PDF (original 10 June 2003 RG text):
    "MADDE 1. - ", "MADDE 4.- ", "GEÇİCİ MADDE  3. - ", "GEÇİCİ MADDE 6. – "
The regex also accepts "EK MADDE n" and the no-period consolidated form
("MADDE 12 –") so a future switch to the consolidated text stays cheap.
"""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pypdf import PdfReader

RAW_PDF_PATH = Path("data/raw/is_kanunu.pdf")
OUTPUT_PATH = Path("data/processed/articles.json")

# Body continues on the same line after the dash (hyphen or en dash).
HEADING_RE = re.compile(r"^\s*(?:(GEÇİCİ|EK)\s+)?MADDE\s+(\d+)\s*\.?\s*[-–]\s*(.*)$")

# "BİRİNCİ BÖLÜM" etc. — ordinal word(s) in caps, then BÖLÜM; section name follows
# on the next non-empty line.
SECTION_RE = re.compile(r"^[A-ZÇĞİÖŞÜ]+\s+BÖLÜM$")

# Lines like "10 Haziran 2003 Tarihli Resmi Gazete   Sayı: 25134" (page-1 header).
NOISE_RE = re.compile(r"Tarihli\s+Resm[iî]\s+Gazete", re.IGNORECASE)

ARTICLE_TYPE_PREFIX = {"": "madde", "GEÇİCİ": "gecici", "EK": "ek"}


class Article(BaseModel):
    article_id: str  # "madde-1" | "gecici-1" | "ek-1"
    article_no: int
    article_type: Literal["madde", "gecici", "ek"]
    article_title: str | None  # heading line above the article, e.g. "Tanımlar"
    section: str | None  # BÖLÜM name, e.g. "Genel Hükümler"
    repealed: bool
    text: str  # full article text, starting at the MADDE heading


def extract_text(pdf_path: Path) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join(page.extract_text() for page in reader.pages)


def clean_text(raw: str) -> str:
    lines = [line.rstrip() for line in raw.splitlines()]
    return "\n".join(line for line in lines if not NOISE_RE.search(line))


def _is_title_line(line: str) -> bool:
    """A title is a short standalone heading ("Amaç ve kapsam"), not a wrapped
    body line or list item — those end with sentence/list punctuation."""
    return 0 < len(line) <= 80 and line[-1] not in ".,;:"


def split_articles(text: str) -> list[Article]:
    articles: list[Article] = []
    current: Article | None = None
    body_lines: list[str] = []
    section: str | None = None
    expecting_section_name = False

    def finalize() -> None:
        nonlocal current
        if current is None:
            return
        # The next article's title line, if any, was buffered here — drop it.
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        current.text = "\n".join(body_lines).strip()
        articles.append(current)
        current = None
        body_lines.clear()

    for line in text.splitlines():
        stripped = line.strip()

        if expecting_section_name:
            if stripped:
                section = stripped
                expecting_section_name = False
            continue

        if SECTION_RE.match(stripped):
            expecting_section_name = True
            continue

        heading = HEADING_RE.match(line)
        if heading:
            prefix, no, rest = heading.group(1) or "", heading.group(2), heading.group(3)
            title = None
            if body_lines and _is_title_line(body_lines[-1].strip()):
                title = body_lines.pop().strip()
            finalize()
            article_type = ARTICLE_TYPE_PREFIX[prefix]
            current = Article(
                article_id=f"{article_type}-{no}",
                article_no=int(no),
                article_type=article_type,
                article_title=title,
                section=section,
                repealed=rest.lstrip().startswith("(Mülga"),
                text="",
            )
            body_lines.append(line.strip())
            continue

        if current is not None:
            body_lines.append(line)
        elif stripped and _is_title_line(stripped):
            # Possible title of the upcoming first article (preamble lines
            # ending with punctuation are ignored).
            body_lines = [stripped]

    finalize()
    return articles


def main() -> None:
    text = clean_text(extract_text(RAW_PDF_PATH))
    articles = split_articles(text)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps([a.model_dump() for a in articles], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    counts = {t: sum(1 for a in articles if a.article_type == t) for t in ("madde", "gecici", "ek")}
    repealed = sum(1 for a in articles if a.repealed)
    print(f"Parsed {len(articles)} articles -> {OUTPUT_PATH}")
    print(f"  madde: {counts['madde']}, gecici: {counts['gecici']}, ek: {counts['ek']}, repealed: {repealed}")
    print(f"  first: {articles[0].article_id}, last: {articles[-1].article_id}")


if __name__ == "__main__":
    main()
